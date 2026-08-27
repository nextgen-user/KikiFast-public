import asyncio
import time
from collections import deque
from tools_and_config.config_loader import get_full_config
from core.stt import STTEngine

async def face_event_listener(
    message_history: list,
    stt: STTEngine,
    idle_mgr,
    loop: asyncio.AbstractEventLoop,
    stt_queue: asyncio.Queue,
    face_history=None,
    gesture_event_handler=None,
):
    """Listen for controller events and route face/hand events asynchronously."""
    
    while True: # Robust retry loop
        config = get_full_config()
        face_config = config.get("face_events", {})

        face_events_enabled = bool(face_config.get("enabled", False))
        if not face_events_enabled and gesture_event_handler is None:
            print("[Face] Face events disabled in config. Checking again in 60s...")
            await asyncio.sleep(60)
            continue

        # Rolling window of injection timestamps for the N-per-5-min rate limit.
        face_event_timestamps = deque()
        max_injections = int(face_config.get("max_injections_per_5min", 2))

        try:
            from kiki_control_client import KikiController
            ctrl_config = config.get("controller", {})
            controller = KikiController(host=ctrl_config.get("host", "192.0.2.20"))

            connected = await controller.connect()
            if not connected:
                print("[Face] Failed to connect to KikiController. Retrying in 10s...")
                await asyncio.sleep(10)
                continue

            print("[Face] Connected to KikiController, listening for face events...")

            # face_detected / unknown_face_enrolled are ONE-SHOT PUB messages: any
            # that fired before we subscribed are gone. So on (re)connect, ask the
            # server who is in front RIGHT NOW and inject them, so Kiki knows who
            # it's talking to immediately at startup instead of waiting for the
            # person to leave and reappear.
            try:
                from core.llm import hot_inject
                active = (
                    await controller.get_active_faces()
                    if face_events_enabled
                    else []
                )
                from core.runtime_controls import context_enabled
                for f in active:
                    nm = f.get("name", "Unknown")
                    if not nm or nm == "Unknown":
                        continue
                    print(f"[Face] Startup sync: '{nm}' is already in view.")
                    # Face TRACKING and auto-enrolment still run; only the
                    # prompt injection is suppressed, because "in front of
                    # Kiki" is not something a roleplay character can hear
                    # without dropping the character.
                    if context_enabled("face_events"):
                        hot_inject(message_history, {
                            "role": "system",
                            "content": f"[System: '{nm}' is currently here in front of Kiki.]"})
                    from robot.stranger_enroll import maybe_ask_real_name, ensure_face_thumb, is_temp_identity
                    if is_temp_identity(nm):
                        try:
                            _t = await ensure_face_thumb(nm, controller)
                            from core.oled_display import oled_manager
                            oled_manager.show_face(nm, image=_t or None, subtitle="who are you?")
                        except Exception:
                            pass
                    maybe_ask_real_name(nm, stt_queue, idle_mgr)
            except Exception as e:
                print(f"[Face] Startup face sync failed: {e}")

            async for event in controller.listen_events():
                print(f"[Face] Received event from controller: {event}")
                event_type = event.get("event")

                # Control gestures must bypass face-event injection/rate limits.
                # The handler is synchronous and non-blocking; it only flips
                # local events, stops output processes, and schedules state
                # cleanup on the main loop.
                if event_type == "hand_gesture":
                    if gesture_event_handler is not None:
                        try:
                            gesture_event_handler(event)
                        except Exception as e:
                            print(f"[Gesture] Handler error: {e}")
                    continue

                # A gesture subscriber still needs the shared controller PUB
                # socket even when conversational face events are disabled.
                if not face_events_enabled:
                    continue

                # --- Stranger auto-enroll events (handled in the background) ---
                if event_type == "unknown_face_enrolled":
                    print(f"[Face] → dispatching process_enrolled_event for '{event.get('temp_name')}'")
                    from robot.stranger_enroll import process_enrolled_event
                    asyncio.create_task(
                        process_enrolled_event(event, controller, stt_queue, idle_mgr))
                    continue
                if event_type == "crowd_detected":
                    from robot.stranger_enroll import show_crowd_notice
                    show_crowd_notice()
                    continue
                if event_type in ("person_renamed", "person_merged"):
                    print(f"[Face] {event_type}: {event}")
                    continue

                if event_type in ["face_detected", "face_lost"]:
                    person_name = event.get("person", event.get("name", "Unknown"))
                    print(f"[Face] Processing {event_type} for {person_name}")
                    now = time.time()

                    # Record face event in shared history buffer (for Workers)
                    if face_history is not None:
                        face_history.record_face(
                            person_name,
                            "detected" if event_type == "face_detected" else "lost"
                        )

                    # Fire face_detected workers
                    if event_type == "face_detected":
                        try:
                            from core.workers.worker_manager import get_worker_manager
                            manager = get_worker_manager()
                            asyncio.create_task(
                                manager.fire_event("face_detected", person=person_name)
                            )
                        except Exception as e:
                            print(f"[Face] Error firing face_detected workers: {e}")

                    # Pending-stranger handling runs BEFORE the injection rate
                    # limit — asking a newly-met person their name must not be
                    # dropped just because a couple of face events already fired.
                    # Also catches a recognised 'guest_...' tag whose enroll event
                    # was missed (maybe_ask_real_name stubs a pending record).
                    if event_type == "face_detected" and person_name != "Unknown":
                        try:
                            from robot.stranger_enroll import (
                                maybe_ask_real_name, face_thumb_path, is_temp_identity)
                            from core.brain.knowledge_base import get_knowledge_base
                            _p = get_knowledge_base().get_person(person_name)
                            _temp = is_temp_identity(person_name)
                            print(f"[Face] pending-check '{person_name}': "
                                  f"in_kb={_p is not None} pending={(_p or {}).get('pending_real_name')} is_temp={_temp}")
                            if (_p and _p.get("pending_real_name")) or _temp:
                                try:
                                    from robot.stranger_enroll import ensure_face_thumb
                                    _thumb = await ensure_face_thumb(person_name, controller)
                                    from core.oled_display import oled_manager
                                    oled_manager.show_face(
                                        person_name,
                                        image=_thumb or (_p or {}).get("face_thumb"),
                                        subtitle="who are you?")
                                except Exception as e:
                                    print(f"[Face] show_face failed: {e}")
                                maybe_ask_real_name(person_name, stt_queue, idle_mgr)
                        except Exception as e:
                            print(f"[Face] pending-name check failed: {e}")

                    # Clean up timestamps older than 5 minutes (300 seconds)
                    while face_event_timestamps and now - face_event_timestamps[0] > 300:
                        face_event_timestamps.popleft()

                    if len(face_event_timestamps) >= max_injections:
                        print(f"[Face] Rate limit ({max_injections} msgs / 5 mins) exceeded. Skipping injection.")
                        continue

                    face_event_timestamps.append(now)

                    if event_type == "face_detected":
                        if person_name != "Unknown":
                            # A little happy hop on the OLED to greet a known face
                            # (one-shot flourish that auto-reverts).
                            try:
                                from core.oled_display import oled_manager
                                oled_manager.play_oneshot("happy")
                            except Exception:
                                pass
                            # Notify the Unified Idle Mind of user activity.
                            if idle_mgr is not None and idle_mgr.is_thinking:
                                idle_mgr.interrupt(reason=f"face:{person_name}")
                                print(f"[Face] Known face {person_name} woke robot from thinking mode")
                                # Trigger a vision update immediately to start the proactive chain
                                await stt_queue.put(("face_wake", person_name))
                            elif idle_mgr is not None:
                                idle_mgr.mark_activity()

                        msg = f"[System: The known person '{person_name}' has just appeared in front of Kiki.]"
                    else:
                        msg = f"[System: The known person '{person_name}' has left Kiki's view.]"

                    from core.runtime_controls import context_enabled
                    if not context_enabled("face_events"):
                        print("[Face] Injection suppressed (roleplay mode)")
                        continue

                    print(f"[Face] Injecting message: {msg}")

                    # Inject into message history as a system event AND hot-load
                    # (prompt-cache) it so the next spoken turn is a cache hit.
                    from core.llm import hot_inject
                    hot_inject(message_history, {"role": "system", "content": msg})

        except ImportError:
            print("[Face] kiki_control_client not available")
            break # No point retrying if file is missing
        except Exception as e:
            print(f"[Face] Event listener error: {e}. Retrying in 5s...")
            await asyncio.sleep(5)
