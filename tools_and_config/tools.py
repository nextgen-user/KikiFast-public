"""
Tool definitions and handlers for KikiFast voice assistant.
All tools from KIKI-SMART ported as standalone functions (no LiveKit).

Each tool has:
  1. An async handler function
  2. An OpenAI function-calling schema in the TOOLS list
  3. A sync wrapper in _TOOL_HANDLERS for the LLM tool-calling loop
"""

import asyncio
import json
import os
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

from tools_and_config.config_loader import get_full_config
import asyncio
from kiki_control_client import quick_command
from core.observability import get_recorder
from core.gesture_controls import (
    activity_generation,
    activity_was_stopped,
    is_output_muted,
)

_skip_followup = False


def _obs_trunc(x, n=800):
    """Cap a value for inclusion in an observability event (never raises)."""
    try:
        s = x if isinstance(x, str) else json.dumps(x, default=str)
    except Exception:
        s = str(x)
    return s if len(s) <= n else s[:n] + f"…[+{len(s) - n} chars]"

def should_skip_followup() -> bool:
    """Consume the flag — returns True once then resets."""
    global _skip_followup
    val = _skip_followup
    _skip_followup = False
    return val

def set_neck_active(state: bool):
    """
    Sync NECK-tracking power toggle (formerly the mis-named `set_motor_relay`).

    Never controlled chassis wheels — enables/disables the controller's autonomous
    neck-tracking (`neck_movement` on/off). Safe to call from synchronous background
    threads. (Chassis tools that used to call this are parked in to_do/chassis_tools.py.)
    """
    command = "on" if state else "off"
    host = get_full_config().get("controller", {}).get("host", "192.0.2.20")
    try:
        asyncio.run(quick_command(host=host, neck_movement=command))
        print(f"[Neck] Neck tracking set to {command.upper()} via KikiController")
    except Exception as e:
        print(f"[Neck] Warning: Could not set neck tracking via KikiController: {e}")


_exa_client = None
_controller = None
_controller_lock = asyncio.Lock()

# NOTE: The chassis motor client (KikiMotorClient / VALID_MOTOR_ACTIONS / :5557) and the
# move/dance/follow_me tools were parked in to_do/chassis_tools.py when Kiki became a
# stationary neck-only unit. Kiki's only motion now is neck rotation — see robot/neck.py
# (expressive tags) and track_person (gaze). Re-add from to_do/ if wheels return.


# Web-search bound. The Exa SDK runs a blocking `requests` call with no timeout
# of its own, so on a flaky/dead network it hangs at the socket level. Two layers
# protect the speaking turn:
#   (1) _patch_exa_request_timeout below injects a real socket timeout into Exa's
#       requests calls so the underlying thread actually dies instead of hanging
#       forever (otherwise the orphaned worker thread leaks and never frees);
#   (2) search_web runs the call on a DEDICATED executor (not the event loop's
#       default one) so asyncio.run()'s teardown never joins a still-hung thread.
#       This was the real cause of the "tool execution timed out after 15s": the
#       12s asyncio.wait_for fired, but asyncio.run() then blocked in
#       shutdown_default_executor() joining the orphaned exa.search thread, so the
#       result Event was never set and main.py's 15s ceiling tripped.
_SEARCH_TIMEOUT_S = 10.0       # HARD wall-clock bound, well below main.py's 15s
_SEARCH_HTTP_TIMEOUT = (4.0, 6.0)  # (connect, read) for the underlying Exa request
_SEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="exa-search")
print("[tools] search_web hardened build loaded "
      f"(hard {_SEARCH_TIMEOUT_S:.0f}s bound + {_SEARCH_HTTP_TIMEOUT} socket timeout)")


def _patch_exa_request_timeout(timeout=_SEARCH_HTTP_TIMEOUT):
    """Inject a default socket timeout into the Exa SDK's `requests` calls.

    Exa calls module-level requests.get/post WITHOUT a timeout, so a stalled
    server or dead DNS hangs forever. We wrap the `requests` reference in Exa's
    own module namespace (scoped — never touches the llama.cpp/local_llm sockets)
    so every Exa HTTP call gets a default timeout when one isn't already set.
    """
    from exa_py import Exa
    g = Exa.request.__globals__
    real = g.get("requests")
    if real is None or getattr(real, "_kiki_timeout_patched", False):
        return

    class _TimeoutRequests:
        _kiki_timeout_patched = True

        def __getattr__(self, name):
            return getattr(real, name)

        def get(self, *a, **kw):
            kw.setdefault("timeout", timeout)
            return real.get(*a, **kw)

        def post(self, *a, **kw):
            kw.setdefault("timeout", timeout)
            return real.post(*a, **kw)

    g["requests"] = _TimeoutRequests()


def _get_exa():
    global _exa_client
    if _exa_client is None:
        from exa_py import Exa
        api_key = os.getenv("EXA_API_KEY")
        if not api_key:
            raise RuntimeError("EXA_API_KEY not set in .env — web search unavailable")
        _patch_exa_request_timeout()
        _exa_client = Exa(api_key)
    return _exa_client


def _prewarm_exa():
    """Import exa_py + build the client off the critical path at boot, so the
    FIRST web search doesn't pay the (potentially multi-second under Pi load)
    import/init cost synchronously — that unbounded first-call cost was the
    likely cause of the turn stalling to main.py's 15s tool ceiling. Best
    effort: never raises, never blocks startup (daemon thread)."""
    try:
        _get_exa()
        print("[tools] Exa client prewarmed")
    except Exception as e:
        print(f"[tools] Exa prewarm skipped: {e}")


threading.Thread(target=_prewarm_exa, daemon=True, name="exa-prewarm").start()


async def _get_controller():
    """Get or create a shared KikiController instance."""
    global _controller
    async with _controller_lock:
        if _controller is None or not _controller._connected:
            from kiki_control_client import KikiController
            config = get_full_config()
            ctrl_config = config.get("controller", {})
            host = ctrl_config.get("host", "192.0.2.20")
            _controller = KikiController(host=host)
            connected = await _controller.connect()
            if not connected:
                raise RuntimeError(f"Failed to connect to KikiController at {host}")
            print(f"[Tools] Connected to KikiController at {host}")
    return _controller


# ============================================================================
# Tool Implementations (async)
# ============================================================================

def _exa_search_blocking(query: str):
    """The blocking Exa call, isolated so it can be run on a worker thread."""
    exa = _get_exa()  # lazy init + socket-timeout patch (first call only)
    return exa.search(
        query,
        num_results=3,
        type="auto",
        user_location="IN",
        contents={"highlights": True},
    )


async def search_web(query: str, search_range: str = "today") -> str:
    """Search the web for information using Exa.

    HARD-BOUNDED: the blocking Exa call runs on a worker thread and we wait for
    it with concurrent.futures.Future.result(timeout=...), which raises at the
    deadline WITHOUT joining the worker. This is the critical difference from
    asyncio.wait_for(run_in_executor(...)): wait_for cannot cancel a blocking
    thread, and asyncio.run()'s teardown then joins the orphan — which is what
    silently stalled the turn to main.py's 15s ceiling (no [Search] log ever
    printed because the call never returned). Here search_web ALWAYS returns
    within _SEARCH_TIMEOUT_S no matter what the socket/network does. The Exa
    request also carries a socket timeout (see _patch_exa_request_timeout) so
    the abandoned worker dies instead of leaking.
    """
    _t0 = time.time()
    print(f"[Search] ▶ start {query!r} (range={search_range})", flush=True)
    # search_range is accepted for API compatibility; Exa's relevance ranking
    # already favours recency for these short queries.
    fut = _SEARCH_EXECUTOR.submit(_exa_search_blocking, query)
    loop = asyncio.get_running_loop()
    try:
        # Run the bounded wait off the event loop so we never block it. The
        # waiter thread returns at the deadline; the Exa worker is abandoned.
        result = await loop.run_in_executor(
            None, lambda: fut.result(timeout=_SEARCH_TIMEOUT_S)
        )

        listofresults = []
        if result and result.results:
            for i in result.results:
                if getattr(i, 'highlights', None):
                    listofresults.append(i.highlights)

        print(f"[Search] ✅ Exa returned in {time.time() - _t0:.2f}s", flush=True)
        if not listofresults:
            return "No results found."
        return str(listofresults)

    except FuturesTimeoutError:
        print(f"[Search] ⏱ HARD timeout after {time.time() - _t0:.2f}s "
              f"(limit {_SEARCH_TIMEOUT_S:.0f}s) — network slow/down.", flush=True)
        return "Web search timed out — the network may be down right now."
    except Exception as e:
        print(f"[Search] Error after {time.time() - _t0:.2f}s: {e}", flush=True)
        return f"Error performing web search: {str(e)}"


async def execute_shell_command(command: str) -> str:
    """Execute a shell command and return the output."""
    try:
        print(f"[System Tool] Executing: {command}")
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                command, shell=True, capture_output=True,
                text=True, timeout=10
            )
        )

        output = ""
        if result.stdout:
            output += f"Output:\n{result.stdout.strip()}\n"
        if result.stderr:
            output += f"Error/Stderr:\n{result.stderr.strip()}"
        if not output:
            output = "Command executed with no output."
        return output

    except subprocess.TimeoutExpired:
        return "Error: Command execution timed out (limit: 10s)."
    except Exception as e:
        return f"Error executing command: {str(e)}"


async def get_current_time() -> str:
    """Get the current time."""
    now = datetime.now()
    return now.strftime("%I:%M %p on %A, %B %d, %Y")


async def recall_memory(query: str) -> str:
    """Search all of Kiki's persistent memory with vague-query recall.

    Results are granular and ranked across the knowledge base, conversation
    summaries, and thinking journal.  If no query term matches, the searcher
    returns a varied set of salient memories instead of an empty result."""
    try:
        from core.brain.memory_search import search_memory
        return search_memory(query)
    except Exception as e:
        return f"Error recalling memory: {e}"


async def save_background_research(topic: str, summary: str, details: str = "",
                                   sources=None, tools_used=None,
                                   mode: str = "light_research") -> str:
    """Save verified, dated research chosen by Kiki's unified idle mind."""
    from core.brain.unified_idle_mind import save_background_research as save
    return save(topic, summary, details, sources, tools_used, mode)


async def set_next_turn_note(text: str = "", reason: str = "",
                             expires_hours: float = 24,
                             action: str = "set") -> str:
    """Create, replace or clear Kiki's one current future-conversation note."""
    from core.brain.unified_idle_mind import set_next_turn_note as set_note
    return set_note(text, reason, expires_hours, action)


async def add_open_question(question: str) -> str:
    from core.brain.unified_idle_mind import add_open_question as add
    return add(question)


async def resolve_open_question(question: str) -> str:
    from core.brain.unified_idle_mind import resolve_open_question as resolve
    return resolve(question)


async def switch_voice(voice: str = "") -> str:
    """Switch Kiki's speaking voice to one of the named voices preloaded by
    the tts-server (--voices-dir). Empty/'default'/'normal-like' names reset
    to the server's default reference voice."""
    from core import tts as tts_mod
    available = tts_mod.list_local_voices()
    requested = (voice or "").strip().lower()
    if not requested or requested in ("default", "reset"):
        tts_mod.set_local_voice("")
        return "Voice reset to the default."
    match = None
    if requested in available:
        match = requested
    else:
        # loose match: substring either way handles "arinabh bachchan" etc.
        for name in available:
            if requested in name or name in requested:
                match = name
                break
    if match is None:
        if available:
            return (f"Unknown voice '{voice}'. Available voices: "
                    f"{', '.join(available)}.")
        return (f"Could not reach the voice server to verify '{voice}'. "
                "Voice unchanged.")
    tts_mod.set_local_voice(match)
    return f"Voice switched to '{match}'. Everything said from now on uses that voice."


async def switch_mode(mode: str) -> str:
    """Switch the active system prompt and apply that mode's configured voice."""
    from core.runtime_controls import switch_mode as _switch_mode

    return _switch_mode(mode)


async def set_followups(enabled: bool) -> str:
    """Control whether the microphone stays open after Kiki answers."""
    from core.runtime_controls import set_followups as _set_followups

    return _set_followups(enabled)


async def adjust_volume(action: str, amount: int = None) -> str:
    """Increase, decrease, or set the default audio sink's volume."""
    from core.ir_controls import get_bt_volume, set_bt_volume

    action = str(action or "").strip().lower()
    config = get_full_config().get("assistant_modes", {})
    default_step = int(config.get("volume_step_percent", 10))
    current = get_bt_volume()

    if action == "set":
        if amount is None:
            return "Please specify a volume percentage."
        target = int(amount)
    elif action in {"increase", "up", "raise"}:
        target = current + (int(amount) if amount is not None else default_step)
    elif action in {"decrease", "down", "lower"}:
        target = current - (int(amount) if amount is not None else default_step)
    else:
        return "Volume action must be increase, decrease, or set."

    applied = set_bt_volume(target)
    return f"Volume set to {applied} percent."


async def play_music(song: str) -> str:
    """Resolve and play one exact YouTube video, retaining it for later tools."""
    global _skip_followup
    if is_output_muted():
        return "Music is muted. Show on the LCD that sound must be unmuted first."
    activity_token = activity_generation()
    from core.media_manager import music_manager
    music_config = get_full_config().get("tools", {}).get("music", {})
    music_manager.configure(music_config)
    loop = asyncio.get_running_loop()
    success, result = await loop.run_in_executor(
        None, music_manager.play_query, song, music_config,
        lambda: activity_was_stopped(activity_token),
    )
    if success:
        _skip_followup = True
        # Expose the permanent watch URL to the speaking model and observability
        # result.  Never expose mpv's signed googlevideo stream URL: it expires
        # and is not the exact shareable YouTube video the user asked for.
        current = music_manager.snapshot().get("current") or {}
        webpage_url = current.get("webpage_url", "")
        video = f"{result} - {webpage_url}" if webpage_url else result
        return f"Now playing {video}"
    if result == "stopped by hand gesture":
        return "Music stopped."
    reason = (result or "unknown error").splitlines()[0][:140]
    print(f"[Music] Playback failed: {reason}")
    return f"Music playback failed ({reason}). Tell the user you couldn't play it right now."


async def like_current_song() -> str:
    """Add the exact YouTube video currently playing to liked songs."""
    from core.media_manager import music_manager
    return music_manager.like_current()


async def play_liked_songs() -> str:
    """Play the persistent liked-song list as an automatically advancing playlist."""
    global _skip_followup
    if is_output_muted():
        return "Music is muted. Show on the LCD that sound must be unmuted first."
    activity_token = activity_generation()
    from core.media_manager import music_manager
    music_config = get_full_config().get("tools", {}).get("music", {})
    music_manager.configure(music_config)
    loop = asyncio.get_running_loop()
    success, result = await loop.run_in_executor(
        None, music_manager.play_liked, music_config,
        lambda: activity_was_stopped(activity_token),
    )
    if success:
        _skip_followup = True
        return f"Playing your liked songs, starting with {result}."
    return result


async def play_last_song() -> str:
    """Replay the exact most recently played YouTube video."""
    global _skip_followup
    if is_output_muted():
        return "Music is muted. Show on the LCD that sound must be unmuted first."
    activity_token = activity_generation()
    from core.media_manager import music_manager
    music_config = get_full_config().get("tools", {}).get("music", {})
    music_manager.configure(music_config)
    loop = asyncio.get_running_loop()
    success, result = await loop.run_in_executor(
        None, music_manager.play_last, music_config,
        lambda: activity_was_stopped(activity_token),
    )
    if success:
        _skip_followup = True
        return f"Now playing {result} again."
    return result


async def control_music(action: str) -> str:
    """Pause/resume playback or move inside the active liked-song playlist."""
    global _skip_followup
    from core.media_manager import music_manager
    music_config = get_full_config().get("tools", {}).get("music", {})
    music_manager.configure(music_config)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, music_manager.control, action, music_config
    )
    _skip_followup = True
    return result


async def set_timer(duration: Any) -> str:
    """Set a validated in-process countdown and play its alarm through mpv."""
    from core.media_manager import timer_manager
    timer_config = get_full_config().get("tools", {}).get("timer", {})
    try:
        seconds_total = timer_manager.set_timer(duration, timer_config)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        return f"Could not set the timer: {exc}"

    if seconds_total >= 60:
        minutes = seconds_total // 60
        seconds = seconds_total % 60
        if seconds:
            return f"Timer set for {minutes} minutes and {seconds} seconds"
        return f"Timer set for {minutes} minutes"
    return f"Timer set for {seconds_total} seconds"


async def update_knowledge(category: str, action: str, key: str,
                           value: str = "", attribute: str = "") -> str:
    """Store or retrieve information from long-term memory."""
    from core.brain.knowledge_base import get_knowledge_base, save_knowledge_base

    kb = get_knowledge_base()

    try:
        if category == "self":
            key = "Kiki"
            category = "people"

        if category == "people":
            if action in ("add", "update"):
                if attribute:
                    if attribute in ("routine", "interests", "hobbies"):
                        values = [v.strip() for v in value.split(",")]
                        for v in values:
                            kb.add_person_attribute(key, attribute, v, append=True)
                        result = f"Added to {key}'s {attribute}: {values}"
                    elif attribute == "notes":
                        kb.add_note_to_person(key, value)
                        result = f"Added note about {key}: {value}"
                    elif attribute == "character":
                        kb.set_person_character(key, value)
                        result = f"Set {key}'s character to: {value}"
                    elif attribute == "current_ongoing":
                        kb.set_current_ongoing(key, value)
                        result = f"Set {key}'s current situation: {value}"
                    elif attribute == "appearance":
                        kb.add_person_attribute(key, "appearance", value)
                        result = f"Set {key}'s appearance: {value}"
                    else:
                        kb.add_person_attribute(key, attribute, value)
                        result = f"Set {key}.{attribute} = {value}"
                else:
                    if "," in value and not value.startswith("relationship:"):
                        traits = [t.strip() for t in value.split(",")]
                        kb.add_person(key, traits=traits)
                        result = f"Added/updated {key} with traits: {traits}"
                    elif value.startswith("relationship:"):
                        kb.add_person(key, relationship=value.replace("relationship:", "").strip())
                        result = f"Set {key}'s relationship: {value}"
                    else:
                        kb.add_person(key, notes=value)
                        result = f"Added note about {key}: {value}"
            elif action == "get":
                if attribute:
                    attr_value = kb.get_person_attribute(key, attribute)
                    result = f"{key}'s {attribute}: {attr_value}" if attr_value else f"No {attribute} for {key}"
                else:
                    person = kb.get_person(key)
                    result = f"Info about {key}: {person}" if person else f"No info about {key}"
            elif action == "remove":
                result = f"Removed {key}" if kb.remove_person(key) else f"{key} not found"
            else:
                result = f"Unknown action: {action}"

        elif category == "environments":
            if action in ("add", "update"):
                if attribute == "description":
                    kb.add_environment(key, description=value)
                elif attribute:
                    kb.update_environment(key, attribute, value)
                else:
                    kb.add_environment(key, description=value)
                result = f"Environment '{key}': {value}"
            elif action == "get":
                env = kb.get_environment(key)
                result = f"Environment '{key}': {env}" if env else f"Unknown environment: {key}"
            elif action == "remove":
                result = f"Removed environment: {key}" if kb.remove_environment(key) else f"Not found: {key}"
            else:
                result = f"Unknown action: {action}"

        elif category == "learnings":
            if action in ("add", "update"):
                kb.add_learning(key, value)
                result = f"Learned [{key}]: {value}"
            elif action == "get":
                learnings = kb.get_learnings(key)
                result = f"Learnings about {key}: {learnings}" if learnings else f"No learnings about {key}"
            else:
                result = f"Unknown action: {action}"

        elif category == "experiences":
            if action == "add":
                parts = value.split("|", 1)
                outcome = parts[0].strip() if parts else "neutral"
                details = parts[1].strip() if len(parts) > 1 else None
                kb.add_experience(key, outcome=outcome, details=details)
                result = f"Logged experience: {key} ({outcome})"
            elif action == "get":
                experiences = kb.get_recent_experiences(10)
                result = f"Recent experiences: {experiences}"
            else:
                result = f"Unknown action: {action}"

        elif category == "facts":
            if action in ("add", "update"):
                kb.add_fact(key, value)
                result = f"Stored fact: {key} = {value}"
            elif action == "get":
                fact = kb.get_fact(key)
                result = f"{key}: {fact}" if fact is not None else f"Unknown fact: {key}"
            elif action == "remove":
                result = f"Removed fact: {key}" if kb.remove_fact(key) else f"Not found: {key}"
            else:
                result = f"Unknown action: {action}"

        elif category == "personality":
            if action == "add":
                kb.add_trait(key)
                result = f"Added personality trait: {key}"
            elif action == "update":
                kb.set_preference(key, value)
                result = f"Set preference: {key} = {value}"
            elif action == "get":
                personality = kb.get_personality()
                result = f"My personality: {personality}"
            else:
                result = f"Unknown action: {action}"

        elif action == "search":
            results = kb.search(key)
            if results:
                result = f"Search results for '{key}':\n" + "\n".join(
                    f"[{cat}] {', '.join(items)}" for cat, items in results.items()
                )
            else:
                result = f"No results found for '{key}'"
        else:
            result = f"Unknown category: {category}"

        if action in ("add", "update", "remove"):
            save_knowledge_base()

        print(f"[KnowledgeBase Tool] {result}")
        return result

    except Exception as e:
        return f"Error updating knowledge base: {e}"


async def remember_me(person_name: str) -> str:
    """Remember a person's face via face training."""
    try:
        print(f"[Robot Control] Starting face training for: {person_name}")
        controller = await _get_controller()
        response = await controller.train_person(person_name)

        if response.get("status") == "ok":
            print(f"[Robot Control] Training initiated for {person_name}")

            async def wait_for_training_complete():
                try:
                    async for event in controller.listen_events():
                        if event.get("event") == "training_complete":
                            if event.get("person") == person_name:
                                print(f"[Robot Control] Training complete for {person_name}")
                                break
                except Exception as e:
                    print(f"[Robot Control] Error waiting for training: {e}")

            asyncio.create_task(wait_for_training_complete())
            return f"I'm now remembering your face, {person_name}. Please stay still and look at me for about 10 seconds."
        else:
            error = response.get("error", "Unknown error")
            return f"Sorry, I couldn't start face training. Error: {error}"
    except Exception as e:
        return f"Sorry, error starting face training: {str(e)}"


async def track_person(person_name: str) -> str:
    """Set a specific person to track with the robot's neck."""
    try:
        controller = await _get_controller()
        success = await controller.set_target_person(person_name)
        if success:
            return f"I'm now tracking {person_name}. I'll follow them with my gaze."
        return f"Sorry, I couldn't set {person_name} as tracking target."
    except Exception as e:
        return f"Sorry, error setting tracking target: {str(e)}"


async def set_person_real_name(temp_name: str, real_name: str) -> str:
    """Rename a person Kiki met as a stranger (a 'guest_...' temp identity) to
    their real name, in BOTH the face-recognition DB and Kiki's memory, so Kiki
    recognises them by their real name from now on. Only call this AFTER the
    person has confirmed the name is correct."""
    from core.brain.knowledge_base import get_knowledge_base, save_knowledge_base
    try:
        # 1) Rename in the face-recognition DB (label update, no retraining).
        db_msg = ""
        try:
            controller = await _get_controller()
            resp = await controller.rename_person(temp_name, real_name)
            db_msg = resp.get("message", "") if isinstance(resp, dict) else ""
        except Exception as e:
            db_msg = f"face DB rename failed: {e}"

        # 2) Rename in the knowledge base (merge into an existing record if the
        #    real name is already known, else move the temp record over).
        kb = get_knowledge_base()
        people = kb.data.setdefault("people", {})
        person = people.get(temp_name)
        if person is not None:
            person["is_temp_name"] = False
            person["pending_real_name"] = False
            person["real_name_learned"] = datetime.now().isoformat()
            if real_name in people and real_name != temp_name:
                existing = people[real_name]
                for note in person.get("notes_list", []):
                    kb.add_note_to_person(real_name, note)
                for k in ("appearance", "clothing", "first_met", "face_thumb"):
                    if person.get(k) and not existing.get(k):
                        existing[k] = person[k]
                del people[temp_name]
            elif real_name != temp_name:
                people[real_name] = person
                del people[temp_name]
            kb.add_note_to_person(
                real_name,
                f"Learned their real name is '{real_name}' (met as '{temp_name}') "
                f"on {datetime.now():%Y-%m-%d %H:%M}.")
            save_knowledge_base()

        # 3) Rename the saved face thumbnail + point the record at it.
        try:
            from robot.stranger_enroll import face_thumb_path
            old_p, new_p = face_thumb_path(temp_name), face_thumb_path(real_name)
            if os.path.exists(old_p):
                os.replace(old_p, new_p)
            p2 = kb.get_person(real_name)
            if p2 is not None and os.path.exists(new_p):
                p2["face_thumb"] = new_p
                save_knowledge_base()
            # 4) Celebrate on the OLED.
            from core.oled_display import oled_manager
            oled_manager.show_face(real_name, image=new_p if os.path.exists(new_p) else None,
                                   subtitle="nice to meet you")
        except Exception:
            pass

        return (f"Got it — I've renamed {temp_name} to {real_name} everywhere, "
                f"so I'll recognise them by name from now on."
                + (f" ({db_msg})" if db_msg else ""))
    except Exception as e:
        return f"Sorry, I couldn't rename that person: {e}"



# ============================================================================
# Worker Tools
# ============================================================================

async def schedule_worker(name: str, task_description: str, trigger_type: str,
                          trigger_value: str = "", conditions: str = "") -> str:
    """
    Schedule an autonomous worker task.
    
    trigger_type: "scheduled_time", "event", or "recurring"
    trigger_value: ISO datetime for scheduled_time, event name for event, seconds for recurring
    conditions: JSON string of conditions list, e.g. '[{"condition_type": "person_seen", "params": {"person": "Alex", "within_minutes": 60}}]'
    """
    try:
        from core.workers.worker_manager import get_worker_manager
        manager = get_worker_manager()

        # Parse conditions if provided
        parsed_conditions = None
        if conditions and conditions.strip():
            try:
                parsed_conditions = json.loads(conditions)
            except json.JSONDecodeError:
                return f"Error: Invalid conditions JSON: {conditions}"

        worker = manager.create_worker(
            name=name,
            task_description=task_description,
            trigger_type=trigger_type,
            trigger_value=trigger_value,
            conditions=parsed_conditions,
        )
        return f"Worker scheduled: {worker}"
    except ValueError as e:
        return f"Error scheduling worker: {e}"
    except Exception as e:
        return f"Error scheduling worker: {e}"


async def cancel_worker(worker_id: str) -> str:
    """Cancel a pending worker by its ID or name."""
    try:
        from core.workers.worker_manager import get_worker_manager
        manager = get_worker_manager()
        success = manager.cancel_worker(worker_id)
        if success:
            return f"Worker '{worker_id}' cancelled successfully."
        return f"No active worker found with ID or name '{worker_id}'."
    except Exception as e:
        return f"Error cancelling worker: {e}"


async def list_workers() -> str:
    """List all active and pending workers."""
    try:
        from core.workers.worker_manager import get_worker_manager
        manager = get_worker_manager()
        summary = manager.get_status_summary()
        return summary
    except Exception as e:
        return f"Error listing workers: {e}"


async def execute_python_code(code: str) -> str:
    """Execute Python code in a subprocess and return the output."""
    try:
        print(f"[Tool] Executing Python code ({len(code)} chars)...")
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: subprocess.run(
                ["/usr/bin/python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
                cwd="/srv/kikifast"
            )
        )
        output = ""
        if result.stdout:
            output += result.stdout.strip()
        if result.stderr:
            output += f"\nSTDERR: {result.stderr.strip()}"
        return output if output else "Code executed with no output."
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 30s."
    except Exception as e:
        return f"Error executing code: {e}"


# ============================================================================
# Self-Extend Tool Handlers
# ============================================================================

async def self_extend_list_skills() -> str:
    """List all installed Kiki skills with previews."""
    try:
        from core.self_extend.skill_manager import SkillManager
        sm = SkillManager()
        skills = sm.list_skills()
        if not skills:
            return f"No skills installed yet. Skills dir: {sm.root}"
        lines = [f"📁 Skills directory: {sm.root}\n"]
        for s in skills:
            lines.append(f"🔧 **{s['name']}**\n   {s['preview'][:200]}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing skills: {e}"


async def self_extend_create_skill(name: str, skill_md_content: str,
                                   extra_files: str = "") -> str:
    """
    Create a new Kiki skill (SKILL.md + optional extra files).
    extra_files: JSON string like '{"helper.py": "code here"}'.
    """
    try:
        from core.self_extend.skill_manager import SkillManager
        sm = SkillManager()
        parsed_extras = None
        if extra_files and extra_files.strip():
            try:
                parsed_extras = json.loads(extra_files)
            except json.JSONDecodeError:
                pass
        path = sm.create_skill(name, skill_md_content, parsed_extras)
        return f"✅ Skill '{name}' created at: {path}"
    except Exception as e:
        return f"Error creating skill: {e}"


async def self_extend_search_mcp(query: str = "", page: int = 1) -> str:
    """
    Search the Smithery MCP registry via CLI (smithery mcp search).
    Returns a ranked list of server IDs and descriptions.
    """
    try:
        from core.self_extend import smithery_cli
        return smithery_cli.mcp_search(query, page=page)
    except Exception as e:
        return f"Error searching MCP registry: {e}"


async def self_extend_mcp_add(server: str) -> str:
    """
    Add an MCP server connection via Smithery CLI (smithery mcp add).
    server: registry server ID (e.g. 'exa') or full connection URL.
    """
    try:
        from core.self_extend import smithery_cli
        return smithery_cli.mcp_add(server)
    except Exception as e:
        return f"Error adding MCP server: {e}"


async def self_extend_mcp_list_connections() -> str:
    """List all connected MCP servers (smithery mcp list)."""
    try:
        from core.self_extend import smithery_cli
        return smithery_cli.mcp_list()
    except Exception as e:
        return f"Error listing MCP connections: {e}"


async def self_extend_mcp_remove(connection_ids: str) -> str:
    """
    Remove one or more MCP connections (smithery mcp remove).
    connection_ids: space-separated IDs from mcp_list_connections.
    """
    try:
        from core.self_extend import smithery_cli
        ids = connection_ids.split()
        return smithery_cli.mcp_remove(*ids)
    except Exception as e:
        return f"Error removing MCP connection: {e}"


async def self_extend_tool_find(connection: str, query: str = "") -> str:
    """Search tools available in a connected MCP server (smithery tool find)."""
    try:
        from core.self_extend import smithery_cli
        return smithery_cli.tool_find(connection, query)
    except Exception as e:
        return f"Error finding tools: {e}"


async def self_extend_tool_list(connection: str) -> str:
    """List all tools from a connected MCP server (smithery tool list)."""
    try:
        from core.self_extend import smithery_cli
        return smithery_cli.tool_list(connection)
    except Exception as e:
        return f"Error listing tools: {e}"


async def self_extend_tool_call(connection: str, tool: str,
                                args_json: str = "") -> str:
    """
    Call a tool from a connected MCP server (smithery tool call).
    args_json: JSON string of arguments, e.g. '{"query": "hello"}'.
    """
    try:
        from core.self_extend import smithery_cli
        return smithery_cli.tool_call(connection, tool, args_json)
    except Exception as e:
        return f"Error calling tool: {e}"


async def read_gmail(query: str = "", max_results: int = 5) -> str:
    """Read a small Gmail result set without raw HTML, MIME data or headers."""
    from core.self_extend.mcp_data_access import read_gmail as read
    return await asyncio.get_running_loop().run_in_executor(
        None, lambda: read(query=query, max_results=max_results))


async def read_gmail_message(message_id: str,
                             max_body_chars: int = 5000) -> str:
    """Read one Gmail message as clean, bounded text."""
    from core.self_extend.mcp_data_access import read_gmail_message as read
    return await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: read(message_id=message_id, max_body_chars=max_body_chars))


async def read_gmail_thread(thread_id: str, max_messages: int = 10,
                            max_body_chars: int = 2000) -> str:
    """Read one Gmail thread as a bounded list of clean messages."""
    from core.self_extend.mcp_data_access import read_gmail_thread as read
    return await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: read(
            thread_id=thread_id, max_messages=max_messages,
            max_body_chars=max_body_chars))


async def search_notion(query: str, max_results: int = 5,
                        highlight_chars: int = 240) -> str:
    """Search Notion with an explicitly small result/highlight budget."""
    from core.self_extend.mcp_data_access import search_notion as search
    return await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: search(
            query=query, max_results=max_results,
            highlight_chars=highlight_chars))


async def read_notion(entity_id: str, max_chars: int = 6000) -> str:
    """Read one Notion entity with bounded enhanced-Markdown content."""
    from core.self_extend.mcp_data_access import read_notion as read
    return await asyncio.get_running_loop().run_in_executor(
        None, lambda: read(entity_id=entity_id, max_chars=max_chars))


async def _whatsapp_call(name: str, arguments: dict,
                         resolve_recipient: bool = False,
                         resolve_contact: bool = False,
                         timeout: Optional[float] = None) -> str:
    """Run one local MCP call outside the voice/event-loop thread."""
    from core.self_extend.whatsapp_mcp import call_whatsapp_tool_json
    return await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: call_whatsapp_tool_json(
            name, arguments,
            resolve_recipient=resolve_recipient,
            resolve_contact=resolve_contact,
            timeout=timeout),
    )


async def search_contacts(query: str) -> str:
    return await _whatsapp_call("search_contacts", {"query": query})


def _empty_whatsapp_result(payload: str) -> bool:
    return str(payload or "").strip() in ("[]", "null", "{}", "")


# Descriptions are keyed by message id and never change, so one read per image
# per process is enough even if the agent re-lists the same chat.
_IMAGE_DESC_CACHE: Dict[str, str] = {}
_IMAGE_DESC_CACHE_MAX = 300

# Image description gets its OWN threads, never the shared default executor.
# When the inline pass times out, its workers keep running until their Groq
# call returns; on the default executor those stragglers occupy the same small
# thread pool the agent uses for its next tool call, so a slow vision batch
# silently queued the whole agent behind it (measured: a 5s image cap still
# produced a 108s turn). Isolating them means a timeout really is a timeout.
_IMAGE_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wa-image")


def _whatsapp_image_config() -> tuple[bool, int, float]:
    cfg = get_full_config().get("whatsapp", {}) or {}
    return (bool(cfg.get("describe_images", True)),
            max(0, int(cfg.get("describe_images_limit", 4))),
            max(1.0, float(cfg.get("describe_images_timeout", 7.0))))


_MEDIA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "whatsapp-mcp", "whatsapp-bridge", "store")
_MESSAGES_DB = os.path.join(_MEDIA_ROOT, "messages.db")


def _local_media_path(message_id: str, chat_jid: str) -> Optional[str]:
    """Path to this message's media IF the bridge already downloaded it.

    The bridge files media at ``store/<chat_jid>/<filename>`` and records the
    filename in messages.db, so presence is a cheap stat — no MCP round trip.
    This is what keeps the inline pass bounded: cold media is fetched over the
    network and those fetches serialize behind the single MCP session lock,
    which is how one image-heavy chat blocked a turn for 106 seconds.
    """
    try:
        import sqlite3
        conn = sqlite3.connect(f"file:{_MESSAGES_DB}?mode=ro", uri=True, timeout=2)
        try:
            row = conn.execute(
                "SELECT filename FROM messages WHERE id=? AND chat_jid=?",
                (message_id, chat_jid)).fetchone()
        finally:
            conn.close()
    except Exception:
        return None
    if not row or not row[0]:
        return None
    path = os.path.join(_MEDIA_ROOT, chat_jid, row[0])
    return path if os.path.isfile(path) else None


# NOTE: there is deliberately NO background media prefetch here. Downloading
# cold media on a daemon thread looked like a free win, but every download
# holds the single MCP session lock, so the agent's OWN next WhatsApp call
# queues behind it — measured: a chat summary went from 3.8s to 25.4s, and an
# image-heavy one to over 100s, purely from prefetch contention. Images that
# are not on disk stay unviewed until something explicitly asks for one via
# read_whatsapp_image, which is a single bounded call.


async def _describe_images_inline(rows: list) -> list:
    """Replace empty image rows with what the picture actually says.

    An image message arrives as ``content: ""`` with ``media_type: "image"`` —
    indistinguishable from a blank message, so the agent skipped pictures
    entirely when summarising (measured: 18 images in a chat, 0 reads). Rather
    than hoping the model asks, the newest few are described up front through
    the same free Groq qwen VLM `look_at_scene` uses.

    Bounded and concurrent: the cap keeps the shared 8K/min Groq pool safe, and
    gathering means N images cost about as long as one.
    """
    enabled, limit, budget = _whatsapp_image_config()
    if not enabled:
        return rows

    targets = [
        row for row in rows
        if isinstance(row, dict)
        and str(row.get("media_type") or "").lower() == "image"
        and not str(row.get("content") or "").strip()
        and row.get("id") and row.get("chat_jid")
    ]
    if not targets:
        return rows

    # SLICE FIRST. An active group can carry dozens of images, and doing any
    # per-image work (a disk check, a download, a vision call) across all of
    # them is unbounded — that is what turned one chat into a 106-second turn.
    # Newest first, because a summary cares about what just arrived.
    targets.sort(key=lambda r: str(r.get("timestamp") or ""), reverse=True)
    chosen, skipped = targets[:limit], targets[limit:]

    for row in skipped:
        row["content"] = (
            f"[image — not viewed; call read_whatsapp_image("
            f"message_id='{row['id']}', chat_jid='{row['chat_jid']}') to see it]")

    from core.vision.instant_vision import describe_image_file
    loop = asyncio.get_running_loop()

    def read_one(row):
        """Disk check + vision, in one worker thread."""
        cached = _IMAGE_DESC_CACHE.get(row["id"])
        if cached:
            return cached
        path = _local_media_path(row["id"], row["chat_jid"])
        if not path:
            # Not downloaded yet. Deliberately NOT fetched inline: cold media
            # goes over the network and serializes behind the single MCP
            # session lock. Warmed in the background instead.
            return None
        return describe_image_file(
            path, question=("Describe this image in detail. Read out any text, "
                            "dates, times, names, amounts or instructions exactly."))

    async def describe(row):
        try:
            return row, await loop.run_in_executor(_IMAGE_POOL, read_one, row)
        except Exception as exc:
            return row, f"[image — could not be read: {exc}]"

    # ONE bound over the WHOLE pass. On timeout the chat is summarised without
    # the pictures rather than the turn hanging — a slightly thinner summary
    # beats a frozen Kiki.
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*(describe(r) for r in chosen),
                           return_exceptions=True),
            timeout=budget)
    except (asyncio.TimeoutError, TimeoutError):
        print(f"[WhatsApp] image pass exceeded {budget:.0f}s — "
              f"summarising without images")
        for row in chosen:
            if not str(row.get("content") or "").strip():
                row["content"] = "[image — not viewed this time]"
        return rows

    described = 0
    pending = []
    for item in results:
        if isinstance(item, Exception) or not isinstance(item, tuple):
            continue
        row, text = item
        if text is None:
            row["content"] = "[image — still downloading, not viewed yet]"
            pending.append(row)
            continue
        text = str(text or "").strip()
        if not text or text.lower().startswith("error"):
            row["content"] = "[image — could not be read]"
            continue
        if len(_IMAGE_DESC_CACHE) < _IMAGE_DESC_CACHE_MAX:
            _IMAGE_DESC_CACHE[row["id"]] = text
        row["content"] = f"[image] {text}"
        described += 1
    if described:
        print(f"[WhatsApp] auto-described {described} image(s) via Groq qwen")
    return rows


async def list_messages(after: str = "", before: str = "",
                        sender_phone_number: str = "", chat_jid: str = "",
                        query: str = "", limit: int = 20, page: int = 0,
                        include_context: bool = True, context_before: int = 1,
                        context_after: int = 1) -> str:
    def _args(jid):
        return {
            "after": after or None,
            "before": before or None,
            "sender_phone_number": sender_phone_number or None,
            "chat_jid": jid or None,
            "query": query or None,
            "limit": min(50, max(1, int(limit))),
            "page": max(0, int(page)),
            "include_context": bool(include_context),
            "context_before": min(10, max(0, int(context_before))),
            "context_after": min(10, max(0, int(context_after))),
        }

    result = await _whatsapp_call("list_messages", _args(chat_jid),
                                  resolve_contact=bool(sender_phone_number))
    # A direct chat is ADDRESSED by phone but FILED under its @lid, so looking
    # up the resolved phone JID returns nothing and the agent then states "there
    # are no messages" as fact. Retry the other form before believing that.
    if chat_jid and _empty_whatsapp_result(result):
        from core.self_extend.whatsapp_contacts import jid_variants
        for alt in jid_variants(chat_jid)[1:]:
            retry = await _whatsapp_call("list_messages", _args(alt))
            if not _empty_whatsapp_result(retry):
                print(f"[WhatsApp] list_messages: {chat_jid} empty → used {alt}")
                result = retry
                break
    return await _with_image_descriptions(result)


def _compact_message_rows(rows: list) -> list:
    """Strip a message list down to what a reader actually needs.

    The MCP's rows repeat `chat_jid`, `sender` and a 32-char `id` on EVERY
    message, so one row costs ~300 characters of which ~50 carry meaning. The
    agent truncates each tool result, so that overhead was the binding
    constraint on summary quality: asking for 100 messages and getting 1500
    characters back meant summarising a chat from its last FIVE lines.

    Compacting to ~60 chars/row fits roughly 10x more conversation in the same
    budget. `id` and `chat_jid` survive only on media rows, where
    read_whatsapp_image needs them.
    """
    out = []
    for row in rows:
        if not isinstance(row, dict):
            out.append(row)
            continue
        stamp = str(row.get("timestamp") or "")
        slim = {
            "time": stamp[5:16].replace("T", " ") or stamp,
            "from": row.get("sender_name") or row.get("sender") or "?",
            "text": row.get("content") or "",
        }
        if row.get("media_type"):
            slim["media"] = row["media_type"]
            slim["id"] = row.get("id")
            slim["chat_jid"] = row.get("chat_jid")
        out.append(slim)
    return out


async def _with_image_descriptions(payload: str) -> str:
    """Run the inline image pass over a list_messages JSON payload."""
    try:
        rows = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return payload
    if not isinstance(rows, list):
        return payload
    rows = _compact_message_rows(await _describe_images_inline(rows))
    return json.dumps(rows, ensure_ascii=False, default=str)


async def list_chats(query: str = "", limit: int = 20, page: int = 0,
                     include_last_message: bool = True,
                     sort_by: str = "last_active") -> str:
    result = await _whatsapp_call("list_chats", {
        "query": query or None,
        "limit": min(50, max(1, int(limit))),
        "page": max(0, int(page)),
        "include_last_message": bool(include_last_message),
        "sort_by": sort_by,
    })
    # The MCP's own search is a substring LIKE, so a misheard name ("project cereal")
    # finds nothing for the group really called "project circle club" and the agent
    # wastes a whole turn (~1.5s) discovering that. Retry once, fuzzily.
    if query and result.strip() in ("[]", "null", ""):
        from core.self_extend.whatsapp_mcp import fuzzy_chat_search
        matches = await asyncio.get_running_loop().run_in_executor(
            None, lambda: fuzzy_chat_search(query, limit=min(10, int(limit))))
        if matches:
            print(f"[WhatsApp] list_chats({query!r}) empty → fuzzy found "
                  f"{[m['name'] for m in matches[:3]]}")
            return json.dumps(matches, ensure_ascii=False, default=str)
    return result


async def get_chat(chat_jid: str, include_last_message: bool = True) -> str:
    result = await _whatsapp_call("get_chat", {
        "chat_jid": chat_jid,
        "include_last_message": bool(include_last_message),
    })
    # Same phone-vs-@lid split as list_messages: the chat row may only exist
    # under the alternate identifier.
    if chat_jid and _empty_whatsapp_result(result):
        from core.self_extend.whatsapp_contacts import jid_variants
        for alt in jid_variants(chat_jid)[1:]:
            retry = await _whatsapp_call("get_chat", {
                "chat_jid": alt,
                "include_last_message": bool(include_last_message),
            })
            if not _empty_whatsapp_result(retry):
                print(f"[WhatsApp] get_chat: {chat_jid} empty → used {alt}")
                return retry
    return result


async def get_direct_chat_by_contact(sender_phone_number: str) -> str:
    return await _whatsapp_call("get_direct_chat_by_contact", {
        "sender_phone_number": sender_phone_number,
    }, resolve_contact=True)


async def get_contact_chats(jid: str, limit: int = 20, page: int = 0) -> str:
    return await _whatsapp_call("get_contact_chats", {
        "jid": jid,
        "limit": min(50, max(1, int(limit))),
        "page": max(0, int(page)),
    }, resolve_contact=True)


async def get_last_interaction(jid: str) -> str:
    return await _whatsapp_call(
        "get_last_interaction", {"jid": jid}, resolve_contact=True)


async def get_message_context(message_id: str, before: int = 5,
                              after: int = 5) -> str:
    return await _whatsapp_call("get_message_context", {
        "message_id": message_id,
        "before": min(20, max(0, int(before))),
        "after": min(20, max(0, int(after))),
    })


async def send_message(recipient: str, message: str) -> str:
    return await _whatsapp_call(
        "send_message", {"recipient": recipient, "message": message},
        resolve_recipient=True)


async def send_file(recipient: str, media_path: str) -> str:
    return await _whatsapp_call(
        "send_file", {"recipient": recipient, "media_path": media_path},
        resolve_recipient=True)


async def send_audio_message(recipient: str, media_path: str) -> str:
    return await _whatsapp_call(
        "send_audio_message",
        {"recipient": recipient, "media_path": media_path},
        resolve_recipient=True)


async def download_media(message_id: str, chat_jid: str,
                         timeout: Optional[float] = None) -> str:
    return await _whatsapp_call("download_media", {
        "message_id": message_id,
        "chat_jid": chat_jid,
    }, timeout=timeout)


def _media_path_from_download(raw: str) -> str:
    """Pull a local file path out of whatever download_media returned.

    The MCP replies with either a bare path or a small JSON object, and the
    shape has changed between bridge versions — so probe rather than assume.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    if text.startswith("{") or text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            for key in ("path", "file_path", "media_path", "filename", "message"):
                value = data.get(key)
                if isinstance(value, str) and os.path.isfile(value):
                    return value
            # Some builds bury the path in a human-readable status string.
            for value in data.values():
                if isinstance(value, str):
                    for token in value.split():
                        if os.path.isfile(token):
                            return token
            return ""
    return text if os.path.isfile(text) else ""


async def read_whatsapp_image(message_id: str, chat_jid: str,
                              question: str = "",
                              download_timeout: Optional[float] = None) -> str:
    """Download a WhatsApp image/media message and describe what it shows.

    A WhatsApp message whose content starts with "[image - Message ID: X -
    Chat JID: Y]" has a picture attached; this downloads it and reads it with
    the same free Groq vision model `look_at_scene` uses.

    ``download_timeout`` bounds the fetch. Already-downloaded media returns in
    ~0.02s, but never-seen media goes over the network AND serializes behind the
    single MCP session lock, so the inline summariser passes a short value —
    without it, a chat full of fresh images ran for 110 seconds.
    """
    from core.vision.instant_vision import describe_image_file

    loop = asyncio.get_running_loop()
    raw = await download_media(message_id, chat_jid, timeout=download_timeout)
    path = await loop.run_in_executor(None, _media_path_from_download, raw)
    if not path:
        return (f"Error: could not download media for message {message_id}. "
                f"Bridge said: {str(raw)[:300]}")
    try:
        return await loop.run_in_executor(
            None, lambda: describe_image_file(path, question=question or None))
    except Exception as e:
        return f"Error reading the image: {e}"


async def record_voice_note(seconds: int = 10) -> str:
    """Record the microphone for a few seconds and return the wav path.

    Pair with send_audio_message(recipient, path) to send it as a WhatsApp
    voice note (the bridge converts the wav to Opus/OGG with ffmpeg).
    """
    from core.stt import get_active_engine

    engine = get_active_engine()
    if engine is None:
        return "Error: the microphone is not running, so nothing can be recorded."
    try:
        seconds = max(1, min(int(seconds), 60))
    except (TypeError, ValueError):
        seconds = 10
    try:
        path = await asyncio.get_running_loop().run_in_executor(
            None, lambda: engine.record_clip(seconds))
    except Exception as e:
        return f"Error recording audio: {e}"
    return json.dumps({"ok": True, "media_path": path, "seconds": seconds})


async def complex_query(request: str, context: str = "") -> str:
    """Hand a multi-step request to the fast cloud action agent.

    Imported lazily so nothing about this feature is paid for on Kiki's boot
    or speaking import path.
    """
    from core.brain.action_agent import run_complex_query

    return await run_complex_query(request, context)


async def self_extend_create_mcp_server(server_name: str, description: str,
                                        tools_json: str) -> str:
    """
    Generate, save, and register a FastMCP Python MCP server.
    tools_json: JSON array of tool specs:
      [{"name":"...","description":"...","params":[{"name":"p","type":"str"}],"impl":"return 'ok'"}]
    """
    try:
        from core.self_extend.mcp_manager import MCPServerCreator
        creator = MCPServerCreator()
        tools = json.loads(tools_json) if tools_json else []
        result = creator.create_and_register(
            server_name=server_name,
            description=description,
            tools=tools,
        )
        return (
            f"✅ MCP server '{server_name}' created!\n"
            f"   Code: {result['code_path']}\n"
            f"   Config: {result['config_status']}"
        )
    except Exception as e:
        return f"Error creating MCP server: {e}"


async def self_extend_run_task(goal: str) -> str:
    """
    Run the full KikiSelfExtendAgent to accomplish a complex self-extension goal.
    Examples:
      - 'Find a web automation MCP server and add it to my config'
      - 'Create a skill for analyzing robot sensor logs'
      - 'Create a custom MCP server for home automation control'
    """
    try:
        from core.self_extend.kiki_self_extend_agent import KikiSelfExtendAgent
        agent = KikiSelfExtendAgent()
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool,
                lambda: agent.run_task(goal)
            )
        return result
    except Exception as e:
        return f"Error running self-extend task: {e}"


async def self_extend_search_smithery_skills(query: str, page: int = 1) -> str:
    """
    Search the Smithery Skills registry via CLI (smithery skill search).
    Returns skill IDs (namespace/slug) and descriptions.
    Use self_extend_install_smithery_skill to install a result.
    """
    try:
        from core.self_extend import smithery_cli
        return smithery_cli.skill_search(query, page=page)
    except Exception as e:
        return f"Error searching Smithery skills: {e}"


async def self_extend_view_smithery_skill(identifier: str) -> str:
    """
    View the FULL content of a Smithery skill WITHOUT installing it.
    identifier: 'namespace/slug', e.g. 'langfuse/skill-developer'
    Uses smithery skill view which returns the complete SKILL.md content.
    """
    try:
        from core.self_extend import smithery_cli
        return smithery_cli.skill_view(identifier)
    except Exception as e:
        return f"Error viewing skill: {e}"


async def self_extend_install_smithery_skill(identifier: str) -> str:
    """
    Download and install a Smithery skill into Kiki's local skills directory.
    Uses 'smithery skill view' to fetch COMPLETE skill content (full documentation,
    not just a brief description). Replaces any previous incomplete install.
    identifier: 'namespace/slug', e.g. 'langfuse/skill-developer'
    """
    try:
        from core.self_extend import smithery_cli
        result = smithery_cli.install_skill_to_kiki(identifier)

        if "error" in result:
            return f"Error: {result['error']}"

        already = " (updated with latest)" if result.get("already_existed") else ""
        chars = result.get("content_length", 0)
        return (
            f"\u2705 Installed skill '{identifier}'{already}\n"
            f"   Local name: {result['skill_name']}\n"
            f"   Saved to: {result['path']}\n"
            f"   Content size: {chars} characters (full skill documentation)\n"
            f"   The skill is now available in Kiki's skills directory."
        )
    except Exception as e:
        return f"Error installing Smithery skill: {e}"


# ============================================================================
# Senior Citizen Mode Tools (care plan + family email)
# ============================================================================

def _normalize_spoken_care_schedule(value: Any) -> Any:
    """Turn the compact time shapes used by the speaking model into a schedule.

    The local tool instruction only exposes ``data:obj`` to stay prompt-light,
    so the model commonly emits ``"schedule": "19:20"`` or a separate
    ``"time": "08:00 AM"``. Both are unambiguous daily schedules and should
    not be rejected merely because the model omitted the canonical wrapper.
    Unknown shapes are returned unchanged so the strict store still rejects
    them instead of guessing.
    """
    if isinstance(value, dict) or value in (None, ""):
        return value
    text = str(value).strip()
    for fmt in ("%H:%M", "%I:%M %p", "%I %p"):
        try:
            hhmm = datetime.strptime(text.upper(), fmt).strftime("%H:%M")
            return {"kind": "daily", "value": hhmm}
        except ValueError:
            pass
    try:
        datetime.fromisoformat(text)
        return {"kind": "once", "value": text}
    except ValueError:
        return value


async def update_care_plan(section: str, action: str, data: Any = None) -> str:
    """Create or edit the caregiver care plan (voice-first) and re-sync workers.

    section: routine_event | care_session | reminder | exercise | family_contact |
             approved_music | approved_topics | senior | care_log
    action:  add | edit | remove | set
    data:    JSON object with the fields for that section/action, e.g.
             reminder add -> {"category":"medicine","message":"Take BP pill","schedule":{"kind":"daily","value":"09:00"}}
             reminder edit/remove -> {"id":"ab12cd34", ...changed fields}
             exercise add -> {"name":"Morning stretch","steps":["...","..."],"schedule":{"kind":"daily","value":"08:00"},"prescribed_by":"Dr. Rao"}
             family_contact add -> {"name":"Priya","email":"p@x.com","relationship":"daughter","notify_on":["alert","daily_summary"]}
             approved_music/approved_topics add|remove -> {"value":"old bollywood"}
             senior set -> {"name":"Amma","language":"hi","health_conditions":["diabetes"]}
             care_log add -> {"kind":"note","text":"..."}
    """
    try:
        from core.senior.care_plan import get_care_plan_store
        from core.senior.senior_care_manager import get_senior_care_manager
        plan = get_care_plan_store()

        section = (section or "").strip().lower()
        action = (action or "").strip().lower()
        section = {
            "reminders": "reminder",
            "exercises": "exercise",
            "family_contacts": "family_contact",
            "contacts": "family_contact",
            "profile": "senior",
            "care_logs": "care_log",
            "routine": "routine_event",
            "routines": "routine_event",
            "routine_events": "routine_event",
            "daily_routine": "routine_event",
            "session": "care_session",
            "active_session": "care_session",
        }.get(section, section)
        action = {
            "create": "add",
            "update": "edit",
            "delete": "remove",
        }.get(action, action)

        d: Dict[str, Any] = {}
        if isinstance(data, dict):
            d = data
        elif data and str(data).strip():
            try:
                d = json.loads(data)
            except json.JSONDecodeError:
                # Compatibility for the exact live failure where the model sent
                # a spoken reminder as raw Hindi instead of an object. Keep the
                # original Unicode text for natural Hindi TTS; LCD rendering
                # already derives Hinglish through romanize_hindi_for_lcd().
                if section == "reminder" and action == "add":
                    d = {"message": str(data).strip()}
                else:
                    return ("ERROR: No change was saved. 'data' must be a JSON "
                            f"object, not {data!r}.")
        if not isinstance(d, dict):
            return "ERROR: No change was saved. 'data' must be a JSON object."

        # Normalize the small, predictable variants the speaking model uses.
        # This includes both live failures: task/time keys and a bare HH:MM
        # schedule. Original Hindi text is preserved verbatim.
        if section == "reminder":
            if not d.get("message"):
                d["message"] = d.get("task") or d.get("text") or ""
            if not d.get("schedule") and d.get("time"):
                d["schedule"] = d["time"]
            if "schedule" in d:
                d["schedule"] = _normalize_spoken_care_schedule(d["schedule"])
        elif section == "exercise" and action == "add":
            if not d.get("schedule") and d.get("time"):
                d["schedule"] = d["time"]
            if "schedule" in d:
                d["schedule"] = _normalize_spoken_care_schedule(d["schedule"])
            # "Remind me to exercise" is a reminder, not a guided routine. The
            # model used the exercise section with reminder-shaped fields in the
            # live test; route that intent to a real schedulable reminder.
            if d.get("message") and not d.get("name") and not d.get("steps"):
                d = {
                    "category": "exercise",
                    "message": d["message"],
                    "schedule": d.get("schedule"),
                    "enabled": d.get("enabled", True),
                }
                section = "reminder"
        elif section == "routine_event":
            if not d.get("schedule") and d.get("time"):
                d["schedule"] = d["time"]
            if "schedule" in d:
                d["schedule"] = _normalize_spoken_care_schedule(d["schedule"])

        allowed_actions = {
            "reminder": {"add", "edit", "remove"},
            "exercise": {"add", "edit", "remove"},
            "family_contact": {"add", "remove"},
            "approved_music": {"add", "remove"},
            "approved_topics": {"add", "remove"},
            "senior": {"set", "edit"},
            "care_log": {"add"},
            "routine_event": {"add", "edit", "remove"},
            "care_session": {"start", "advance", "adapt", "set_vision", "complete", "cancel", "decline"},
        }
        if section not in allowed_actions:
            return (f"ERROR: No change was saved. Unknown section '{section}'. Use "
                    "reminder, exercise, family_contact, approved_music, "
                    "approved_topics, senior, care_log, routine_event, or care_session.")
        if action not in allowed_actions[section]:
            return (f"ERROR: No change was saved. Action '{action}' is not valid "
                    f"for section '{section}'.")

        if section == "reminder" and action == "add":
            if not str(d.get("message", "")).strip():
                return ("NEEDS_CLARIFICATION: No reminder was saved. Ask what the "
                        "reminder should say.")
            if not d.get("schedule"):
                return ("NEEDS_CLARIFICATION: No reminder was saved. Ask what time "
                        "it should run. For a daily reminder, save schedule as "
                        "{'kind':'daily','value':'HH:MM'} after the user answers.")
        if section == "routine_event" and action == "add":
            missing = [key for key in ("title", "schedule", "actions") if not d.get(key)]
            if missing:
                return ("NEEDS_CLARIFICATION: No routine event was saved. Missing: "
                        + ", ".join(missing) + ".")
        ok = True
        msg = ""
        scheduled_item_id = ""
        # A stable machine-readable result lets the orchestrating agent verify
        # the exact worker without scraping translated/human-facing prose.
        receipt_metadata: Dict[str, str] = {}

        if section == "reminder":
            if action == "add":
                item = plan.add_reminder(d.get("category", "other"), d.get("message", ""),
                                         d.get("schedule", {}), d.get("enabled", True))
                msg = f"Added {item['category']} reminder (id {item['id']})."
                scheduled_item_id = item["id"]
            elif action == "edit":
                ok = plan.edit_reminder(d.get("id", ""), **{k: v for k, v in d.items() if k != "id"})
                msg = "Reminder updated." if ok else "No reminder with that id."
            elif action == "remove":
                ok = plan.remove_reminder(d.get("id", ""))
                msg = "Reminder removed." if ok else "No reminder with that id."
        elif section == "exercise":
            if action == "add":
                item = plan.add_exercise(d.get("name", ""), d.get("steps", []),
                                         d.get("schedule"), d.get("prescribed_by", ""),
                                         d.get("enabled", True))
                msg = f"Added exercise '{item['name']}' (id {item['id']})."
                scheduled_item_id = item["id"] if item.get("schedule") else ""
            elif action == "edit":
                ok = plan.edit_exercise(d.get("id", ""), **{k: v for k, v in d.items() if k != "id"})
                msg = "Exercise updated." if ok else "No exercise with that id."
            elif action == "remove":
                ok = plan.remove_exercise(d.get("id", ""))
                msg = "Exercise removed." if ok else "No exercise with that id."
        elif section == "routine_event":
            if action == "add":
                item = plan.add_routine_event(
                    title=d.get("title", ""),
                    category=d.get("category", "other"),
                    schedule=d.get("schedule", {}),
                    actions=d.get("actions", []),
                    enabled=d.get("enabled", True),
                    source=d.get("source", "user"),
                    evidence=d.get("evidence", ""),
                    adaptation=d.get("adaptation"),
                    continuous_vision=d.get("continuous_vision", False),
                    objective=d.get("objective", ""),
                )
                scheduled_item_id = item["id"]
                receipt_metadata = {
                    "section": "routine_event",
                    "item_id": scheduled_item_id,
                }
                msg = (("Routine event already existed" if item.get("_existing")
                        else "Added routine event")
                       + f" '{item['title']}' (id {item['id']}).")
            elif action == "edit":
                event_id = d.get("id", "")
                ok = plan.edit_routine_event(
                    event_id, **{k: v for k, v in d.items() if k != "id"})
                scheduled_item_id = event_id if ok else ""
                if scheduled_item_id:
                    receipt_metadata = {
                        "section": "routine_event",
                        "item_id": scheduled_item_id,
                    }
                msg = (f"Routine event updated (id {event_id})." if ok
                       else "No routine event with that id.")
            elif action == "remove":
                ok = plan.remove_routine_event(d.get("id", ""))
                msg = "Routine event removed." if ok else "No routine event with that id."
        elif section == "care_session":
            if action == "start":
                state = plan.start_care_session(d.get("event_id", ""))
            elif action == "advance":
                state = plan.advance_care_session(d.get("response", ""))
            elif action == "adapt":
                state = plan.adapt_care_session(
                    d.get("response", ""), d.get("remaining_actions", []),
                    d.get("reason", ""))
            elif action == "set_vision":
                state = plan.set_care_session_vision(d.get("enabled"))
            elif action in {"complete", "cancel", "decline"}:
                status = {"cancel": "cancelled", "decline": "declined"}.get(
                    action, "completed")
                state = plan.finish_care_session(status, d.get("response", ""))
            msg = "Care session state: " + json.dumps(state, ensure_ascii=False)
        elif section == "family_contact":
            if action == "add":
                c = plan.add_family_contact(d.get("name", ""), d.get("email", ""),
                                            d.get("relationship", ""), d.get("notify_on"))
                msg = f"Added family contact {c['name']}."
            elif action == "remove":
                ok = plan.remove_family_contact(d.get("name", ""))
                msg = "Contact removed." if ok else "No contact with that name."
        elif section in ("approved_music", "approved_topics"):
            if action == "add":
                ok = plan.add_to_list(section, d.get("value", ""))
            elif action == "remove":
                ok = plan.remove_from_list(section, d.get("value", ""))
            msg = f"{section} updated." if ok else f"Could not update {section}."
        elif section == "senior":
            ok = plan.set_senior_profile(**d)
            msg = "Senior profile updated." if ok else "Could not update profile."
        elif section == "care_log":
            plan.add_care_log(d.get("kind", "note"), d.get("text", ""))
            msg = "Care log entry added."
        # Re-sync workers so schedule changes take effect immediately.
        sync_problem = ""
        try:
            mgr = get_senior_care_manager()
            schedule_changed = section in {"reminder", "exercise", "routine_event"}
            if schedule_changed and mgr is not None and mgr.is_active():
                mgr.sync_workers()
                if (scheduled_item_id and hasattr(mgr, "is_item_scheduled")
                        and not mgr.is_item_scheduled(scheduled_item_id)):
                    sync_problem = (
                        f"Care item {scheduled_item_id} was saved, but its worker "
                        "was not scheduled.")
        except Exception as e:
            print(f"[update_care_plan] sync failed: {e}")
            if scheduled_item_id:
                sync_problem = f"Care item was saved, but worker sync failed: {e}"

        if not ok:
            return f"ERROR: No change was saved. {msg or 'No change made.'}"
        if sync_problem:
            return f"PARTIAL: {sync_problem} Do not promise that it will trigger."
        result = f"SUCCESS: {msg or 'Care plan updated.'}"
        if receipt_metadata:
            result += ("\nRECEIPT_JSON: "
                       + json.dumps(receipt_metadata, ensure_ascii=False,
                                    separators=(",", ":")))
        return result
    except ValueError as e:
        return f"ERROR: No change was saved. {e}"
    except Exception as e:
        return f"ERROR: No change was saved. Error updating care plan: {e}"


async def get_care_plan(section: str = "") -> str:
    """Read the care plan. section: (empty)=overview | reminder | exercise | family_contact |
    approved_music | approved_topics | senior | care_log. Returns JSON."""
    try:
        from core.senior.care_plan import get_care_plan_store
        plan = get_care_plan_store()
        section = (section or "").strip().lower()
        alias = {"reminder": "reminders", "exercise": "exercises",
                 "family_contact": "family_contacts", "routine": "routine_events",
                 "routine_event": "routine_events", "routines": "routine_events",
                 "day": "daily_routine", "daily_routine": "daily_routine",
                 "care_session": "active_session", "session": "active_session",
                 "health": "health_trend", "heart_rate": "health_trend",
                 "health_measurements": "health_measurements"}
        if not section:
            heart_rate = plan.health_trend("heart_rate", days=7, limit=1)
            heart_rate_summary = {
                key: value for key, value in heart_rate.items() if key != "recent"}
            if heart_rate.get("recent"):
                heart_rate_summary["latest_measured_at"] = (
                    heart_rate["recent"][-1].get("measured_at"))
            overview = {
                "senior": plan.get_section("senior"),
                "reminders": plan.get_section("reminders"),
                "exercises": plan.get_section("exercises"),
                "routine_events": plan.get_section("routine_events"),
                "daily_routine": plan.daily_routine(),
                "active_session": plan.care_session_state(),
                "family_contacts": plan.get_section("family_contacts"),
                "approved_music": plan.get_section("approved_music"),
                "approved_topics": plan.get_section("approved_topics"),
                "heart_rate_trend": heart_rate_summary,
            }
            return json.dumps(overview, ensure_ascii=False, indent=2)
        key = alias.get(section, section)
        value = (plan.care_session_state() if key == "active_session"
                 else plan.daily_routine() if key == "daily_routine"
                 else plan.health_trend("heart_rate") if key == "health_trend"
                 else plan.get_section(key))
        if key == "care_log" and isinstance(value, list):
            value = value[-30:]  # only recent entries
        if value is None:
            return f"No such section '{section}'."
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error reading care plan: {e}"


async def heart_rate_measurement(action: str, site: str = "finger",
                                 seconds: Optional[float] = None,
                                 context: str = "", days: int = 7) -> str:
    """Operate the MAX30102 through structured phases; never emits dialogue."""
    try:
        from core.senior.care_plan import get_care_plan_store
        from core.senior.heart_rate import get_heart_rate_controller

        action = str(action or "").strip().lower()
        action = {"start": "prepare", "measure": "capture",
                  "read": "capture", "history": "trend"}.get(action, action)
        if action not in {"prepare", "capture", "cancel", "status", "trend"}:
            return json.dumps({
                "status": "error", "reason": "invalid_action",
                "allowed_actions": ["prepare", "capture", "cancel", "status", "trend"],
            })
        site = str(site or "finger").strip().lower()
        if site not in {"finger", "wrist"}:
            return json.dumps({"status": "error", "reason": "invalid_site",
                               "allowed_sites": ["finger", "wrist"]})

        plan = get_care_plan_store()
        if action == "trend":
            return json.dumps(plan.health_trend("heart_rate", days=days),
                              ensure_ascii=False)
        controller = get_heart_rate_controller()
        if action == "status":
            result = controller.state()
        elif action == "cancel":
            result = controller.cancel()
        elif action == "prepare":
            result = await asyncio.to_thread(controller.prepare, site)
        else:
            result = await asyncio.to_thread(controller.capture, seconds)
            if result.get("status") == "trusted_reading":
                session = plan.care_session_state()
                entry = plan.add_health_measurement(
                    measurement="heart_rate", value=result["bpm"],
                    unit="bpm", quality=result.get("quality", "FAIR"),
                    site=result.get("site", site), context=context,
                    signal=result.get("signal", {}),
                    routine_event_id=session.get("event_id", ""),
                    session_id=session.get("id", ""))
                result["record_id"] = entry["id"]
                result["measured_at"] = entry["measured_at"]
                result["trend"] = plan.health_trend("heart_rate", days=days, limit=7)
                plan.add_care_log(
                    "heart_rate", f"Trusted reading recorded: {result['bpm']} bpm "
                    f"({result.get('quality')}, {result.get('site')}).")
            elif result.get("status") not in {"cancelled", "cancel_requested"}:
                plan.add_care_log(
                    "heart_rate_attempt",
                    "Heart-rate attempt was not recorded as a trusted reading: "
                    + str(result.get("reason") or result.get("status")))
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    except Exception as exc:
        return json.dumps({"status": "error", "reason": "tool_failure",
                           "detail": str(exc)[:300]}, ensure_ascii=False)


async def get_care_schedule_status(item_id: str = "") -> str:
    """Return a verified worker receipt with the exact next trigger time."""
    try:
        from core.senior.senior_care_manager import get_senior_care_manager
        manager = get_senior_care_manager()
        if manager is None:
            return json.dumps({
                "status": "inactive", "item_id": item_id,
                "scheduled": False,
                "reason": "senior care manager is not initialized",
            })
        receipt = manager.schedule_receipt(str(item_id or "").strip())
        return json.dumps(receipt, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error", "item_id": item_id, "scheduled": False,
            "reason": str(e),
        }, ensure_ascii=False)


async def send_care_email(to: str, subject: str, body: str) -> str:
    """Send an email to family/caregivers via the configured Gmail MCP connection."""
    try:
        from core.self_extend import smithery_cli
        cfg = get_full_config().get("senior_mode", {}).get("email", {})
        connection = cfg.get("connection", "")
        tool = cfg.get("tool", "")
        arg_map = cfg.get("arg_map", {"to": "to", "subject": "subject", "body": "body"})
        if not connection or not tool:
            return ("Email is not configured. Ask an admin to set senior_mode.email.connection "
                    "and .tool in config.json after installing a Gmail MCP.")
        args = {arg_map.get("to", "to"): to,
                arg_map.get("subject", "subject"): subject,
                arg_map.get("body", "body"): body}
        result = smithery_cli.tool_call(connection, tool, json.dumps(args))
        return f"Email sent to {to}. ({result})"
    except Exception as e:
        return f"Error sending email: {e}"


async def alert_family(reason: str, urgency: str = "normal") -> str:
    """Alert the senior's family by email (use on distress/emergency/medical need).

    urgency: normal | urgent. Emails every family contact whose notify_on includes 'alert'.
    """
    try:
        from core.senior.care_plan import get_care_plan_store
        plan = get_care_plan_store()
        contacts = plan.contacts_for("alert")
        senior_name = plan.get_section("senior").get("name") or "the senior"
        plan.add_care_log("alert", f"[{urgency}] {reason}")
        if not contacts:
            return ("No family contacts with alert enabled. Logged the concern, but nobody "
                    "could be emailed. Ask to add a family contact.")
        prefix = "URGENT: " if str(urgency).lower() == "urgent" else ""
        subject = f"{prefix}Kiki alert about {senior_name}"
        body = (f"Kiki is flagging something about {senior_name}:\n\n{reason}\n\n"
                f"Urgency: {urgency}. Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}.")
        sent = []
        for c in contacts:
            res = await send_care_email(c["email"], subject, body)
            sent.append(f"{c.get('name', c['email'])}: {res}")
        return "Family alerted. " + " | ".join(sent)
    except Exception as e:
        return f"Error alerting family: {e}"


# ============================================================================
# Tool Schemas (OpenAI function-calling format)
# ============================================================================

_WHATSAPP_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_contacts",
            "description": "Search WhatsApp contacts by name or phone number.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Name or phone number."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_messages",
            "description": "Read bounded WhatsApp messages with optional date, sender, chat, text, pagination and surrounding-context filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "after": {"type": "string", "description": "Optional ISO-8601 lower timestamp bound."},
                    "before": {"type": "string", "description": "Optional ISO-8601 upper timestamp bound."},
                    "sender_phone_number": {"type": "string", "description": "Optional sender number/JID."},
                    "chat_jid": {"type": "string", "description": "Optional direct or group chat JID."},
                    "query": {"type": "string", "description": "Optional text contained in the message."},
                    "limit": {"type": "integer", "description": "Results per page, 1-50."},
                    "page": {"type": "integer", "description": "Zero-based page."},
                    "include_context": {"type": "boolean", "description": "Include nearby messages."},
                    "context_before": {"type": "integer", "description": "Nearby messages before each match, 0-10."},
                    "context_after": {"type": "integer", "description": "Nearby messages after each match, 0-10."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_chats",
            "description": "List WhatsApp chats with JIDs, names, activity time and optional last-message metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Optional name or JID filter."},
                    "limit": {"type": "integer", "description": "Results per page, 1-50."},
                    "page": {"type": "integer", "description": "Zero-based page."},
                    "include_last_message": {"type": "boolean"},
                    "sort_by": {"type": "string", "enum": ["last_active", "name"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chat",
            "description": "Get one WhatsApp chat by exact JID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_jid": {"type": "string"},
                    "include_last_message": {"type": "boolean"},
                },
                "required": ["chat_jid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_direct_chat_by_contact",
            "description": "Find a direct WhatsApp chat from a contact phone number.",
            "parameters": {
                "type": "object",
                "properties": {"sender_phone_number": {"type": "string"}},
                "required": ["sender_phone_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_contact_chats",
            "description": "List direct and group WhatsApp chats involving a contact JID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "jid": {"type": "string"},
                    "limit": {"type": "integer"},
                    "page": {"type": "integer"},
                },
                "required": ["jid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_last_interaction",
            "description": "Get the most recent WhatsApp message involving a contact JID.",
            "parameters": {
                "type": "object",
                "properties": {"jid": {"type": "string"}},
                "required": ["jid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_message_context",
            "description": "Read messages immediately before and after one WhatsApp message ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string"},
                    "before": {"type": "integer"},
                    "after": {"type": "integer"},
                },
                "required": ["message_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a WhatsApp text after the user explicitly requests it. Recipient may be a unique contact name, phone number, or group JID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["recipient", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_file",
            "description": "Send an existing local image, video, raw audio or document through WhatsApp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "media_path": {"type": "string", "description": "Absolute local file path."},
                },
                "required": ["recipient", "media_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_audio_message",
            "description": "Send an existing local audio file as a WhatsApp voice message; non-Opus audio is converted with ffmpeg.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "media_path": {"type": "string", "description": "Absolute local audio path."},
                },
                "required": ["recipient", "media_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_media",
            "description": "Download media attached to a WhatsApp message and return its local path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string"},
                    "chat_jid": {"type": "string"},
                },
                "required": ["message_id", "chat_jid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_whatsapp_image",
            "description": (
                "Look at an image sent in a WhatsApp message and describe it. Use this "
                "when a message's content begins with '[image - Message ID: X - Chat JID: Y]'. "
                "Downloads the media and reads it with the vision model."),
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string", "description": "Message ID from the [image - ...] marker."},
                    "chat_jid": {"type": "string", "description": "Chat JID from the [image - ...] marker."},
                    "question": {"type": "string", "description": "Optional specific question about the image."},
                },
                "required": ["message_id", "chat_jid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_voice_note",
            "description": (
                "Record the microphone for a few seconds and return the wav path, for "
                "sending as a WhatsApp voice note. Follow it with send_audio_message."),
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "integer", "description": "How long to record, 1-60 (default 10)."},
                },
                "required": [],
            },
        },
    },
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "complex_query",
            "description": (
                "Handle a request that needs SEVERAL steps or touches WhatsApp, email, "
                "Gmail, Notion, reminders, or files. Hands the whole request to a fast "
                "agent that runs the steps itself and reports the result. Use it for "
                "anything like 'check my messages and remind me', 'research X and send "
                "it on WhatsApp', 'send this file to <group>', 'draft an email about X'. "
                "Pass the user's request verbatim."),
            "parameters": {
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "The user's full request, in their own words.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional extra detail from the conversation that the agent needs.",
                    },
                },
                "required": ["request"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information using Exa. Use for latest news, facts, weather, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "search_range": {"type": "string", "enum": ["today", "last 5 days", "last month"], "description": "Time range"}
                },
                "required": ["query"]
            }
        }
    },
    *_WHATSAPP_TOOL_SCHEMAS,
    {
        "type": "function",
        "function": {
            "name": "execute_shell_command",
            "description": "Execute a shell command. Useful for checking system status like temperature, disk space, etc.Install all python packagaes in  /usr/bin/python3 venv.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local time and date.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "MUST use for past conversations, memories, or what was discussed. Never guess.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "memory topic"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_background_research",
            "description": "Save verified dated background research after evidence gathering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "summary": {"type": "string"},
                    "details": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "tools_used": {"type": "array", "items": {"type": "string"}},
                    "mode": {"type": "string", "enum": ["light_research", "deep_research"]}
                },
                "required": ["topic", "summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_next_turn_note",
            "description": "Set, replace, or clear the single thought Kiki may naturally use next conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "reason": {"type": "string"},
                    "expires_hours": {"type": "number"},
                    "action": {"type": "string", "enum": ["set", "clear"]}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_open_question",
            "description": "Save one unresolved curiosity for possible future investigation.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_open_question",
            "description": "Mark a matching curiosity question resolved.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "switch_voice",
            "description": "Switch Kiki's speaking voice to a named voice (e.g. arinabh, modi, alex, ansh, normal). Use 'default' to reset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "voice": {"type": "string", "description": "voice name, or 'default' to reset"}
                },
                "required": ["voice"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "switch_mode",
            "description": "Switch Kiki to a configured personality mode, changing its system prompt and mode voice.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "description": "configured mode name, such as default or funny"}
                },
                "required": ["mode"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_followups",
            "description": "Enable or disable listening for follow-up speech after each answer. When disabled, Kiki listens only after the wake word.",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean", "description": "true to enable follow-ups, false to require the wake word"}
                },
                "required": ["enabled"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_volume",
            "description": "Increase, decrease, or set Kiki's speaker volume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["increase", "decrease", "set"]},
                    "amount": {"type": "integer", "minimum": 0, "maximum": 100, "description": "optional step size, or exact percentage for set"}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "play_music",
            "description": "Search YouTube and play one exact song/video. Returns its resolved title and permanent YouTube watch URL. Use play_last_song for the previous request and play_liked_songs for the saved playlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "song": {"type": "string", "description": "Song name, artist, or genre to search and play"}
                },
                "required": ["song"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "like_current_song",
            "description": "Add the exact YouTube video currently playing to the persistent liked songs playlist. Use for 'like this song' or 'add this song to my liked songs'.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "play_liked_songs",
            "description": "Play all saved liked songs as a playlist that automatically advances.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "play_last_song",
            "description": "Replay the exact YouTube video that was played most recently.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "control_music",
            "description": "Control current music playback. Pause/resume works on any song; next/previous moves within the liked-songs playlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["pause", "resume", "next", "previous"]}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Set a countdown timer. Convert the user's duration to total seconds when possible; common strings such as 'five minutes' are also accepted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {
                        "description": "Timer duration as total seconds (preferred), or a phrase such as '5 minutes'.",
                        "anyOf": [{"type": "integer"}, {"type": "string"}]
                    }
                },
                "required": ["duration"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_knowledge",
            "description": "Store or retrieve information from long-term memory. Categories: people, environments, learnings, experiences, facts, personality, self.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["people", "environments", "learnings", "experiences", "facts", "personality", "self"]},
                    "action": {"type": "string", "enum": ["add", "update", "remove", "get", "search"]},
                    "key": {"type": "string", "description": "Identifier - person name, place name, topic, etc."},
                    "value": {"type": "string", "description": "The value to store"},
                    "attribute": {"type": "string", "description": "For people: appearance, character, routine, interests, current_ongoing, notes"}
                },
                "required": ["category", "action", "key"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember_me",
            "description": "Remember a person's face. Call when user asks to be remembered.",
            "parameters": {
                "type": "object",
                "properties": {
                    "person_name": {"type": "string", "description": "Name of the person to remember"}
                },
                "required": ["person_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "track_person",
            "description": "Set a specific person to track/follow with the robot's neck.",
            "parameters": {
                "type": "object",
                "properties": {
                    "person_name": {"type": "string", "description": "Name of the person to track"}
                },
                "required": ["person_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_person_real_name",
            "description": "When a person you met as a stranger (a temporary 'guest_...' name) tells you their REAL name and confirms it, call this to rename them everywhere so you recognise them by name from now on. Confirm the spelling first, then call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "temp_name": {"type": "string", "description": "The current temporary name (e.g. 'guest_20260722_1530')"},
                    "real_name": {"type": "string", "description": "The person's confirmed real name"}
                },
                "required": ["temp_name", "real_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_worker",
            "description": "Schedule an autonomous background worker task. Workers are LLM-powered agents that run at specific times, on events (startup/shutdown/sleep/wake/after_response/face_detected), or on recurring intervals. Workers have full tool access and retry on failure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Human-readable name of the worker task"},
                    "task_description": {"type": "string", "description": "Detailed description of what the worker should do. Be specific about the task, tools to use, and desired outcome."},
                    "trigger_type": {"type": "string", "enum": ["scheduled_time", "event", "recurring"], "description": "When to trigger: scheduled_time (specific datetime), event (lifecycle event), or recurring (every N seconds)"},
                    "trigger_value": {"type": "string", "description": "For scheduled_time: ISO datetime (e.g. '2026-03-04T17:00:00'). For event: event name (startup/shutdown/sleep/wake/after_response/face_detected). For recurring: interval in seconds."},
                    "conditions": {"type": "string", "description": "Optional JSON array of conditions. Example: '[{\"condition_type\": \"person_seen\", \"params\": {\"person\": \"Alex\", \"within_minutes\": 60}}]'"}
                },
                "required": ["name", "task_description", "trigger_type", "trigger_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_worker",
            "description": "Cancel a pending or active worker by its ID or name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "worker_id": {"type": "string", "description": "ID or name of the worker to cancel"}
                },
                "required": ["worker_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_workers",
            "description": "List all active and pending workers with their status.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": "Execute Python code in a subprocess and return the output. Useful for calculations, data processing, file operations, or any task that benefits from code execution.All python code in run in the /usr/bin/python3 venv.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "self_extend_list_skills",
            "description": "List all Kiki skills currently installed in the skills directory. Use this to browse what capabilities Kiki already has before creating new ones.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "self_extend_create_skill",
            "description": "Create a new Kiki skill by saving a SKILL.md file. Skills are reusable instruction sets that guide how Kiki handles specific tasks. Always check existing skills first to avoid duplicates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Unique snake_case skill folder name, e.g. 'data_analysis'"},
                    "skill_md_content": {"type": "string", "description": "Full SKILL.md content with YAML frontmatter (description, triggers, instructions, examples)"},
                    "extra_files": {"type": "string", "description": "Optional JSON object string of extra files to include: '{\"helper.py\": \"code\"}'"}  
                },
                "required": ["name", "skill_md_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "self_extend_run_task",
            "description": (
                "Run the full Kiki self-extension agent for a complex multi-step goal. "
                "The agent uses Smithery CLI and can: search/connect MCP servers, "
                "search/install skills, create custom MCP servers, and read/write files autonomously. "
                "Use for goals like: 'Find and connect a web search MCP server', "
                "'Install the best coding skill from Smithery', "
                "'Create a custom MCP server and install a related skill'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "Natural language description of what to accomplish"}
                },
                "required": ["goal"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "self_extend_view_smithery_skill",
            "description": "View the FULL content of a Smithery skill without installing it. Returns the complete SKILL.md documentation. identifier format: 'namespace/slug' e.g. 'langfuse/skill-developer'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "description": "Skill identifier in 'namespace/slug' format"}
                },
                "required": ["identifier"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "self_extend_install_smithery_skill",
            "description": (
                "Download and install a Smithery skill into Kiki's local skills directory using the CLI. "
                "Fetches the COMPLETE skill documentation (not just a brief summary). "
                "Use self_extend_search_smithery_skills first to find the identifier. "
                "identifier format: 'namespace/slug', e.g. 'langfuse/skill-developer'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "description": "Skill identifier in 'namespace/slug' format, from search results"}
                },
                "required": ["identifier"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_gmail",
            "description": (
                "Fast, compact Gmail search/inbox read. Returns only message IDs, "
                "thread IDs, time, sender, recipient, subject, short snippet, labels "
                "and attachment names—never raw HTML, MIME payloads or headers. "
                "Use this directly for all routine Gmail reads; repeat calls are "
                "allowed because inbox data can change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Optional Gmail search query, e.g. 'is:unread', "
                            "'from:name@example.com', or 'newer_than:2d'.")
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of compact results, 1-20 (default 5)."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_gmail_message",
            "description": (
                "Read one Gmail message selected by read_gmail. Returns clean bounded "
                "text plus essential metadata; strips HTML, raw headers and MIME data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {
                        "type": "string",
                        "description": "Message ID returned by read_gmail."
                    },
                    "max_body_chars": {
                        "type": "integer",
                        "description": "Maximum clean body characters, 500-12000 (default 5000)."
                    }
                },
                "required": ["message_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_gmail_thread",
            "description": (
                "Read a Gmail thread selected by read_gmail as a bounded list of "
                "clean messages. Excludes raw HTML, headers and MIME payloads."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thread_id": {
                        "type": "string",
                        "description": "Thread ID returned by read_gmail."
                    },
                    "max_messages": {
                        "type": "integer",
                        "description": "Maximum messages, 1-20 (default 10)."
                    },
                    "max_body_chars": {
                        "type": "integer",
                        "description": "Maximum clean characters per message, 300-8000 (default 2000)."
                    }
                },
                "required": ["thread_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_notion",
            "description": (
                "Fast, compact search across the connected Notion workspace. "
                "Returns a small ranked list of IDs, titles, types, timestamps, "
                "short highlights and URLs. Use read_notion only for the selected result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "One focused Notion topic or question."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results, 1-10 (default 5)."
                    },
                    "highlight_chars": {
                        "type": "integer",
                        "description": "Maximum highlight characters per result, 0-400 (default 240)."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_notion",
            "description": (
                "Read one Notion page, database or data source selected by "
                "search_notion. Returns title, URL, type and bounded enhanced-Markdown "
                "content without the raw MCP envelope."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {
                        "type": "string",
                        "description": "Notion ID or URL returned by search_notion."
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum content characters, 500-16000 (default 6000)."
                    }
                },
                "required": ["entity_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "self_extend_search_mcp",
            "description": "Search the Smithery MCP server registry via CLI. Returns server IDs and descriptions. Use self_extend_mcp_add to connect one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What kind of MCP server to find"},
                    "page": {"type": "integer", "description": "Page number (default 1)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "self_extend_mcp_add",
            "description": "Add/connect an MCP server via Smithery CLI (smithery mcp add). Use after searching to connect the server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {"type": "string", "description": "Server registry ID (e.g. 'exa') or full connection URL"}
                },
                "required": ["server"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "self_extend_mcp_list_connections",
            "description": "List all currently connected MCP servers (smithery mcp list). Shows connection IDs for use with tool_list/tool_call.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "self_extend_mcp_remove",
            "description": "Remove one or more MCP server connections (smithery mcp remove).",
            "parameters": {
                "type": "object",
                "properties": {
                    "connection_ids": {"type": "string", "description": "Space-separated connection IDs to remove (from mcp_list_connections)"}
                },
                "required": ["connection_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "self_extend_tool_find",
            "description": "Search tools by name or intent from a connected MCP server (smithery tool find).",
            "parameters": {
                "type": "object",
                "properties": {
                    "connection": {"type": "string", "description": "MCP connection ID from mcp_list_connections"},
                    "query": {"type": "string", "description": "Tool name or intent to search for"}
                },
                "required": ["connection"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "self_extend_tool_list",
            "description": "List all tools from a connected MCP server (smithery tool list).",
            "parameters": {
                "type": "object",
                "properties": {
                    "connection": {"type": "string", "description": "MCP connection ID from mcp_list_connections"}
                },
                "required": ["connection"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "self_extend_tool_call",
            "description": "Call a specific tool from a connected MCP server (smithery tool call). Can execute real actions like web search, file operations, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "connection": {"type": "string", "description": "MCP connection ID"},
                    "tool": {"type": "string", "description": "Tool name to call"},
                    "args_json": {"type": "string", "description": "JSON string of arguments e.g. '{\"query\": \"hello\"}' (optional)"}
                },
                "required": ["connection", "tool"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "self_extend_create_mcp_server",
            "description": "Generate, save, and register a custom FastMCP Python MCP server from scratch. Only use when no suitable existing server exists on Smithery.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_name": {"type": "string", "description": "snake_case server name"},
                    "description": {"type": "string", "description": "What this MCP server does"},
                    "tools_json": {"type": "string", "description": "JSON array: [{\"name\":\"...\",\"description\":\"...\",\"params\":[{\"name\":\"p\",\"type\":\"str\"}],\"impl\":\"return 'ok'\"}]"}
                },
                "required": ["server_name", "description", "tools_json"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_care_plan",
            "description": "Care-agent only: create/edit the person's daily routine as scheduled multi-action routine_events, manage active care sessions, and maintain legacy reminders/exercises/contacts. Changes sync immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {"type": "string", "description": "routine_event | care_session | reminder | exercise | family_contact | approved_music | approved_topics | senior | care_log"},
                    "action": {"type": "string", "description": "routine_event: add/edit/remove; care_session: start/advance/adapt/set_vision/complete/cancel/decline; others: add/edit/remove/set"},
                    "data": {"type": "object", "description": "routine_event add: {title,objective,category,schedule:{kind,value},actions:[{type,instruction,needs_response,success_signal,on_concern,vital_type?}],continuous_vision:boolean,source,evidence,adaptation}. Actions are adaptable goals, not fixed dialogue. Action types include speak,check_in,guided_step,measure_vital,memory_activity,observe,log,notify_caregiver. For MAX30102 use category=vitals and {type:measure_vital,vital_type:heart_rate}. care_session adapt changes only the live session unless routine_event is explicitly edited. Preserve original language."}
                },
                "required": ["section", "action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_care_plan",
            "description": "Care agent: read the person's daily routine, active care session, reminders, exercises and contacts. Empty section returns an overview.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {"type": "string", "description": "empty=overview | routine_event | care_session | heart_rate | health_measurements | reminder | exercise | family_contact | approved_music | approved_topics | senior | care_log"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_care_schedule_status",
            "description": "Care agent: verify that a care-plan event has an active worker and return its exact next_trigger_at receipt. Must be called after every scheduled care-plan add/edit before claiming success.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "Care-plan item id returned by update_care_plan; empty lists all active care workers"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "heart_rate_measurement",
            "description": "Care-agent-only MAX30102 measurement state machine. Use prepare after asking the person to uncover the sensor; after ready_for_contact, guide placement and stillness, then capture. Only trusted_reading is recorded/trended. The tool returns data and never speaks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["prepare", "capture", "cancel", "status", "trend"], "description": "Measurement phase"},
                    "site": {"type": "string", "enum": ["finger", "wrist"], "description": "Sensor placement; finger is default"},
                    "seconds": {"type": "number", "description": "Optional 10-60 second capture override; omit for validated preset"},
                    "context": {"type": "string", "description": "Short factual context such as resting or recently walked; stored with a trusted reading"},
                    "days": {"type": "integer", "description": "Trend window, 1-365 days"}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_care_email",
            "description": "Send an email to family/caregivers via the configured Gmail MCP. Used by the daily-summary worker; prefer alert_family for emergencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body (plain text)"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "alert_family",
            "description": "Senior mode: email the senior's family when you detect distress, a fall, a medical need, or an emergency. Emails all alert-enabled contacts and logs the concern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "What happened / why you are alerting"},
                    "urgency": {"type": "string", "description": "normal | urgent"}
                },
                "required": ["reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "look_at_scene",
            "description": (
                "Look through your camera eyes RIGHT NOW to answer anything about what you can "
                "currently SEE: someone's appearance or outfit ('how do I look?', 'does my shirt "
                "look good?'), an object they are holding or showing ('what is this?', 'should I "
                "buy this?'), reading text/labels in view, or the room/surroundings ('describe "
                "what you see', 'what's around you?'). Call this the MOMENT a question needs your "
                "live vision. Output ONLY this call and nothing else."
            ),
            "parameters": {"type": "object", "properties": {}}
        }
    },
]

# ============================================================================
# Tool Dispatch
# ============================================================================

_TOOL_SCHEMAS_BY_NAME = {
    tool.get("function", {}).get("name"): tool.get("function", {})
    for tool in TOOLS
}


_TRUE_WORDS = {"true", "yes", "on", "1"}
_FALSE_WORDS = {"false", "no", "off", "0"}


def _coerce_scalar(value, expected):
    """Return ``value`` converted to ``expected``, or ``None`` if it can't be.

    Only LOSSLESS, unambiguous conversions: a JSON string that spells exactly
    one number/boolean. `"sixty"` and `"60%"` still fail, so a genuinely wrong
    argument still produces a corrective error instead of a silent guess.
    """
    if expected == "integer":
        if isinstance(value, float) and not isinstance(value, bool) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            try:
                return int(text, 10)
            except ValueError:
                try:
                    number = float(text)
                except ValueError:
                    return None
                return int(number) if number.is_integer() else None
    elif expected == "number":
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
    elif expected == "boolean":
        if isinstance(value, str):
            text = value.strip().casefold()
            if text in _TRUE_WORDS:
                return True
            if text in _FALSE_WORDS:
                return False
        elif isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
            return bool(value)
    return None


def validate_tool_arguments(name: str, arguments: dict) -> tuple[bool, str]:
    """Schema gate shared by speaking, workers and the idle mind.

    Scalars are COERCED IN PLACE before they are judged. The speaking model is
    only ever shown `{"param_name": "value"}` in the tools instruction, so it
    quotes numbers — and `adjust_volume({"action":"set","amount":"60"})` was
    hard-rejected here even though the handler's own `int(amount)` would have
    taken it (live failure 2026-07-29 00:07: Kiki said "did I mess that up?"
    and the volume never moved). Coercion happens on the caller's dict, which
    is the same object every call site then dispatches with.

    Enums are deliberately NOT coerced: a category outside the allowed set is a
    real mistake, and the error text is what teaches the model the right one.
    """
    fn = _TOOL_SCHEMAS_BY_NAME.get(name)
    if fn is None:
        return False, f"Unknown tool '{name}'"
    if not isinstance(arguments, dict):
        return False, "arguments must be an object"

    # Backward compatibility for an already-warm speaking prompt and for the
    # exact senior-mode failure seen in the field. The updated schema asks for
    # an object, but an old model turn may still send either stringified JSON or
    # the raw reminder sentence. Convert unambiguously here, then let the care
    # handler ask for a missing schedule instead of failing on Unicode text.
    if name == "update_care_plan" and isinstance(arguments.get("data"), str):
        raw_data = arguments["data"].strip()
        try:
            parsed_data = json.loads(raw_data)
        except json.JSONDecodeError:
            parsed_data = None
        if isinstance(parsed_data, dict):
            arguments["data"] = parsed_data
        elif (str(arguments.get("section", "")).strip().lower()
              in ("reminder", "reminders")
              and str(arguments.get("action", "")).strip().lower()
              in ("add", "create") and raw_data):
            arguments["data"] = {"message": raw_data}

    schema = fn.get("parameters", {}) or {}
    props = schema.get("properties", {}) or {}
    missing = [key for key in schema.get("required", []) if key not in arguments]
    if missing:
        return False, f"missing required argument(s): {', '.join(missing)}"
    unknown = [key for key in arguments if key not in props]
    if unknown:
        return False, f"unsupported argument(s): {', '.join(unknown)}"
    for key, value in arguments.items():
        spec = props.get(key, {})
        expected = spec.get("type")
        if "enum" not in spec and expected in ("integer", "number", "boolean"):
            coerced = _coerce_scalar(value, expected)
            if coerced is not None:
                arguments[key] = value = coerced
        if "enum" in spec and value not in spec["enum"]:
            return False, f"{key} must be one of {spec['enum']}, got {value!r}"
        valid = (
            expected is None
            or expected == "string" and isinstance(value, str)
            or expected == "integer" and isinstance(value, int) and not isinstance(value, bool)
            or expected == "number" and isinstance(value, (int, float)) and not isinstance(value, bool)
            or expected == "boolean" and isinstance(value, bool)
            or expected == "array" and isinstance(value, list)
            or expected == "object" and isinstance(value, dict)
        )
        if not valid:
            return False, f"{key} must be {expected}"
    return True, "ok"

async def look_at_scene() -> str:
    """Signal-only tool: the speaking path intercepts a ``look_at_scene`` call
    BEFORE execution and routes the turn to the Groq live-image model (see
    core/vision/instant_vision.py). This handler is a harmless no-op for any
    other caller (e.g. the background brain) so the tool never errors."""
    return "(live vision handled by the instant-vision path)"


# Map tool names to async handler functions
_ASYNC_TOOL_HANDLERS = {
    "look_at_scene": look_at_scene,
    "search_web": search_web,
    "execute_shell_command": execute_shell_command,
    "get_current_time": get_current_time,
    "recall_memory": recall_memory,
    "save_background_research": save_background_research,
    "set_next_turn_note": set_next_turn_note,
    "add_open_question": add_open_question,
    "resolve_open_question": resolve_open_question,
    "switch_voice": switch_voice,
    "switch_mode": switch_mode,
    "set_followups": set_followups,
    "adjust_volume": adjust_volume,
    "play_music": play_music,
    "like_current_song": like_current_song,
    "play_liked_songs": play_liked_songs,
    "play_last_song": play_last_song,
    "control_music": control_music,
    "set_timer": set_timer,
    "update_knowledge": update_knowledge,
    "remember_me": remember_me,
    "track_person": track_person,
    "set_person_real_name": set_person_real_name,
    "schedule_worker": schedule_worker,
    "cancel_worker": cancel_worker,
    "list_workers": list_workers,
    "execute_python_code": execute_python_code,
    # Self-extend tools — all CLI-backed
    "self_extend_list_skills": self_extend_list_skills,
    "self_extend_create_skill": self_extend_create_skill,
    "self_extend_search_mcp": self_extend_search_mcp,
    "self_extend_mcp_add": self_extend_mcp_add,
    "self_extend_mcp_list_connections": self_extend_mcp_list_connections,
    "self_extend_mcp_remove": self_extend_mcp_remove,
    "self_extend_tool_find": self_extend_tool_find,
    "self_extend_tool_list": self_extend_tool_list,
    "self_extend_tool_call": self_extend_tool_call,
    "read_gmail": read_gmail,
    "read_gmail_message": read_gmail_message,
    "read_gmail_thread": read_gmail_thread,
    "search_notion": search_notion,
    "read_notion": read_notion,
    # Bundled local WhatsApp MCP (all calls stay off the speaking thread)
    "search_contacts": search_contacts,
    "list_messages": list_messages,
    "list_chats": list_chats,
    "get_chat": get_chat,
    "get_direct_chat_by_contact": get_direct_chat_by_contact,
    "get_contact_chats": get_contact_chats,
    "get_last_interaction": get_last_interaction,
    "get_message_context": get_message_context,
    "send_message": send_message,
    "send_file": send_file,
    "send_audio_message": send_audio_message,
    "download_media": download_media,
    "read_whatsapp_image": read_whatsapp_image,
    "record_voice_note": record_voice_note,
    # The fast multi-step research + action agent (core/brain/action_agent.py)
    "complex_query": complex_query,
    "self_extend_create_mcp_server": self_extend_create_mcp_server,
    "self_extend_run_task": self_extend_run_task,
    "self_extend_search_smithery_skills": self_extend_search_smithery_skills,
    "self_extend_view_smithery_skill": self_extend_view_smithery_skill,
    "self_extend_install_smithery_skill": self_extend_install_smithery_skill,
    # Senior citizen mode
    "update_care_plan": update_care_plan,
    "get_care_plan": get_care_plan,
    "get_care_schedule_status": get_care_schedule_status,
    "heart_rate_measurement": heart_rate_measurement,
    "send_care_email": send_care_email,
    "alert_family": alert_family,
}


# Default hard ceiling for one synchronous tool execution. Per-tool overrides
# exist because a multi-step agent legitimately runs longer than a lookup, and
# silently truncating it produces a half-done action reported as success.
_EXEC_TIMEOUT_DEFAULT = 30
_EXEC_TIMEOUT_BY_TOOL = {
    # Must exceed action_agent.deadline_seconds so the agent's own deadline
    # wins and it can report what it managed to finish.
    "complex_query": 90,
    "record_voice_note": 70,   # bounded by record_clip's own 60s cap
}


def execute_tool(name: str, arguments: dict) -> str:
    """
    Execute a tool SYNCHRONOUSLY (for the LLM tool-calling loop in core/llm.py).
    Runs the async handler in a new event loop if needed.
    """
    handler = _ASYNC_TOOL_HANDLERS.get(name)
    if handler is None:
        return f"Error: Unknown tool '{name}'"
    valid, reason = validate_tool_arguments(name, arguments)
    if not valid:
        return f"Error: Invalid arguments for '{name}': {reason}"

    with get_recorder().span("tool", name=name, args=_obs_trunc(arguments)) as s:
        try:
            # Try to use the running loop
            try:
                asyncio.get_running_loop()
                # We're inside an event loop, run in a worker thread.
                import concurrent.futures
                pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                try:
                    future = pool.submit(asyncio.run, handler(**arguments))
                    result = future.result(timeout=_EXEC_TIMEOUT_BY_TOOL.get(
                        name, _EXEC_TIMEOUT_DEFAULT))
                finally:
                    # wait=False is LOAD-BEARING. `with ThreadPoolExecutor()`
                    # calls shutdown(wait=True) on exit, so a timed-out tool
                    # still blocked here until its worker finished — a slow
                    # WhatsApp media fetch turned a 30s cap into a 108s hang,
                    # and a hung turn means wake-word detection never re-arms.
                    # Abandoning the thread lets the timeout actually bound us.
                    pool.shutdown(wait=False)
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                result = asyncio.run(handler(**arguments))
            s["result"] = _obs_trunc(result)
            return result
        except Exception as e:
            s["error"] = str(e)
            return f"Error executing tool '{name}': {str(e)}"


async def execute_tool_async(name: str, arguments: dict) -> str:
    """Execute a tool asynchronously for autonomous background agents."""
    handler = _ASYNC_TOOL_HANDLERS.get(name)
    if handler is None:
        return f"Error: Unknown tool '{name}'"
    valid, reason = validate_tool_arguments(name, arguments)
    if not valid:
        return f"Error: Invalid arguments for '{name}': {reason}"

    with get_recorder().span("tool", name=name, args=_obs_trunc(arguments)) as s:
        try:
            result = await handler(**arguments)
            s["result"] = _obs_trunc(result)
            return result
        except Exception as e:
            s["error"] = str(e)
            return f"Error executing tool '{name}': {str(e)}"


def get_tool_descriptions() -> Dict[str, str]:
    """Get a tool-name to description map for background-agent context."""
    return {
        tool["function"]["name"]: tool["function"]["description"]
        for tool in TOOLS
    }


def get_detailed_tool_descriptions(excluded_names=None) -> str:
    """Format tool schemas, optionally omitting tools unavailable to a caller."""
    excluded = set(excluded_names or ())
    lines = []
    for tool in TOOLS:
        fn = tool.get("function", {})
        name = fn.get("name")
        if name in excluded:
            continue
        desc = fn.get("description")
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])

        param_desc = []
        if params:
            for p_name, p_info in params.items():
                p_type = p_info.get("type", "any")
                p_desc = p_info.get("description", "")
                req_str = "required" if p_name in required else "optional"
                param_desc.append(f"    * {p_name} ({p_type}, {req_str}): {p_desc}")

        tool_line = f"- {name}: {desc}"
        if param_desc:
            tool_line += "\n  Parameters:\n" + "\n".join(param_desc)
        lines.append(tool_line)
    return "\n".join(lines)
