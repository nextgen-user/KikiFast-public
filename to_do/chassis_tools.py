"""
PARKED chassis tool layer — moved out of tools_and_config/tools.py.

NOT imported by the running app. Kept verbatim so the wheeled-chassis tools can be
reintegrated if wheels are added back. See to_do/README.md.

Contains:
  - VALID_MOTOR_ACTIONS, ZMQ_MOTOR_HOST/PORT
  - KikiMotorClient   (ZMQ REQ/REP client to the motor server on :5557, owned by
                       ~/Kiki/navigation/hailo_follower_webcam_only.py)
  - move(steps)       tool
  - dance(song, steps) tool
  - follow_me(duration) tool
  - _music_safety_monitor(proc)  (was nested in play_music — stops music if an
                                  ultrasonic obstacle comes within 10 cm)

REINTEGRATION CHECKLIST (when wheels return):
  1. Move these back into tools_and_config/tools.py (or import from here).
  2. They depend on helpers that stayed in tools.py:
        set_neck_active (formerly set_motor_relay), should_skip_followup,
        _get_controller, get_full_config
     and on the controller method KikiController.set_full_body_movement (kept in
     kiki_control_client.py).
  3. Re-add each tool's schema to the TOOLS list and its entry to
     _ASYNC_TOOL_HANDLERS ("move", "dance", "follow_me").
  4. Restore the safety-monitor thread launch inside play_music's try_play():
        threading.Thread(target=_music_safety_monitor, args=(proc,), daemon=True).start()
"""

import asyncio
import os
import signal
import subprocess
import threading
import time

import zmq as _zmq

# Pull the helpers that stayed in tools.py (available once reintegrated).
from tools_and_config.config_loader import get_full_config

ZMQ_MOTOR_HOST = "127.0.0.1"
ZMQ_MOTOR_PORT = 5557

# Full set of valid motor actions mirroring every function in motor_control.py
VALID_MOTOR_ACTIONS = frozenset([
    "forward", "backward", "stop",
    "turn_left", "turn_right",
    "strafe_left", "strafe_right",
    "diagonal_front_left", "diagonal_front_right",
    "diagonal_back_left",  "diagonal_back_right",
    "forward_left",   "forward_right",
    "backward_left",  "backward_right",
    "turn_rear_axis_left",   "turn_rear_axis_right",
    "turn_front_axis_left",  "turn_front_axis_right",
    "swing_turn_right",      "swing_turn_left",
    "swing_turn_back_right", "swing_turn_back_left",
])


class KikiMotorClient:
    """
    Thin ZMQ REQ/REP client for the chassis motor server.

    All collision avoidance is performed server-side — callers just
    send steps and get back a status + live sensor readings.

    Every method is a classmethod so callers never need to instantiate.
    A new socket is opened per call: keeps things stateless and avoids
    broken-socket issues in long-running threads.
    """

    @classmethod
    def _send(cls, msg: dict, timeout_ms: int = 15_000) -> dict:
        """Open a fresh REQ socket, send msg, return reply."""
        ctx  = _zmq.Context()
        sock = ctx.socket(_zmq.REQ)
        sock.setsockopt(_zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(_zmq.SNDTIMEO, 2_000)
        try:
            sock.connect(f"tcp://{ZMQ_MOTOR_HOST}:{ZMQ_MOTOR_PORT}")
            sock.send_json(msg)
            return sock.recv_json()
        except _zmq.Again:
            return {"status": "error", "message": "Motor server timeout — is hailo_follower running?"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            sock.close()
            ctx.term()

    @classmethod
    def execute_step(cls, action: str, speed: int, duration: float) -> dict:
        """
        Execute ONE timed movement step with server-side collision avoidance.
        Blocks until the step finishes (or collides).

        Returns:
            {"status": "ok"|"collision_front"|"collision_rear"|"error",
             "front_dist": float, "rear_dist": float}
        """
        action = action.lower().strip().replace(" ", "_")
        if action not in VALID_MOTOR_ACTIONS:
            return {"status": "error",
                    "message": f"Unknown action '{action}'. Valid: {sorted(VALID_MOTOR_ACTIONS)}"}
        # Timeout = step duration + generous 5 s overhead
        timeout_ms = int(duration * 1000) + 5_000
        return cls._send(
            {"cmd": "execute_step",
             "action":   action,
             "speed":    max(30, min(100, int(speed))),
             "duration": max(0.1, min(15.0, float(duration)))},
            timeout_ms=timeout_ms
        )

    @classmethod
    def stop(cls) -> dict:
        """Immediately stop all chassis motors."""
        return cls._send({"cmd": "stop"}, timeout_ms=2_000)

    @classmethod
    def get_sensors(cls) -> dict:
        """
        Read cached ultrasonic distances (updated every ~50 ms server-side).
        Returns {"status": "ok", "front_dist": float, "rear_dist": float}
        """
        return cls._send({"cmd": "get_sensors"}, timeout_ms=2_000)


def _music_safety_monitor(proc):
    """
    Poll the motor server for the front sensor reading.
    Stops mpv if something gets within 10 cm.
    No GPIO opened here — completely eliminates "Device or resource busy".
    (Was nested inside play_music; only meaningful for a MOVING chassis.)
    """
    while proc.poll() is None:          # while music subprocess is alive
        resp = KikiMotorClient.get_sensors()
        if resp.get("status") == "ok":
            dist = resp.get("front_dist", -1)
            if 0 < dist < 10:
                print(f"[Music Safety] Obstacle at {dist:.1f} cm — stopping music")
                subprocess.Popen(["pkill", "-9", "mpv"])
                break
        time.sleep(0.5)
    try:
        from core.oled_display import oled_manager
        oled_manager.music_off()
    except Exception:
        pass


async def track_full_body__follow_me(duration: int) -> str:
    """Enable full body following mode. (formerly follow_me)"""
    from tools_and_config.tools import _get_controller
    try:
        controller = await _get_controller()
        success = await controller.set_full_body_movement(True)
        if success:
            return "I'm now following you! Say 'stop' to stop."
        return "Sorry, I couldn't enable follow mode."
    except Exception as e:
        return f"Sorry, error enabling follow mode: {str(e)}"


async def move(steps: list) -> str:
    """
    Perform a sequence of physical movements via the central motor server.
    All wheel states supported: forward, backward, strafe_left/right,
    diagonals, pivot turns, swing turns, etc.
    Collision avoidance is handled server-side.
    """
    from tools_and_config.tools import set_neck_active
    _stop = threading.Event()

    def _worker():
        set_neck_active(True)
        try:
            for raw in steps:
                if _stop.is_set():
                    break

                if isinstance(raw, str):
                    import ast
                    try:
                        step_data = ast.literal_eval(raw)
                    except Exception:
                        print(f"[Move] Could not parse step: {raw!r}")
                        continue
                else:
                    step_data = raw

                if not isinstance(step_data, dict):
                    print(f"[Move] Skipping non-dict step: {step_data!r}")
                    continue

                interval = float(step_data.get('interval', 0))
                action   = str(step_data.get('step', 'stop')).lower().strip().replace(' ', '_')
                duration = max(0.1, min(15.0, float(step_data.get('duration', 0.5))))
                speed    = max(30,  min(100,  int(step_data.get('speed', 50))))

                waited = 0.0
                while waited < interval and not _stop.is_set():
                    chunk = min(0.1, interval - waited)
                    time.sleep(chunk)
                    waited += chunk

                if _stop.is_set():
                    break

                if action not in VALID_MOTOR_ACTIONS:
                    print(f"[Move] Unknown action '{action}' — skipping. "
                          f"Valid: {sorted(VALID_MOTOR_ACTIONS)}")
                    continue

                resp   = KikiMotorClient.execute_step(action, speed, duration)
                status = resp.get("status", "error")

                if status == "ok":
                    pass
                elif status.startswith("collision"):
                    print(f"[Move] Collision ({status}) — "
                          f"front={resp.get('front_dist')} cm, "
                          f"rear={resp.get('rear_dist')} cm — continuing to next step")
                elif status == "error":
                    print(f"[Move] Server error on step '{action}': {resp.get('message')}")
                else:
                    print(f"[Move] Unexpected status '{status}' — continuing")

        except Exception as e:
            print(f"[Move] Worker exception: {e}")
            import traceback
            traceback.print_exc()
        finally:
            KikiMotorClient.stop()
            set_neck_active(True)
            print("[Move] Sequence complete")

    threading.Thread(target=_worker, daemon=True, name="move_worker").start()
    return (f"Movement sequence started — {len(steps)} step(s), "
            f"all wheel states supported, collision safety active.")


async def dance(song: str, steps: list) -> str:
    """
    Perform a choreographed dance with music (from YouTube) and chassis movements.
    All wheel states supported. Motor commands via motor server — no direct GPIO.
    Music subprocess and motor steps are interleaved using per-step intervals.
    """
    from tools_and_config.tools import set_neck_active
    import tools_and_config.tools as _tools
    _tools._skip_followup = True
    _stop = threading.Event()

    def _worker():
        set_neck_active(True)

        music_cfg = get_full_config().get("tools", {}).get("music", {})
        yt_dlp_path = music_cfg.get("yt_dlp_path", "/srv/kikifast/.venv/bin/yt-dlp")
        if not os.path.exists(yt_dlp_path):
            yt_dlp_path = "yt-dlp"
        try:
            res = subprocess.run(
                [yt_dlp_path, "-f", "ba/b", f"ytsearch:{song}", "-g"],
                capture_output=True, text=True, timeout=20)
        except Exception as e:
            print(f"[Dance] yt-dlp failed: {e}")
            set_neck_active(True)
            return
        url = (res.stdout or "").strip().splitlines()[-1] if (res.stdout or "").strip() else ""
        if res.returncode != 0 or not url.startswith("http"):
            err_lines = (res.stderr or "").strip().splitlines()
            print(f"[Dance] Could not resolve music URL for '{song}' — aborting dance "
                  f"({err_lines[-1] if err_lines else res.returncode})")
            set_neck_active(True)
            return
        try:
            music_proc = subprocess.Popen(
                ["mpv", "--no-video", url],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid
            )
        except Exception as e:
            print(f"[Dance] Failed to start music subprocess: {e}")
            set_neck_active(True)
            return

        music_started = False
        t_start = time.time()
        while time.time() - t_start < 30:
            if _stop.is_set():
                break
            line = music_proc.stdout.readline()
            if not line:
                break
            if "A:" in line:
                music_started = True
                break

        if not music_started:
            print(f"[Dance] Music failed to start for '{song}' — aborting dance")
            try:
                os.killpg(os.getpgid(music_proc.pid), signal.SIGTERM)
            except Exception:
                pass
            set_neck_active(True)
            return

        print(f"[Dance] Music started — beginning choreography ({len(steps)} steps)")
        time.sleep(20)

        try:
            for raw in steps:
                if _stop.is_set():
                    break

                if isinstance(raw, str):
                    import ast
                    try:
                        step_data = ast.literal_eval(raw)
                    except Exception:
                        print(f"[Dance] Could not parse step: {raw!r}")
                        continue
                else:
                    step_data = raw

                if not isinstance(step_data, dict):
                    print(f"[Dance] Skipping non-dict step: {step_data!r}")
                    continue

                interval = float(step_data.get('interval', 0))
                action   = str(step_data.get('step', 'stop')).lower().strip().replace(' ', '_')
                duration = max(0.1, min(15.0, float(step_data.get('duration', 0.5))))
                speed    = max(30,  min(100,  int(step_data.get('speed', 50))))

                waited = 0.0
                while waited < interval and not _stop.is_set():
                    chunk = min(0.1, interval - waited)
                    time.sleep(chunk)
                    waited += chunk

                if _stop.is_set():
                    break
                if action == "pause":
                    KikiMotorClient.stop()
                    waited = 0.0
                    while waited < duration and not _stop.is_set():
                        chunk = min(0.1, duration - waited)
                        time.sleep(chunk)
                        waited += chunk
                    print(f"[Dance] Pause held for {duration:.1f}s")
                    continue
                if action not in VALID_MOTOR_ACTIONS:
                    print(f"[Dance] Unknown action '{action}' — skipping. "
                          f"Valid: {sorted(VALID_MOTOR_ACTIONS)}")
                    continue
                phase = step_data.get("phase", "")
                if phase:
                    print(f"[Dance] [{phase.upper()}] {action} spd={speed} dur={duration:.2f}s")

                resp   = KikiMotorClient.execute_step(action, speed, duration)
                status = resp.get("status", "error")

                if status == "ok":
                    pass
                elif status.startswith("collision"):
                    print(f"[Dance] Collision ({status}) — "
                          f"front={resp.get('front_dist')} cm, "
                          f"rear={resp.get('rear_dist')} cm — skipping step")
                elif status == "error":
                    print(f"[Dance] Server error on step '{action}': {resp.get('message')}")

        except Exception as e:
            print(f"[Dance] Worker exception: {e}")
            import traceback
            traceback.print_exc()

        finally:
            print("[Dance] Choreography complete — stopping music")
            try:
                os.killpg(os.getpgid(music_proc.pid), signal.SIGTERM)
            except Exception:
                pass
            subprocess.run("pkill mpv", shell=True, capture_output=True)
            KikiMotorClient.stop()
            set_neck_active(True)
            print("[Dance] Complete")

    threading.Thread(target=_worker, daemon=True, name="dance_worker").start()
    return (f"Dance to '{song}' started — {len(steps)} choreographed step(s), "
            f"all wheel states supported, collision safety active.")
