"""
Neck-gesture tags for Kiki's expressive head movement.

Kiki is a stationary unit; its only motion is NECK rotation (left / right / center),
driven by a GPIO stepper motor owned by the Hailo controller process
(`~/Kiki/navigation/hailo_follower_webcam_only.py`). Two layers of neck motion coexist:

  1. AUTOMATIC face-tracking — the controller turns the neck to keep the target person
     centered (enabled via `KikiController.set_neck_movement` / `set_target_person`).
  2. EXPLICIT expressive turns — the speaking model can emit an inline tag to deliberately
     turn its neck for expression, e.g.  "<neck:left> that is wild"  or  "<neck:right:30>".

This module handles layer 2. Tags are parsed out of the streamed reply, the surrounding
text is stripped before TTS *and* before the history append (KV-cache rule: the history
stores the stripped text), and the gestures are applied asynchronously via the controller.

Replaces the retired chassis-tag path. The tag
shape mirrors the old one so `main.py`'s plumbing stays the same — only neck-only now.
"""

import re
import asyncio
import threading

from tools_and_config.config_loader import get_full_config

# <neck:left>  <neck:right>  <neck:center>   (optional :<steps> intensity, e.g. <neck:left:30>)
_NECK_TAG_RE = re.compile(r'<neck:(left|right|center)(?::(\d+))?>', re.IGNORECASE)


def extract_neck_tags(text: str) -> list:
    """Extract neck gestures from LLM response text → [{'direction', 'steps'}]."""
    gestures = []
    for m in _NECK_TAG_RE.finditer(text or ""):
        direction = m.group(1).lower()
        steps = int(m.group(2)) if m.group(2) else None
        gestures.append({"direction": direction, "steps": steps})
    return gestures


def strip_neck_tags(text: str) -> str:
    """Remove neck tags before sending to TTS / storing in history."""
    return _NECK_TAG_RE.sub('', text or '').strip()



def apply_neck(gestures: list):
    """
    Apply queued neck gestures via the Hailo controller (NON-BLOCKING).

    Runs in a daemon thread so it never stalls the speaking path (same contract as the
    old `execute_movements`). Sends a one-shot `look` command per gesture. If the running
    controller build does not yet implement explicit `look` turns, the command is a
    harmless no-op (automatic face-tracking still drives the neck).
    """
    if not gestures:
        return

    host = get_full_config().get("controller", {}).get("host", "127.0.0.1")

    def _worker():
        try:
            from core.hardware.controller import quick_command
            for g in gestures:
                direction = g.get("direction", "center")
                steps = g.get("steps")
                try:
                    asyncio.run(quick_command(host=host, look=direction, look_steps=steps))
                    print(f"[Neck] look {direction}" + (f" ({steps} steps)" if steps else ""))
                    # print("BYPASSED")
                except Exception as e:
                    print(f"[Neck] Warning: could not apply gesture '{direction}': {e}")
        except Exception as e:
            print(f"[Neck] apply_neck error: {e}")

    threading.Thread(target=_worker, daemon=True, name="neck_gesture").start()
