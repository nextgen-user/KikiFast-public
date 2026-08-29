"""Audible hold timing for guided exercise.

A guided routine is not a conversation. When Kiki says "hold this for ten
seconds", the person needs the ten seconds *marked* — the way an exercise video
ticks them off — not a sentence about them.

Two earlier behaviours this replaces, both seen live on 2026-08-29:

* "Hold it there for five seconds." → TTS finished, the microphone opened, and
  Kiki waited for the person to say something. The hold was never timed; the
  session stalled until the person spoke.
* "hold that position for five seconds... four, three, two, one. Great job." →
  the model tried to count in its own summary, and TTS read the whole countdown
  aloud in under two seconds. The person got the words without the time.

The countdown is rendered as ONE wav and played with a single process. Spawning
a player per beep costs a fresh ALSA/Bluetooth sink open each time (measured at
300-500 ms in sound_effects.py), which would both smear the timing it exists to
convey and cost more than the beep itself.
"""

from __future__ import annotations

import math
import os
import time
import struct
import subprocess
import threading
import wave
from typing import Optional

SAMPLE_RATE = 24000          # matches the other sound_effects assets
_TICK_HZ = 1500.0            # per-second marker: short, bright, easy to count
_TICK_MS = 70
_FINAL_HZ = 950.0            # "done" tone: lower and longer, unmistakably an end
_FINAL_MS = 320
_AMPLITUDE = 0.35            # well under full scale; this plays between spoken
                             # instructions and must not startle anyone

# Rendered tracks are cached per duration — a routine repeats the same holds
# over and over, and re-synthesising 10 s of near-silence each time is waste.
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "_cadence_cache")
_CACHE_LOCK = threading.Lock()

# A hold longer than this is treated as a model mistake rather than an
# instruction. Standing still for minutes because a number had an extra digit
# is exactly the kind of thing an unattended person should not be asked to do.
MAX_HOLD_SECONDS = 120


def _tone(frequency: float, milliseconds: int) -> bytes:
    """A single 16-bit mono tone with short fades so it clicks at neither end."""
    total = int(SAMPLE_RATE * milliseconds / 1000.0)
    fade = max(1, int(SAMPLE_RATE * 0.005))       # 5 ms
    out = bytearray()
    for n in range(total):
        envelope = 1.0
        if n < fade:
            envelope = n / fade
        elif n > total - fade:
            envelope = max(0.0, (total - n) / fade)
        sample = _AMPLITUDE * envelope * math.sin(
            2.0 * math.pi * frequency * n / SAMPLE_RATE)
        out += struct.pack("<h", int(sample * 32767))
    return bytes(out)


def _silence(milliseconds: float) -> bytes:
    return b"\x00\x00" * int(SAMPLE_RATE * milliseconds / 1000.0)


def countdown_track(seconds: int) -> Optional[str]:
    """Path to a wav that ticks once per second for ``seconds``, then ends.

    The track is exactly ``seconds`` long plus the final tone, so playing it to
    completion *is* the hold. Returns None for a non-positive duration.
    """
    seconds = int(seconds)
    if seconds <= 0:
        return None
    seconds = min(seconds, MAX_HOLD_SECONDS)

    path = os.path.join(_CACHE_DIR, f"hold_{seconds}s.wav")
    with _CACHE_LOCK:
        if os.path.exists(path):
            return path
        os.makedirs(_CACHE_DIR, exist_ok=True)

        tick = _tone(_TICK_HZ, _TICK_MS)
        gap = _silence(1000 - _TICK_MS)
        frames = bytearray()
        for _ in range(seconds):
            frames += tick
            frames += gap
        frames += _tone(_FINAL_HZ, _FINAL_MS)

        tmp = path + ".tmp"
        with wave.open(tmp, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(bytes(frames))
        os.replace(tmp, path)          # never expose a half-written track
        return path


def play_countdown(seconds: int, stop_event: Optional[threading.Event] = None,
                   player: str = "mpv", capture_fn=None,
                   max_captures: int = 3) -> bool:
    """Play the hold countdown, blocking until it finishes.

    Blocking is deliberate: the caller keeps the microphone muted across this
    whole window, which is what stops Kiki from hearing her own beeps and
    treating them as a reply. Returns True if the hold ran to completion.

    ``capture_fn`` is called a few times DURING the hold, on a side thread, and
    whatever it returns is collected into :func:`last_hold_frames`. The hold is
    the only moment the position actually exists — a frame grabbed after the
    beeps stop shows someone already relaxing out of it, which is how a
    half-done stretch got praised as correct. Capturing must never delay the
    beeps, so failures are swallowed and the countdown is never waited on.
    """
    global _LAST_HOLD_FRAMES, _LAST_HOLD_AT
    path = countdown_track(seconds)
    if not path:
        return False
    try:
        proc = subprocess.Popen(
            [player, "--no-video", "--audio-device=alsa", "--really-quiet", path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"[Cadence] could not start the hold countdown: {exc}")
        return False

    print(f"[Cadence] holding {seconds}s (beeping each second)")
    frames: list = []
    capture_thread = None
    if capture_fn is not None and max_captures > 0:
        # Spread the shots across the middle of the hold: the first instant is
        # still the person getting into position, and the last is them coming
        # out of it.
        shots = min(max_captures, max(1, int(seconds)))
        offsets = [seconds * (i + 1) / (shots + 1) for i in range(shots)]

        def _capture_loop():
            start = time.monotonic()
            for offset in offsets:
                delay = offset - (time.monotonic() - start)
                if delay > 0:
                    if stop_event is not None and stop_event.wait(timeout=delay):
                        return
                    elif stop_event is None:
                        time.sleep(delay)
                if stop_event is not None and stop_event.is_set():
                    return
                try:
                    frame = capture_fn()
                    if frame:
                        frames.append(frame)
                except Exception as exc:
                    print(f"[Cadence] hold capture failed: {exc}")

        capture_thread = threading.Thread(
            target=_capture_loop, name="hold_capture", daemon=True)
        capture_thread.start()

    completed = True
    try:
        while proc.poll() is None:
            if stop_event is not None and stop_event.is_set():
                # An abort ("stop", open palm) must silence the beeps at once.
                proc.kill()
                proc.wait(timeout=2)
                print("[Cadence] hold aborted")
                completed = False
                break
            if stop_event is not None:
                stop_event.wait(timeout=0.1)
            else:
                try:
                    proc.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
    except Exception as exc:
        print(f"[Cadence] hold playback error: {exc}")
        completed = False

    if capture_thread is not None:
        # Bounded: a wedged camera must not hold the routine open.
        capture_thread.join(timeout=2.0)
    _LAST_HOLD_FRAMES = list(frames)
    _LAST_HOLD_AT = time.monotonic()
    if frames:
        print(f"[Cadence] captured {len(frames)} frame(s) during the hold")
    return completed


# Frames taken during the most recent hold, newest hold only. Read once by the
# turn that follows the hold, then cleared, so a later turn can never judge a
# posture from an old exercise.
_LAST_HOLD_FRAMES: list = []
_LAST_HOLD_AT: float = 0.0

# Frames older than this are not "the position they are in" any more. The turn
# that follows a hold runs within a couple of seconds; anything much later means
# the routine was interrupted, and judging a stretch from a minute-old
# photograph is the same mistake as judging it from a frozen camera.
HOLD_FRAME_TTL_S = 30.0


def take_hold_frames(ttl: float = HOLD_FRAME_TTL_S) -> list:
    """Return and clear the frames captured during the last hold.

    Empty if they have gone stale, so the caller falls back to a live capture
    rather than describing an old movement as the current one.
    """
    global _LAST_HOLD_FRAMES, _LAST_HOLD_AT
    frames, _LAST_HOLD_FRAMES = _LAST_HOLD_FRAMES, []
    age = time.monotonic() - _LAST_HOLD_AT
    _LAST_HOLD_AT = 0.0
    if frames and age > ttl:
        print(f"[Cadence] discarding {len(frames)} hold frame(s), "
              f"{age:.0f}s old")
        return []
    return frames


def clear_hold_frames() -> None:
    global _LAST_HOLD_FRAMES, _LAST_HOLD_AT
    _LAST_HOLD_FRAMES = []
    _LAST_HOLD_AT = 0.0


def clamp_hold_seconds(value, maximum: int = MAX_HOLD_SECONDS) -> int:
    """Coerce a model-supplied hold into a safe integer number of seconds."""
    try:
        held = int(float(value))
    except (TypeError, ValueError):
        return 0
    if held <= 0:
        return 0
    return min(held, maximum)
