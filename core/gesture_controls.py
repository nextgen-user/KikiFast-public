"""Small, thread-safe state shared by camera gesture handlers and output tools.

The camera process sends edge-triggered events, but Kiki's ZMQ listener, TTS
threads, and tool workers all consume them from different threads.  Keep the
cross-thread state here so gesture handling never needs to poll the speaking
loop or add work to the model's time-to-first-word path.
"""

import threading


_output_muted = threading.Event()
_mute_lock = threading.Lock()
_activity_lock = threading.Lock()
_activity_stop_generation = 0


def is_output_muted() -> bool:
    """Whether new spoken output must be rendered on the LCD instead."""
    return _output_muted.is_set()


def toggle_output_mute() -> bool:
    """Toggle output mute and return the new state (True means muted)."""
    with _mute_lock:
        if _output_muted.is_set():
            _output_muted.clear()
            return False
        _output_muted.set()
        return True


def set_output_muted(muted: bool) -> None:
    """Explicit setter used by tests and future non-gesture controls."""
    with _mute_lock:
        if muted:
            _output_muted.set()
        else:
            _output_muted.clear()


def activity_generation() -> int:
    """Snapshot the current stop generation before starting an activity."""
    with _activity_lock:
        return _activity_stop_generation


def mark_activities_stopped() -> int:
    """Invalidate in-flight activities and return the new generation."""
    global _activity_stop_generation
    with _activity_lock:
        _activity_stop_generation += 1
        return _activity_stop_generation


def activity_was_stopped(generation: int) -> bool:
    """True when an open-palm/idle stop happened after ``generation``."""
    with _activity_lock:
        return generation != _activity_stop_generation
