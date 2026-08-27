"""Near-field speech gate — the crowded-room fix for Kiki's endpointer.

The problem this solves
-----------------------
Silero VAD answers "is this speech?", not "is this speech *addressed to Kiki*".
In a crowded room the background babble IS speech, so Silero reports voiced
frames continuously.  The endpointer then never accumulates trailing silence,
``silence_ms`` stays pinned at 0, no endpoint ever fires, and the 1 s interim
heartbeat keeps resetting main.py's listen-window timer.  Kiki listens forever.

RNNoise does not help here: it is trained to separate speech from *non-speech*
noise (fans, traffic, motors), so it deliberately preserves background voices.

What actually distinguishes the user from the crowd is level.  Someone talking
to Kiki is near-field and lands well above the room's babble, which sits near
the noise floor.  This module tracks that floor and reports how far above it the
current frame is, so the endpointer can require real near-field level before
counting a frame as "the user is talking to me".

Design notes
------------
* The floor is an asymmetric envelope follower: it falls fast (so it tracks a
  room going quiet) and rises slowly (so the user's own speech cannot drag the
  floor up and gate them out mid-sentence).  Rates are expressed in dB per
  second so they are independent of frame size.
* The gate stays completely inert until the floor itself climbs above
  ``engage_floor_dbfs``.  In a quiet home the floor sits far below that, the
  gate never engages, and endpointing behaves exactly as it does today.
* Open/close thresholds are hysteretic.  Speech has to stand ``open_margin_db``
  above the floor to start counting, but only ``close_margin_db`` to keep
  counting, so quiet trailing syllables are not clipped off a sentence.
"""

from __future__ import annotations

import math

import numpy as np


# Level assigned to digital silence.  Low enough to be below any real floor.
SILENCE_DBFS = -90.0


def frame_dbfs(frame: np.ndarray) -> float:
    """RMS level of a float32 [-1, 1] frame, in dBFS."""
    if frame.size == 0:
        return SILENCE_DBFS
    rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
    if rms <= 1e-9:
        return SILENCE_DBFS
    return max(SILENCE_DBFS, 20.0 * math.log10(rms))


class NearFieldGate:
    """Adaptive noise-floor tracker + hysteretic near-field level gate.

    ``update()`` is called once per audio frame, before the VAD decision is
    used.  It returns whether this frame passes the near-field test.  When the
    room is quiet (or the gate is disabled) it always returns True, so the
    caller's behaviour is unchanged.
    """

    def __init__(self, cfg: dict | None = None, frame_ms: float = 32.0):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.engage_floor_dbfs = float(cfg.get("engage_floor_dbfs", -50.0))
        self.open_margin_db = float(cfg.get("open_margin_db", 9.0))
        # Never let the close threshold sit above the open threshold, or the
        # gate would chatter instead of holding a sentence together.
        self.close_margin_db = min(
            float(cfg.get("close_margin_db", 4.0)), self.open_margin_db
        )

        frame_s = max(1e-3, frame_ms / 1000.0)
        self._rise_step = float(cfg.get("floor_rise_per_s_db", 3.0)) * frame_s
        self._fall_step = float(cfg.get("floor_fall_per_s_db", 30.0)) * frame_s

        self.floor_dbfs = SILENCE_DBFS
        self.last_level_dbfs = SILENCE_DBFS
        self._open = False
        self._primed = False

    # -------------------------------------------------------------- #
    @property
    def engaged(self) -> bool:
        """True when the room is noisy enough for the gate to actually gate."""
        return self.enabled and self.floor_dbfs >= self.engage_floor_dbfs

    def reset(self, keep_floor: bool = True):
        """Clear the open/closed state at an utterance boundary.

        The floor is a property of the *room*, not of the utterance, so it is
        retained by default — re-learning it from scratch on every mute/unmute
        would leave the gate wide open for the first seconds of every turn,
        which is exactly when the crowd noise needs rejecting.
        """
        self._open = False
        if not keep_floor:
            self.floor_dbfs = SILENCE_DBFS
            self._primed = False

    def update(self, frame: np.ndarray, is_speech: bool) -> bool:
        """Track the floor for one frame and report whether it is near-field.

        ``is_speech`` is Silero's verdict for this frame.  It is used only to
        hold the floor still during speech; the floor must never learn from the
        user's own voice.
        """
        level = frame_dbfs(frame)
        self.last_level_dbfs = level

        if not self._primed:
            # Seed on the first frame so the follower does not have to climb up
            # from -90 dBFS one slow step at a time.
            self.floor_dbfs = level
            self._primed = True
        elif level < self.floor_dbfs:
            # Room got quieter than our estimate — follow it down quickly.
            self.floor_dbfs = max(level, self.floor_dbfs - self._fall_step)
        elif not is_speech:
            # Only non-speech frames may push the floor up, and only slowly.
            self.floor_dbfs = min(level, self.floor_dbfs + self._rise_step)

        if not self.engaged:
            self._open = False
            return True

        threshold = self.floor_dbfs + (
            self.close_margin_db if self._open else self.open_margin_db
        )
        self._open = level >= threshold
        return self._open

    def describe(self) -> str:
        """Short human-readable state, for the endpointer's debug logging."""
        return (
            f"floor={self.floor_dbfs:.1f}dBFS level={self.last_level_dbfs:.1f}dBFS "
            f"{'ENGAGED' if self.engaged else 'inert'}"
        )
