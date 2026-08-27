"""Near-field gate: reject crowd babble without changing quiet-room behaviour."""

import unittest

import numpy as np

from core.near_field_gate import NearFieldGate, SILENCE_DBFS, frame_dbfs

FRAME = 512
FRAME_MS = 32.0

CFG = {
    "enabled": True,
    "engage_floor_dbfs": -50.0,
    "open_margin_db": 9.0,
    "close_margin_db": 4.0,
    "floor_rise_per_s_db": 3.0,
    "floor_fall_per_s_db": 30.0,
}


def noise_at(db, rng, n=FRAME):
    """White noise whose RMS is approximately `db` dBFS."""
    return (rng.standard_normal(n).astype(np.float32) * (10 ** (db / 20.0)))


def seconds(value):
    return int(round(value * 1000.0 / FRAME_MS))


class FrameLevelTests(unittest.TestCase):
    def test_digital_silence_reports_the_silence_floor(self):
        self.assertEqual(frame_dbfs(np.zeros(FRAME, dtype=np.float32)), SILENCE_DBFS)

    def test_level_tracks_rms_within_a_decibel(self):
        rng = np.random.default_rng(0)
        for target in (-60.0, -40.0, -20.0):
            self.assertAlmostEqual(frame_dbfs(noise_at(target, rng)), target, delta=1.0)


class QuietRoomTests(unittest.TestCase):
    """At home the gate must be completely inert — no behaviour change."""

    def test_gate_never_engages_in_a_quiet_room(self):
        rng = np.random.default_rng(1)
        gate = NearFieldGate(CFG, frame_ms=FRAME_MS)
        for _ in range(seconds(10.0)):
            self.assertTrue(gate.update(noise_at(-62.0, rng), False))
        self.assertFalse(gate.engaged)

    def test_quiet_room_speech_always_passes(self):
        rng = np.random.default_rng(2)
        gate = NearFieldGate(CFG, frame_ms=FRAME_MS)
        for _ in range(seconds(5.0)):
            gate.update(noise_at(-62.0, rng), False)
        for _ in range(seconds(2.0)):
            self.assertTrue(gate.update(noise_at(-30.0, rng), True))

    def test_disabled_gate_passes_everything(self):
        rng = np.random.default_rng(3)
        gate = NearFieldGate({**CFG, "enabled": False}, frame_ms=FRAME_MS)
        for _ in range(seconds(6.0)):
            self.assertTrue(gate.update(noise_at(-30.0, rng), True))
        self.assertFalse(gate.engaged)


class CrowdedRoomTests(unittest.TestCase):
    def _crowded_gate(self, rng, floor_db=-34.0, warmup_s=6.0):
        gate = NearFieldGate(CFG, frame_ms=FRAME_MS)
        # In a crowd Silero calls almost every frame speech — that is exactly
        # why the endpointer needs this gate — so warm up with is_speech=True.
        for _ in range(seconds(warmup_s)):
            gate.update(noise_at(floor_db, rng), True)
        return gate

    def test_gate_engages_once_the_room_is_noisy(self):
        gate = self._crowded_gate(np.random.default_rng(4))
        self.assertTrue(gate.engaged)

    def test_crowd_babble_is_rejected(self):
        rng = np.random.default_rng(5)
        gate = self._crowded_gate(rng)
        passed = sum(gate.update(noise_at(-34.0, rng), True) for _ in range(seconds(4.0)))
        self.assertEqual(passed, 0, "babble at the noise floor must not count as speech")

    def test_near_field_speech_still_passes(self):
        rng = np.random.default_rng(6)
        gate = self._crowded_gate(rng)
        passed = sum(gate.update(noise_at(-18.0, rng), True) for _ in range(seconds(1.0)))
        self.assertEqual(passed, seconds(1.0))

    def test_hysteresis_holds_a_sentence_together(self):
        """Quiet trailing syllables must not be clipped off the utterance."""
        rng = np.random.default_rng(7)
        gate = self._crowded_gate(rng)
        for _ in range(seconds(0.5)):                      # open the gate loudly
            self.assertTrue(gate.update(noise_at(-18.0, rng), True))
        # Now sit between the close and open thresholds: still counts as speech.
        between = gate.floor_dbfs + (CFG["open_margin_db"] + CFG["close_margin_db"]) / 2
        self.assertTrue(gate.update(noise_at(between, rng), True))

    def test_the_users_own_speech_does_not_drag_the_floor_up(self):
        rng = np.random.default_rng(8)
        gate = self._crowded_gate(rng)
        floor_before = gate.floor_dbfs
        for _ in range(seconds(3.0)):
            gate.update(noise_at(-15.0, rng), True)
        self.assertLessEqual(gate.floor_dbfs, floor_before + 0.5)

    def test_floor_falls_when_the_room_empties(self):
        rng = np.random.default_rng(9)
        gate = self._crowded_gate(rng)
        self.assertTrue(gate.engaged)
        for _ in range(seconds(6.0)):
            gate.update(noise_at(-65.0, rng), False)
        self.assertFalse(gate.engaged, "gate must go inert again once the room is quiet")

    def test_reset_keeps_the_learned_floor(self):
        rng = np.random.default_rng(10)
        gate = self._crowded_gate(rng)
        floor = gate.floor_dbfs
        gate.reset(keep_floor=True)
        self.assertAlmostEqual(gate.floor_dbfs, floor)
        self.assertTrue(gate.engaged, "a mute/unmute must not reopen the gate to the crowd")


if __name__ == "__main__":
    unittest.main()
