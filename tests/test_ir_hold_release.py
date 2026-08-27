"""Push-to-talk hold/release behaviour of the IR sensor control surface.

The regression these guard: a flickering IR read used to reset the release
debounce window on every spurious sample, so lifting your hand never actually
ended the hold and Kiki kept listening.
"""

import sys
import types
import unittest
from unittest.mock import Mock

# core.ir_controls imports gpiod (absent off-robot) and core.lcd_display (which
# talks to I2C at import time). Stub both before importing the module.
sys.modules.setdefault("gpiod", types.ModuleType("gpiod"))
if "core.lcd_display" not in sys.modules:
    _lcd = types.ModuleType("core.lcd_display")
    _lcd.lcd_manager = Mock()
    sys.modules["core.lcd_display"] = _lcd

from core import ir_controls as ir  # noqa: E402


class _FakeIR(ir.IRControls):
    """IRControls with the GPIO layer replaced by a scripted sample sequence."""

    def __init__(self, **callbacks):
        # Bypass __init__'s gpiod line request; set up state by hand.
        self.__dict__.update(
            {
                "on_talk_hold_start": callbacks.get("on_talk_hold_start"),
                "on_talk_hold_end": callbacks.get("on_talk_hold_end"),
                "on_talk_hold_cancel": callbacks.get("on_talk_hold_cancel"),
                "on_enter_settings": None,
                "on_exit_settings": None,
                "on_double_tap": callbacks.get("on_double_tap"),
                "enabled": False,
                "_request": None,
                "mode": "normal",
                "hold_active": False,
                "_hold_clear_since": None,
                "_present_run": 0,
                "_hold_started_at": 0.0,
                "_last_hold_end": 0.0,
                "_normal_press_since": {ir.LEFT_PIN: None, ir.RIGHT_PIN: None},
                "_last_tap_pin": None,
                "_last_tap_at": 0.0,
                "_both_since": None,
                "_suppress_until_clear": False,
                "menu": ["BT Volume", "Restart", "Exit"],
                "_sel": 0,
                "_screen": "menu",
                "_vol": 50,
                "_last_settings_action": 0.0,
                "_settings_armed": False,
                "_press_since": {ir.LEFT_PIN: None, ir.RIGHT_PIN: None},
                "_long_fired": {ir.LEFT_PIN: False, ir.RIGHT_PIN: False},
            }
        )

    def feed(self, samples, start=100.0):
        """Drive _tick_normal over a list of booleans (left-sensor presence)."""
        now = start
        for present in samples:
            self._tick_normal(bool(present), False, now)
            now += ir.POLL_S
        return now


class HoldReleaseTests(unittest.TestCase):
    def setUp(self):
        self.started = Mock()
        self.ended = Mock()
        self.ir = _FakeIR(on_talk_hold_start=self.started, on_talk_hold_end=self.ended)

    def _samples_for(self, seconds):
        return max(1, int(round(seconds / ir.POLL_S)))

    def test_hold_starts_on_the_very_first_present_sample(self):
        self.ir.feed([True])
        self.assertTrue(self.ir.hold_active)
        self.assertEqual(self.started.call_count, 1)

    def test_clean_release_commits_after_the_debounce(self):
        self.ir.feed([True] * 10)
        self.ended.assert_not_called()
        self.ir.feed(
            [False] * (self._samples_for(ir.HOLD_RELEASE_DEBOUNCE_S) + 2),
            start=100.0 + 10 * ir.POLL_S,
        )
        self.assertFalse(self.ir.hold_active)
        self.assertEqual(self.ended.call_count, 1)

    def test_single_sample_glitch_does_not_restart_the_release_window(self):
        """The actual bug: one spurious 'present' read while the hand is gone."""
        self.ir.feed([True] * 10)
        # Hand leaves, but the sensor emits an isolated spike partway through
        # the release window. It must not re-arm the hold.
        noisy = [False, True, False, False, True, False, False, False, False, False]
        self.ir.feed(noisy, start=100.0 + 10 * ir.POLL_S)
        self.assertFalse(self.ir.hold_active, "isolated glitches must not hold the mic open")
        self.assertEqual(self.ended.call_count, 1)

    def test_sustained_re_touch_does_keep_the_hold_open(self):
        """A real hand coming back must still cancel the pending release."""
        self.ir.feed([True] * 10)
        # Momentary lift, then a solid re-touch (well over HOLD_REASSERT_SAMPLES).
        self.ir.feed([False, False] + [True] * 12, start=100.0 + 10 * ir.POLL_S)
        self.assertTrue(self.ir.hold_active)
        self.ended.assert_not_called()

    def test_stuck_sensor_commits_instead_of_listening_forever(self):
        n = self._samples_for(ir.HOLD_MAX_S) + 5
        self.ir.feed([True] * n)
        self.assertFalse(self.ir.hold_active)
        self.assertEqual(self.ended.call_count, 1, "stuck line must still commit the turn")
        # And it must not immediately re-arm while the line is still asserted.
        self.started.reset_mock()
        self.ir.feed([True] * 50, start=100.0 + n * ir.POLL_S)
        self.started.assert_not_called()

    def test_stuck_sensor_rearms_once_it_reads_clear(self):
        n = self._samples_for(ir.HOLD_MAX_S) + 5
        end = self.ir.feed([True] * n)
        self.started.reset_mock()
        end = self.ir.feed([False] * 30, start=end)          # all-clear + cooldown
        self.ir.feed([True], start=end)
        self.assertTrue(self.ir.hold_active)
        self.assertEqual(self.started.call_count, 1)

    def test_clear_hold_drops_the_hold_without_committing(self):
        self.ir.feed([True] * 10)
        self.ir.clear_hold()
        self.assertFalse(self.ir.hold_active)
        self.ended.assert_not_called()
        # A hand still resting on the sensor must not silently reopen the mic.
        self.started.reset_mock()
        self.ir.feed([True] * 20, start=100.0 + 10 * ir.POLL_S)
        self.started.assert_not_called()


if __name__ == "__main__":
    unittest.main()
