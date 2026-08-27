"""
IR-sensor + LCD control surface for Kiki.

Two infrared proximity sensors (active-LOW: a hand hovering over the sensor pulls
the line LOW, i.e. ``Value.INACTIVE``):

  * LEFT  sensor on GPIO 22  (Kiki's left)
  * RIGHT sensor on GPIO 17  (Kiki's right)

Gestures
--------
NORMAL mode (push-to-talk hold):
  * Double-tap EITHER sensor -> fires ``on_double_tap`` so main.py stops the
    current interaction and returns Kiki to idle mode.
  * Hand over EITHER sensor  ->  fires ``on_talk_hold_start`` IMMEDIATELY on
    detection (no waiting for release): main.py stops music, aborts any current
    speech/generation, unmutes the mic and holds the STT endpoint open — Kiki
    keeps listening for as long as the hand stays put.
  * Hand removed (all sensors clear for ``HOLD_RELEASE_DEBOUNCE_S``)  ->  fires
    ``on_talk_hold_end``: main.py force-commits the utterance so Kiki replies
    right away instead of waiting out the silence window.  Presence is debounced
    asymmetrically — instant to assert, but ``HOLD_REASSERT_SAMPLES`` consecutive
    reads to *re-*assert — so a single flickering sensor read cannot keep
    restarting the release window and leave Kiki listening after the hand left.
    ``HOLD_MAX_S`` is the backstop for a sensor stuck asserted.
  * Long-hover BOTH sensors together (>= ``BOTH_HOLD_S``)  ->  open the SETTINGS
    menu (an active talk-hold is cancelled via ``on_talk_hold_cancel``).

SETTINGS mode (modal; main.py mutes the mic + pauses the hotword recognizer via the
``on_enter_settings`` / ``on_exit_settings`` callbacks):
  * Double-tap EITHER sensor -> exit settings and return Kiki to idle.
  * Tap the RIGHT sensor  ->  move the selection LEFT   (Kiki's right == user's left).
  * Tap the LEFT  sensor  ->  move the selection RIGHT.
  * Long-hover either sensor (>= ``LONG_HOVER_S``)  ->  SELECT the highlighted item.
  Menu items: "BT Volume", "Restart", "Exit".  Auto-exits after ``IDLE_EXIT_S`` of
  no gesture.

  BT-Volume sub-screen:
    * Tap LEFT  -> +``VOL_STEP``,  Tap RIGHT -> -``VOL_STEP``  (applied live via pactl).
    * Long-hover either sensor    -> save & return to the menu.

The whole thing degrades gracefully: if ``gpiod`` is missing or the lines cannot be
requested (e.g. running off-robot) the manager simply stays disabled.
"""

import os
import re
import sys
import time
import subprocess
import threading

try:
    import gpiod
    from gpiod.line import Direction, Value
    GPIOD_AVAILABLE = True
except Exception:  # pragma: no cover - off-robot
    GPIOD_AVAILABLE = False

from core.lcd_display import lcd_manager

# --- Hardware config ---
CHIP_PATH = "/dev/gpiochip4"
LEFT_PIN = 22    # Kiki's left
RIGHT_PIN = 17   # Kiki's right

# --- Timing / behaviour tunables (seconds) ---
POLL_S = 0.03            # sensor poll interval (~33 Hz)
LONG_HOVER_S = 1.0       # hold this long to count as a "select" / long-hover
BOTH_HOLD_S = 1.2        # hold BOTH sensors this long to open settings
HOLD_RELEASE_DEBOUNCE_S = 0.18  # sensors must stay clear this long -> hand really left
HOLD_COOLDOWN_S = 0.4    # min gap between a hold ending and the next one starting
# These IR proximity sensors chatter at the edge of their detection cone, and
# ambient IR (sunlight, PWM lighting, a TV remote) makes it worse. A hold STARTS
# on the first present sample so hold-to-talk stays instant, but cancelling a
# pending release needs this many consecutive present samples — otherwise one
# spurious read restarts the release window and the hold never ends, which is
# what made Kiki keep listening after the hand was already gone.
HOLD_REASSERT_SAMPLES = 2       # ~60ms of solid presence re-arms the hold
# Backstop for a sensor that is stuck asserted (failed, or staring at a hot IR
# source): commit the turn instead of listening forever. A new hold cannot begin
# until the sensors genuinely read clear again, so a stuck line cannot loop.
HOLD_MAX_S = 25.0
IDLE_EXIT_S = 20.0       # auto-leave settings after this much inactivity
MIN_TAP_S = 0.05         # debounce: ignore blips shorter than this
ACTION_COOLDOWN_S = 0.35 # min gap between settings actions
DOUBLE_TAP_GAP_S = 0.8   # max release-to-release gap for two taps on one sensor

VOL_STEP = 10            # % per tap in the volume sub-screen
VOL_MIN, VOL_MAX = 0, 100


def _present(value) -> bool:
    """True when a hand is over the sensor (active-LOW => INACTIVE)."""
    return value == Value.INACTIVE


# --------------------------------------------------------------------------- #
# Bluetooth speaker volume (PulseAudio bluez sink == default sink)
# --------------------------------------------------------------------------- #
def get_bt_volume() -> int:
    try:
        out = subprocess.run(
            ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
            capture_output=True, text=True, timeout=3,
        ).stdout
        m = re.search(r"(\d+)%", out)
        if m:
            return int(m.group(1))
    except Exception as e:
        print(f"[IR] get volume failed: {e}")
    return 50


def set_bt_volume(pct: int) -> int:
    pct = max(VOL_MIN, min(VOL_MAX, int(pct)))
    try:
        subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct}%"],
            capture_output=True, text=True, timeout=3,
        )
    except Exception as e:
        print(f"[IR] set volume failed: {e}")
    return pct


class IRControls:
    """Polls the two IR sensors and drives the talk-interrupt + settings UI."""

    def __init__(self, on_talk_hold_start=None, on_talk_hold_end=None,
                 on_talk_hold_cancel=None,
                 on_enter_settings=None, on_exit_settings=None,
                 on_double_tap=None):
        self.on_talk_hold_start = on_talk_hold_start
        self.on_talk_hold_end = on_talk_hold_end
        self.on_talk_hold_cancel = on_talk_hold_cancel
        self.on_enter_settings = on_enter_settings
        self.on_exit_settings = on_exit_settings
        self.on_double_tap = on_double_tap

        self.enabled = False
        self._request = None
        self._stop = threading.Event()
        self._thread = None

        # mode: "normal" or "settings"
        self.mode = "normal"

        # --- normal-mode push-to-talk hold tracking ---
        self.hold_active = False         # a hand is on a sensor (read by main.py)
        self._hold_clear_since = None    # ts sensors first read all-clear mid-hold
        self._present_run = 0            # consecutive present samples (glitch filter)
        self._hold_started_at = 0.0      # ts the active hold began (max-hold valve)
        self._last_hold_end = 0.0
        self._normal_press_since = {LEFT_PIN: None, RIGHT_PIN: None}
        self._last_tap_pin = None
        self._last_tap_at = 0.0

        # --- both-hold (settings entry) tracking ---
        self._both_since = None          # ts both sensors first present together
        self._suppress_until_clear = False  # ignore gestures until both released

        # --- settings UI state ---
        self.menu = ["BT Volume", "Restart", "Exit"]
        self._sel = 0
        self._screen = "menu"            # "menu" or "volume"
        self._vol = 50
        self._last_settings_action = 0.0
        # Settings ignores input until both sensors have cleared once, so the
        # hands still resting on the sensors from the entry gesture don't get
        # read as navigation taps.
        self._settings_armed = False

        # --- per-sensor press tracking (for tap vs long-hover) ---
        self._press_since = {LEFT_PIN: None, RIGHT_PIN: None}
        self._long_fired = {LEFT_PIN: False, RIGHT_PIN: False}

        if not GPIOD_AVAILABLE:
            print("[IR] gpiod not available - IR controls disabled.")
            return
        try:
            self._request = gpiod.request_lines(
                CHIP_PATH,
                consumer="kiki-ir-controls",
                config={
                    LEFT_PIN: gpiod.LineSettings(direction=Direction.INPUT),
                    RIGHT_PIN: gpiod.LineSettings(direction=Direction.INPUT),
                },
            )
            self.enabled = True
            print(f"[IR] Initialized IR controls (L=GPIO{LEFT_PIN}, R=GPIO{RIGHT_PIN}).")
        except Exception as e:
            print(f"[IR] Failed to request IR lines ({e}) - IR controls disabled.")

    # ----------------------------------------------------------------- #
    def start(self):
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._request is not None:
            try:
                self._request.release()
            except Exception:
                pass

    def clear_hold(self):
        """Drop an active talk-hold without firing the release callback.

        Used when something else already ended the turn (e.g. the thumbs-up
        gesture force-committed the utterance).  Starting a new hold still has
        to wait out ``HOLD_COOLDOWN_S``, so a hand left resting on the sensor
        cannot immediately re-open the mic on the next poll.
        """
        if not self.hold_active:
            return
        self.hold_active = False
        self._hold_clear_since = None
        self._present_run = 0
        self._last_hold_end = time.time()
        # The hand may still be resting on the sensor. Require a real all-clear
        # before a new hold can re-open the mic, so the turn we just committed
        # is not immediately reopened on the next poll.
        self._suppress_until_clear = True

    # ----------------------------------------------------------------- #
    def _read(self):
        try:
            left = _present(self._request.get_value(LEFT_PIN))
            right = _present(self._request.get_value(RIGHT_PIN))
            return left, right
        except Exception as e:
            print(f"[IR] read error: {e}")
            return False, False

    def _loop(self):
        print("[IR] Sensor loop running.")
        while not self._stop.is_set():
            left, right = self._read()
            now = time.time()
            try:
                if self.mode == "normal":
                    self._tick_normal(left, right, now)
                else:
                    self._tick_settings(left, right, now)
            except Exception as e:
                print(f"[IR] tick error: {e}")
            time.sleep(POLL_S)

    # ----------------------------------------------------------------- #
    # NORMAL mode
    # ----------------------------------------------------------------- #
    def _tick_normal(self, left, right, now):
        any_on = left or right

        # --- both-sensor hold opens settings (cancels an active talk-hold) ---
        if left and right:
            if self._both_since is None:
                self._both_since = now
            elif now - self._both_since >= BOTH_HOLD_S and not self._suppress_until_clear:
                if self.hold_active:
                    self.hold_active = False
                    self._fire(self.on_talk_hold_cancel, "talk-hold-cancel")
                self._enter_settings()
                return
        else:
            self._both_since = None

        # Wait for both sensors to clear before re-enabling gestures after a
        # settings exit (prevents the lingering hold from starting a talk-hold).
        if self._suppress_until_clear:
            if not any_on:
                self._suppress_until_clear = False
                self._last_hold_end = now   # cooldown counts from the all-clear
            return

        # Tap recognition is independent of push-to-talk's release debounce.
        # That preserves instant hold-to-talk while still allowing the second
        # short press on either individual sensor to become a terminal gesture.
        if self._track_normal_taps(left, right, now):
            return

        # --- push-to-talk: EITHER sensor held => listen; released => commit ---
        # Presence is debounced asymmetrically: instant to assert (hold-to-talk
        # must feel immediate) but a sustained run to *re-*assert, so a single
        # flickering read cannot keep restarting the release window forever.
        self._present_run = self._present_run + 1 if any_on else 0
        presence_confirmed = self._present_run >= HOLD_REASSERT_SAMPLES

        if not self.hold_active:
            self._hold_clear_since = None
            if any_on and (now - self._last_hold_end) >= HOLD_COOLDOWN_S:
                self.hold_active = True
                self._hold_started_at = now
                print("[IR] Hand detected -> talk hold (listening).")
                self._fire(self.on_talk_hold_start, "talk-hold-start")
            return

        # A hold that never ends means the sensor is stuck, not that the user
        # has been talking for half a minute. Commit and refuse to re-arm until
        # the line actually reads clear.
        if now - self._hold_started_at >= HOLD_MAX_S:
            print(f"[IR] Hold exceeded {HOLD_MAX_S:.0f}s (sensor stuck?) "
                  "-> committing turn and waiting for an all-clear.")
            self.hold_active = False
            self._hold_clear_since = None
            self._present_run = 0
            self._last_hold_end = now
            self._suppress_until_clear = True
            self._fire(self.on_talk_hold_end, "talk-hold-end")
            return

        if presence_confirmed:
            self._hold_clear_since = None
        elif self._hold_clear_since is None:
            # debounce the release so a marginal hover doesn't rapid-fire commits
            self._hold_clear_since = now
        elif now - self._hold_clear_since >= HOLD_RELEASE_DEBOUNCE_S:
            self.hold_active = False
            self._hold_clear_since = None
            self._present_run = 0
            self._last_hold_end = now
            print("[IR] Hand removed -> commit turn.")
            self._fire(self.on_talk_hold_end, "talk-hold-end")

    def _track_normal_taps(self, left, right, now):
        for pin, is_on in ((LEFT_PIN, left), (RIGHT_PIN, right)):
            started = self._normal_press_since[pin]
            if is_on:
                if started is None:
                    self._normal_press_since[pin] = now
                continue
            if started is None:
                continue
            self._normal_press_since[pin] = None
            held = now - started
            if MIN_TAP_S <= held < LONG_HOVER_S and self._is_double_tap(pin, now):
                print(f"[IR] Double-tap on GPIO{pin} -> return to idle.")
                if self.hold_active:
                    self.hold_active = False
                    self._hold_clear_since = None
                    self._last_hold_end = now
                    self._fire(self.on_talk_hold_cancel, "talk-hold-cancel")
                self._fire(self.on_double_tap, "double-tap")
                return True
        return False

    def _is_double_tap(self, pin, now):
        is_double = (
            self._last_tap_pin == pin
            and now - self._last_tap_at <= DOUBLE_TAP_GAP_S
        )
        if is_double:
            self._last_tap_pin = None
            self._last_tap_at = 0.0
        else:
            self._last_tap_pin = pin
            self._last_tap_at = now
        return is_double

    def _fire(self, cb, label):
        if cb is None:
            return
        try:
            cb()
        except Exception as e:
            print(f"[IR] {label} callback error: {e}")

    # ----------------------------------------------------------------- #
    # SETTINGS mode
    # ----------------------------------------------------------------- #
    def _enter_settings(self):
        print("[IR] Both-hold -> entering settings.")
        self.mode = "settings"
        self._screen = "menu"
        self._sel = 0
        self._both_since = None
        self._suppress_until_clear = True  # consumed when both release (below)
        self._present_run = 0
        self._settings_armed = False
        self._normal_press_since = {LEFT_PIN: None, RIGHT_PIN: None}
        self._press_since = {LEFT_PIN: None, RIGHT_PIN: None}
        self._long_fired = {LEFT_PIN: False, RIGHT_PIN: False}
        self._last_tap_pin = None
        self._last_tap_at = 0.0
        self._last_settings_action = time.time()
        if self.on_enter_settings:
            try:
                self.on_enter_settings()
            except Exception as e:
                print(f"[IR] enter-settings callback error: {e}")
        self._render_menu()

    def _exit_settings(self):
        print("[IR] Exiting settings.")
        self.mode = "normal"
        self._screen = "menu"
        # require both sensors to clear before normal gestures resume
        self._suppress_until_clear = True
        self.hold_active = False
        self._hold_clear_since = None
        self._present_run = 0
        self._both_since = None
        self._normal_press_since = {LEFT_PIN: None, RIGHT_PIN: None}
        if self.on_exit_settings:
            try:
                self.on_exit_settings()
            except Exception as e:
                print(f"[IR] exit-settings callback error: {e}")

    def _tick_settings(self, left, right, now):
        # Ignore the hands still resting from the entry/select gesture until both
        # sensors have been clear at least once.
        if not self._settings_armed:
            if not left and not right:
                self._settings_armed = True
                self._press_since = {LEFT_PIN: None, RIGHT_PIN: None}
                self._long_fired = {LEFT_PIN: False, RIGHT_PIN: False}
            return

        # auto-exit on inactivity
        if now - self._last_settings_action >= IDLE_EXIT_S:
            self._exit_settings()
            return

        for pin, is_on in ((LEFT_PIN, left), (RIGHT_PIN, right)):
            if is_on:
                if self._press_since[pin] is None:
                    self._press_since[pin] = now
                    self._long_fired[pin] = False
                elif (not self._long_fired[pin]
                        and now - self._press_since[pin] >= LONG_HOVER_S):
                    # long-hover crossed the threshold -> SELECT
                    self._long_fired[pin] = True
                    self._on_long_hover(pin)
            else:
                if self._press_since[pin] is not None:
                    held = now - self._press_since[pin]
                    self._press_since[pin] = None
                    if not self._long_fired[pin] and MIN_TAP_S <= held < LONG_HOVER_S:
                        self._on_tap(pin)
                    self._long_fired[pin] = False

    def _cooldown_ok(self, now=None):
        now = now or time.time()
        if now - self._last_settings_action < ACTION_COOLDOWN_S:
            return False
        self._last_settings_action = now
        return True

    def _on_tap(self, pin):
        now = time.time()
        if self._is_double_tap(pin, now):
            print(f"[IR] Double-tap on GPIO{pin} -> exiting settings to idle.")
            self._exit_settings()
            self._fire(self.on_double_tap, "double-tap")
            return
        if not self._cooldown_ok(now):
            return
        if self._screen == "menu":
            # Kiki's RIGHT (GPIO17) moves selection LEFT; LEFT moves RIGHT.
            if pin == RIGHT_PIN:
                self._sel = (self._sel - 1) % len(self.menu)
            else:
                self._sel = (self._sel + 1) % len(self.menu)
            self._render_menu()
        elif self._screen == "volume":
            if pin == LEFT_PIN:
                self._vol = set_bt_volume(self._vol + VOL_STEP)
            else:
                self._vol = set_bt_volume(self._vol - VOL_STEP)
            self._render_volume()

    def _on_long_hover(self, pin):
        # long-hover always registers activity (keep-alive)
        self._last_settings_action = time.time()
        if self._screen == "menu":
            self._select_menu_item()
        elif self._screen == "volume":
            # confirm & return to menu
            print(f"[IR] BT volume set to {self._vol}%.")
            self._screen = "menu"
            self._render_menu()

    def _select_menu_item(self):
        item = self.menu[self._sel]
        print(f"[IR] Settings select: {item}")
        if item == "BT Volume":
            self._screen = "volume"
            self._vol = get_bt_volume()
            self._render_volume()
        elif item == "Restart":
            self._do_restart()
        elif item == "Exit":
            self._exit_settings()

    # ----------------------------------------------------------------- #
    # Actions / rendering
    # ----------------------------------------------------------------- #
    def _do_restart(self):
        print("[IR] Restarting script...")
        try:
            lcd_manager.write("Restarting all", "services...")
        except Exception:
            pass
        try:
            subprocess.Popen(["pkill", "-9", "mpv"])
        except Exception:
            pass
        time.sleep(0.5)
        try:
            self.stop()
        except Exception:
            pass
        # Full-stack restart: kiki_boot.py --force restarts the laptop servers
        # (llama/tts/whisper) and the hailo webcam service, waits until all four
        # are live, then execs main.py. execv keeps our PID, so if we run under
        # kikifast.service systemd keeps tracking the same unit.
        boot = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "kiki_boot.py")
        os.execv(sys.executable, [sys.executable, boot, "--force"])

    # NOTE: lcd_display._clean_text() strips < > [ ] # * _ - (it treats < > as
    # XML/movement tags), so menu/volume rendering must avoid those characters.
    def _render_menu(self):
        item = self.menu[self._sel]
        lcd_manager.write("Settings:", f">> {item}")

    def _render_volume(self):
        bars = "=" * (self._vol // 10)
        lcd_manager.write(f"BT Vol: {self._vol}%", bars.ljust(10))
