"""Boot-time IR/LCD configuration wizard.

This module deliberately stays independent from ``main.py``. It runs after the
Bluetooth speaker is connected, owns GPIO22/17 only while a choice is on screen,
and releases the lines before handing Wi-Fi selection to :mod:`core.wifi_setup`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Optional

try:
    import gpiod
    from gpiod.line import Direction, Value

    GPIOD_AVAILABLE = True
except Exception:  # pragma: no cover - robot-only dependency
    gpiod = None
    Direction = Value = None
    GPIOD_AVAILABLE = False


CHIP_PATH = "/dev/gpiochip4"
LEFT_PIN = 22
RIGHT_PIN = 17
POLL_S = 0.03
MIN_TAP_S = 0.05
SELECT_HOLD_S = 1.0
EDIT_PROMPT_TIMEOUT_S = 20.0
# An IR sensor that is obstructed, unpowered or wired to a module that boots
# asserted reads "held" forever. Waiting for it to clear must never outlast this
# budget, or the whole boot stalls on whatever screen was last drawn.
SENSOR_CLEAR_TIMEOUT_S = 5.0
VOLUME_STEP = 10
VOLUME_MIN = 0
VOLUME_MAX = 100

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BEEP = os.path.join(
    ROOT, "sound_effects", "soundeffects", "startup_config_beep.wav"
)
CALIBRATION_SCRIPT = os.path.join(ROOT, "tests", "calibrate_lcd_sync.py")
CALIBRATION_TIMEOUT_S = 900.0
# Mirrors core.tts._SAMPLE_RATE; imported here would drag the whole TTS stack
# into the boot wizard just to answer one yes/no question.
TTS_SAMPLE_RATE = 24000


class SensorsStuckError(RuntimeError):
    """Raised when an IR sensor never releases, so the wizard cannot be driven."""


def play_startup_config_beep(path: str = DEFAULT_BEEP) -> None:
    """Play the short wizard feedback sound without delaying IR polling."""
    if not os.path.isfile(path):
        return
    try:
        subprocess.Popen(
            ["play", "-q", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def get_default_sink_volume() -> int:
    """Read the first channel percentage from PulseAudio's default sink."""
    try:
        result = subprocess.run(
            ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        match = re.search(r"(\d+)%", result.stdout or "")
        if result.returncode == 0 and match:
            return max(VOLUME_MIN, min(VOLUME_MAX, int(match.group(1))))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return 50


def set_default_sink_volume(percent: int) -> int:
    """Apply and return a clamped PulseAudio default-sink volume."""
    percent = max(VOLUME_MIN, min(VOLUME_MAX, int(percent)))
    try:
        subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        pass
    return percent


def apply_configured_volume(config: dict) -> Optional[int]:
    """Apply a previously saved startup volume, if the setting exists."""
    value = config.get("bluetooth_speaker", {}).get("volume_percent")
    if value is None:
        return None
    try:
        volume = set_default_sink_volume(int(value))
    except (TypeError, ValueError):
        print(f"[StartupConfig] Ignoring invalid saved volume: {value!r}", flush=True)
        return None
    print(f"[StartupConfig] Applied saved volume: {volume}%", flush=True)
    return volume


def calibration_status(config: dict) -> tuple[bool, str]:
    """Return (needs_calibration, reason) for the saved LCD/audio sync profile.

    Used only to pick the wizard's default answer, so an unexpected failure
    here must never look like a stale profile and never raise into boot.
    """
    try:
        from core.tts_sync import TimingCalibration

        tts_cfg = config.get("tts", {}) or {}
        calibration = TimingCalibration.load(tts_cfg, TTS_SAMPLE_RATE)
        if calibration.valid:
            return False, ""
        return True, calibration.reason or "profile unusable"
    except Exception as exc:  # pragma: no cover - defensive boot guard
        print(f"[StartupConfig] calibration check skipped: {exc}", flush=True)
        return False, ""


def _calibration_progress(line: str) -> Optional[tuple[str, str]]:
    """Map one calibration stdout line to a two-row LCD frame."""
    if line.startswith("Calibrating "):
        return "Calibrating...", line[len("Calibrating "):].rstrip(".")
    if line.startswith("Playing three"):
        return "Speaker test", "listening..."
    if line.startswith("Acoustic output latency:"):
        return "Speaker delay", line.split(":", 1)[1].split("(")[0].strip()
    if line.startswith("Validation:"):
        return "Checking sync", line.split("max=", 1)[-1].split(" ")[0]
    return None


def run_lcd_sync_calibration(
    config: dict,
    display: Callable[[str, str], None],
    *,
    python_executable: str = sys.executable,
    script_path: str = CALIBRATION_SCRIPT,
    timeout_s: Optional[float] = None,
) -> bool:
    """Run the offline LCD/audio sync calibration, streaming progress to the LCD.

    This is deliberately invoked from the boot orchestrator *after* the TTS and
    Whisper servers are live and *after* the looping boot sound has stopped:
    the script measures real speaker→microphone latency, so any other audio
    playing would corrupt the measurement. Failure is never fatal — Kiki keeps
    booting and the LCD simply falls back to its uncalibrated speech display.
    """
    def show(line1: str, line2: str = "") -> None:
        try:
            display(line1[:16], line2[:16])
        except Exception:
            pass

    if not os.path.isfile(script_path):
        print(f"[StartupConfig] calibration script missing: {script_path}", flush=True)
        show("Calib skipped", "script missing")
        time.sleep(1.0)
        return False

    if timeout_s is None:
        sync_cfg = (config.get("tts", {}) or {}).get("display_sync", {}) or {}
        try:
            timeout_s = float(sync_cfg.get("calibration_timeout_seconds",
                                           CALIBRATION_TIMEOUT_S))
        except (TypeError, ValueError):
            timeout_s = CALIBRATION_TIMEOUT_S

    show("Calibrating...", "LCD + audio")
    process = None
    watchdog = None
    timed_out = threading.Event()
    try:
        process = subprocess.Popen(
            [python_executable, script_path],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def kill_on_timeout() -> None:
            # Reading stdout blocks, so the deadline cannot be enforced from
            # the read loop: a silently stuck calibration would hang the boot.
            timed_out.set()
            if process.poll() is None:
                process.kill()

        watchdog = threading.Timer(max(1.0, float(timeout_s)), kill_on_timeout)
        watchdog.daemon = True
        watchdog.start()

        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            print(f"[Calibration] {line}", flush=True)
            frame = _calibration_progress(line)
            if frame:
                show(*frame)
        code = process.wait(timeout=30)
    except Exception as exc:
        print(f"[StartupConfig] calibration failed: {exc}", flush=True)
        if process is not None and process.poll() is None:
            process.kill()
        show("Calib failed", "using old sync")
        time.sleep(1.5)
        return False
    finally:
        if watchdog is not None:
            watchdog.cancel()
        if process is not None and process.stdout is not None:
            try:
                process.stdout.close()
            except Exception:
                pass

    if timed_out.is_set():
        print("[StartupConfig] calibration timed out", flush=True)
        show("Calib timeout", "using old sync")
        time.sleep(1.5)
        return False
    if code == 0:
        show("Sync calibrated", "LCD + audio OK")
        time.sleep(1.0)
        return True
    # The script refuses to overwrite a known-good profile when validation
    # fails, so the previous calibration (if any) is still intact here.
    print(f"[StartupConfig] calibration exited with {code}", flush=True)
    show("Calib failed", "using old sync")
    time.sleep(1.5)
    return False


def save_config_atomic(config: dict, path: str) -> None:
    """Persist startup defaults without risking a partial JSON file."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(
        prefix=".startup-config-", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2, ensure_ascii=False)
            config_file.write("\n")
            config_file.flush()
            os.fsync(config_file.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


class StartupConfigWizard:
    """Synchronous choice wizard using the same tap/hold model as Settings."""

    def __init__(
        self,
        display: Callable[[str, str], None],
        config: dict,
        config_path: str,
        *,
        password_display: Optional[Callable[[str, str], None]] = None,
        beep: Optional[Callable[[], None]] = play_startup_config_beep,
        wifi_runner: Optional[Callable[[dict, Callable, Callable, Callable], bool]] = None,
        volume_getter: Callable[[], int] = get_default_sink_volume,
        volume_setter: Callable[[int], int] = set_default_sink_volume,
    ):
        self.display = display
        self.password_display = password_display or display
        self.config = config
        self.config_path = config_path
        self.beep = beep
        self.wifi_runner = wifi_runner or self._run_wifi_selector
        self.volume_getter = volume_getter
        self.volume_setter = volume_setter
        self.request = None
        # Set by run(); the calibration itself cannot happen here because the
        # TTS/Whisper servers it needs are only started later in the boot.
        self.calibration_requested = False

    def _show(self, line1: str, line2: str = "") -> None:
        self.display(line1[:16], line2[:16])

    def _feedback(self) -> None:
        if self.beep is None:
            return
        try:
            self.beep()
        except Exception as exc:
            print(f"[StartupConfig] beep failed: {exc}", flush=True)

    def _acquire(self) -> None:
        if self.request is not None:
            return
        self.request = gpiod.request_lines(
            CHIP_PATH,
            consumer="kiki-startup-config",
            config={
                LEFT_PIN: gpiod.LineSettings(direction=Direction.INPUT),
                RIGHT_PIN: gpiod.LineSettings(direction=Direction.INPUT),
            },
        )

    def _release(self) -> None:
        if self.request is None:
            return
        try:
            self.request.release()
        finally:
            self.request = None

    def _read(self) -> tuple[bool, bool]:
        return (
            self.request.get_value(LEFT_PIN) == Value.INACTIVE,
            self.request.get_value(RIGHT_PIN) == Value.INACTIVE,
        )

    def _wait_clear(self, timeout_s: float = SENSOR_CLEAR_TIMEOUT_S) -> None:
        """Wait for both sensors to read clear, or declare them stuck."""
        if self.request is None:
            return
        left, right = self._read()
        if not left and not right:
            return
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            if time.monotonic() >= deadline:
                stuck = " and ".join(
                    name for name, held in
                    (("LEFT", left), ("RIGHT", right)) if held
                )
                raise SensorsStuckError(
                    f"{stuck} IR sensor still reads held after {timeout_s:g}s")
            time.sleep(POLL_S)
            left, right = self._read()
            if not left and not right:
                return

    def _render_choice(self, title: str, options: list[str], index: int) -> None:
        self._show(title, f">> {options[index]}")

    def _render_volume(self, volume: int) -> None:
        bars = "=" * (volume // VOLUME_STEP)
        self._show(f"BT Vol: {volume}%", bars)

    def _choose(
        self,
        title: str,
        options: list[str],
        initial: int = 0,
        timeout_s: Optional[float] = None,
    ) -> int:
        """Tap LEFT/RIGHT to move and hold either sensor to select."""
        if not options:
            raise ValueError(f"{title} has no choices")
        index = max(0, min(int(initial), len(options) - 1))
        self._wait_clear()
        self._render_choice(title, options, index)
        idle_deadline = (
            time.monotonic() + max(0, timeout_s)
            if timeout_s is not None else None
        )

        while True:
            if idle_deadline is not None and time.monotonic() >= idle_deadline:
                return index
            left, right = self._read()
            if left and right:
                self._wait_clear()
                continue
            pin = LEFT_PIN if left else RIGHT_PIN if right else None
            if pin is None:
                time.sleep(POLL_S)
                continue

            started = time.monotonic()
            while True:
                now_left, now_right = self._read()
                held = now_left if pin == LEFT_PIN else now_right
                elapsed = time.monotonic() - started
                if held and elapsed >= SELECT_HOLD_S:
                    self._feedback()
                    self._wait_clear()
                    return index
                if not held:
                    if elapsed >= MIN_TAP_S:
                        # Same physical mapping as runtime Settings.
                        delta = 1 if pin == LEFT_PIN else -1
                        index = (index + delta) % len(options)
                        self._feedback()
                        self._render_choice(title, options, index)
                        if timeout_s is not None:
                            idle_deadline = time.monotonic() + max(0, timeout_s)
                    break
                time.sleep(POLL_S)

    def _adjust_volume(self) -> int:
        """Tap to adjust live; hold either sensor to confirm and continue."""
        try:
            volume = max(
                VOLUME_MIN,
                min(VOLUME_MAX, int(self.volume_getter())),
            )
        except (TypeError, ValueError, OSError):
            volume = 50
        self._wait_clear()
        self._render_volume(volume)

        while True:
            left, right = self._read()
            if left and right:
                self._wait_clear()
                continue
            pin = LEFT_PIN if left else RIGHT_PIN if right else None
            if pin is None:
                time.sleep(POLL_S)
                continue

            started = time.monotonic()
            while True:
                now_left, now_right = self._read()
                held = now_left if pin == LEFT_PIN else now_right
                elapsed = time.monotonic() - started
                if held and elapsed >= SELECT_HOLD_S:
                    self._feedback()
                    self._wait_clear()
                    return volume
                if not held:
                    if elapsed >= MIN_TAP_S:
                        delta = VOLUME_STEP if pin == LEFT_PIN else -VOLUME_STEP
                        volume = self.volume_setter(volume + delta)
                        volume = max(VOLUME_MIN, min(VOLUME_MAX, int(volume)))
                        self._feedback()
                        self._render_volume(volume)
                    break
                time.sleep(POLL_S)

    @staticmethod
    def _initial_index(options: list[str], current: str) -> int:
        wanted = str(current or "").casefold()
        return next(
            (index for index, option in enumerate(options)
             if option.casefold() == wanted),
            0,
        )

    @staticmethod
    def _run_wifi_selector(config, display, password_display, feedback) -> bool:
        from core.wifi_setup import IRWifiSetup

        setup = IRWifiSetup(
            display=display,
            config=config,
            password_display=password_display,
            feedback=feedback,
        )
        return setup.run(force=True)

    def run(self) -> dict:
        """Run the optional wizard and return the in-memory config object."""
        if not GPIOD_AVAILABLE:
            print("[StartupConfig] gpiod unavailable; skipping wizard.", flush=True)
            return self.config

        try:
            self._acquire()
            # A stuck sensor cannot be answered, so the wizard is unusable: keep
            # the saved config and let the rest of the boot run.
            self._wait_clear()
            edit = self._choose(
                "Edit config?",
                ["No", "Yes"],
                timeout_s=EDIT_PROMPT_TIMEOUT_S,
            )
            if edit == 0:
                print("[StartupConfig] Keeping startup defaults.", flush=True)
                return self.config

            wifi_choice = self._choose("Switch WiFi?", ["Keep current", "Change WiFi"])
            if wifi_choice == 1:
                # Wi-Fi setup temporarily owns the same GPIO lines.
                self._release()
                try:
                    wifi_ready = self.wifi_runner(
                        self.config,
                        self.display,
                        self.password_display,
                        self._feedback,
                    )
                    if not wifi_ready:
                        self._show("WiFi unchanged", "using previous")
                        time.sleep(1.0)
                except Exception as exc:
                    print(
                        f"[StartupConfig] Wi-Fi change failed: {exc}",
                        flush=True,
                    )
                    self._show("WiFi unchanged", "using previous")
                    time.sleep(1.0)
                finally:
                    self._acquire()

            llm = self.config.setdefault("llm", {})
            providers = ["local", "cerebras"]
            provider_index = self._choose(
                "Speaking brain",
                ["Local", "Cerebras"],
                self._initial_index(providers, llm.get("speaking_provider", "local")),
            )
            llm["speaking_provider"] = providers[provider_index]

            selected_volume = self._adjust_volume()
            bluetooth = self.config.setdefault("bluetooth_speaker", {})
            bluetooth["volume_percent"] = selected_volume

            assistant_modes = self.config.setdefault("assistant_modes", {})
            languages = ["english", "hindi"]
            language_index = self._choose(
                "Language",
                ["English", "Hindi"],
                self._initial_index(
                    languages,
                    assistant_modes.get("language_on_startup", "english"),
                ),
            )
            selected_language = languages[language_index]
            assistant_modes["language_on_startup"] = selected_language

            modes = assistant_modes.get("modes", {})
            mode_names = list(modes) if isinstance(modes, dict) and modes else ["default"]
            current_mode = assistant_modes.get("active_on_startup", "default")
            mode_index = self._choose(
                "Startup mode",
                mode_names,
                self._initial_index(mode_names, current_mode),
            )
            selected_mode = mode_names[mode_index]
            assistant_modes["active_on_startup"] = selected_mode

            # Speaker/LCD sync is offered last because the answer depends on
            # every choice above: a new speaker, TTS server or mode voice all
            # invalidate the saved word-timing profile. A stale profile
            # preselects Yes so the common case is one hold.
            needs_calibration, reason = calibration_status(self.config)
            if needs_calibration:
                print(f"[StartupConfig] sync profile stale: {reason}", flush=True)
            calibration_index = self._choose(
                "Sync LCD+audio?",
                ["No", "Yes"],
                1 if needs_calibration else 0,
            )
            self.calibration_requested = calibration_index == 1

            save_config_atomic(self.config, self.config_path)
            print(
                "[StartupConfig] Saved "
                f"provider={llm['speaking_provider']} "
                f"volume={selected_volume}% language={selected_language} "
                f"mode={selected_mode} calibrate={self.calibration_requested}",
                flush=True,
            )
            self._show("Config saved", f"Boot: {selected_mode}")
            time.sleep(0.8)
            if self.calibration_requested:
                self._show("Sync after boot", "servers first")
                time.sleep(1.0)
            return self.config
        except SensorsStuckError as exc:
            print(f"[StartupConfig] {exc}; skipping wizard.", flush=True)
            self._show("IR sensor stuck", "skipping config")
            time.sleep(1.2)
            return self.config
        finally:
            self._release()


def offer_startup_config(
    config: dict,
    config_path: str,
    display: Callable[[str, str], None],
    password_display: Optional[Callable[[str, str], None]] = None,
) -> tuple[dict, bool]:
    """Best-effort boot entry point; missing robot GPIO never blocks startup.

    Returns the in-memory config plus whether the user asked for the LCD/audio
    sync calibration, which the caller must run once the TTS/Whisper servers
    are live (see :func:`run_lcd_sync_calibration`).
    """
    wizard = StartupConfigWizard(
        display,
        config,
        config_path,
        password_display=password_display,
    )
    updated = wizard.run()
    return updated, wizard.calibration_requested
