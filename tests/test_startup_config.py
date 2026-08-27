import json
from pathlib import Path
from unittest.mock import patch

from core.startup_config import (
    StartupConfigWizard,
    apply_configured_volume,
    calibration_status,
    play_startup_config_beep,
    run_lcd_sync_calibration,
    save_config_atomic,
    set_default_sink_volume,
)
from core.wifi_setup import IRWifiSetup


def sample_config():
    return {
        "bluetooth_speaker": {},
        "llm": {"speaking_provider": "local"},
        "assistant_modes": {
            "active_on_startup": "default",
            "language_on_startup": "english",
            "modes": {
                "default": {"voice": ""},
                "jarvis": {"voice": "jarvis"},
                "senior": {"voice": ""},
            },
        },
    }


def test_no_keeps_config_and_does_not_write(tmp_path):
    path = tmp_path / "config.json"
    original = sample_config()
    path.write_text(json.dumps(original))
    wizard = StartupConfigWizard(lambda *_: None, original, str(path), beep=None)

    with patch.object(wizard, "_acquire"), \
            patch.object(wizard, "_release"), \
            patch.object(wizard, "_choose", return_value=0):
        result = wizard.run()

    assert result == original
    assert json.loads(path.read_text()) == original


def test_yes_applies_provider_and_mode_then_saves(tmp_path):
    path = tmp_path / "config.json"
    config = sample_config()
    path.write_text(json.dumps(config))
    wizard = StartupConfigWizard(lambda *_: None, config, str(path), beep=None)

    # Yes, keep Wi-Fi, Cerebras, 70% volume, Hindi, Jarvis, no calibration.
    with patch.object(wizard, "_acquire"), \
            patch.object(wizard, "_release"), \
            patch.object(wizard, "_adjust_volume", return_value=70), \
            patch("core.startup_config.calibration_status", return_value=(False, "")), \
            patch.object(wizard, "_choose", side_effect=[1, 0, 1, 1, 1, 0]), \
            patch("core.startup_config.time.sleep"):
        wizard.run()

    saved = json.loads(path.read_text())
    assert saved["llm"]["speaking_provider"] == "cerebras"
    assert saved["bluetooth_speaker"]["volume_percent"] == 70
    assert saved["assistant_modes"]["language_on_startup"] == "hindi"
    assert saved["assistant_modes"]["active_on_startup"] == "jarvis"
    assert wizard.calibration_requested is False
    # The wizard runs before the TTS/Whisper servers exist, so the answer is
    # only a request; it must never be persisted as a config setting.
    assert "calibration_requested" not in json.dumps(saved)


def test_calibration_is_requested_and_defaults_to_stale_profile(tmp_path):
    path = tmp_path / "config.json"
    config = sample_config()
    path.write_text(json.dumps(config))
    wizard = StartupConfigWizard(lambda *_: None, config, str(path), beep=None)
    prompts = []

    def answer(title, options, initial=0, timeout_s=None):
        prompts.append((title, initial))
        # Yes to entering the wizard, and Yes to the new calibration step.
        return 1 if title in ("Edit config?", "Sync LCD+audio?") else 0

    with patch.object(wizard, "_acquire"), \
            patch.object(wizard, "_release"), \
            patch.object(wizard, "_adjust_volume", return_value=50), \
            patch("core.startup_config.calibration_status",
                  return_value=(True, "speaker changed")), \
            patch.object(wizard, "_choose", side_effect=answer), \
            patch("core.startup_config.time.sleep"):
        wizard.run()

    assert ("Sync LCD+audio?", 1) in prompts
    assert wizard.calibration_requested is True


def test_calibration_status_reports_unusable_profile(tmp_path):
    missing = {"tts": {"display_sync": {
        "enabled": True,
        "profile_path": str(tmp_path / "absent.json"),
    }}}
    needs, reason = calibration_status(missing)

    assert needs is True
    assert "missing" in reason


def test_calibration_runner_streams_progress_and_reports_failure(tmp_path):
    script = tmp_path / "fake_calibrate.py"
    script.write_text(
        "print('Calibrating jarvis...')\n"
        "print('Validation: max=12.0ms p95=4.0ms samples=9')\n"
        "raise SystemExit(2)\n"
    )
    frames = []

    with patch("core.startup_config.time.sleep"):
        ok = run_lcd_sync_calibration(
            {}, lambda a, b: frames.append((a, b)),
            python_executable=__import__("sys").executable,
            script_path=str(script),
        )

    assert ok is False
    assert ("Calibrating...", "jarvis") in frames
    assert frames[-1] == ("Calib failed", "using old sync")


def test_calibration_runner_reports_success(tmp_path):
    script = tmp_path / "fake_calibrate.py"
    script.write_text("print('Wrote profile')\n")
    frames = []

    with patch("core.startup_config.time.sleep"):
        ok = run_lcd_sync_calibration(
            {}, lambda a, b: frames.append((a, b)),
            python_executable=__import__("sys").executable,
            script_path=str(script),
        )

    assert ok is True
    assert frames[-1] == ("Sync calibrated", "LCD + audio OK")


def test_calibration_runner_kills_a_silently_stuck_script(tmp_path):
    # Prints nothing and never exits: the deadline cannot come from the read
    # loop, so only the watchdog can stop this from hanging the whole boot.
    script = tmp_path / "stuck_calibrate.py"
    script.write_text("import time\nwhile True:\n    time.sleep(1)\n")
    frames = []

    with patch("core.startup_config.time.sleep"):
        ok = run_lcd_sync_calibration(
            {}, lambda a, b: frames.append((a, b)),
            python_executable=__import__("sys").executable,
            script_path=str(script),
            timeout_s=1.0,
        )

    assert ok is False
    assert frames[-1] == ("Calib timeout", "using old sync")


def test_calibration_runner_survives_a_missing_script(tmp_path):
    with patch("core.startup_config.time.sleep"):
        assert run_lcd_sync_calibration(
            {}, lambda *_: None, script_path=str(tmp_path / "nope.py")
        ) is False


def test_wifi_selector_releases_and_reacquires_gpio(tmp_path):
    path = tmp_path / "config.json"
    config = sample_config()
    path.write_text(json.dumps(config))
    calls = []
    wizard = StartupConfigWizard(
        lambda *_: None,
        config,
        str(path),
        beep=None,
        wifi_runner=lambda *_args: calls.append("wifi") or True,
    )

    with patch.object(wizard, "_acquire", side_effect=lambda: calls.append("acquire")), \
            patch.object(wizard, "_release", side_effect=lambda: calls.append("release")), \
            patch.object(wizard, "_adjust_volume", return_value=50), \
            patch("core.startup_config.calibration_status", return_value=(False, "")), \
            patch.object(wizard, "_choose", side_effect=[1, 1, 0, 0, 0, 0]), \
            patch("core.startup_config.time.sleep"):
        wizard.run()

    assert calls[:4] == ["acquire", "release", "wifi", "acquire"]


def test_atomic_save_leaves_valid_json(tmp_path):
    path = tmp_path / "config.json"
    save_config_atomic({"mode": "वरिष्ठ"}, str(path))

    assert json.loads(path.read_text()) == {"mode": "वरिष्ठ"}
    assert not list(Path(tmp_path).glob(".startup-config-*"))


def test_beep_plays_asynchronously():
    with patch("core.startup_config.os.path.isfile", return_value=True), \
            patch("core.startup_config.subprocess.Popen") as popen:
        play_startup_config_beep("/tmp/beep.wav")

    assert popen.call_args.args[0] == ["play", "-q", "/tmp/beep.wav"]


def test_volume_step_applies_live_and_hold_confirms(tmp_path):
    wizard = StartupConfigWizard(
        lambda *_: None,
        sample_config(),
        str(tmp_path / "config.json"),
        beep=lambda: None,
        volume_getter=lambda: 50,
        volume_setter=lambda value: value,
    )
    readings = [
        (False, False),  # initial all-clear
        (True, False),   # LEFT tap begins
        (False, False),  # LEFT tap released
        (True, False),   # LEFT hold begins
        (True, False),   # hold crosses select threshold
        (False, False),  # release after selection
    ]

    with patch.object(wizard, "_read", side_effect=readings), \
            patch("core.startup_config.time.monotonic",
                  side_effect=[0.0, 0.1, 1.0, 2.1]):
        assert wizard._adjust_volume() == 60


def test_saved_volume_is_clamped_and_applied():
    with patch("core.startup_config.set_default_sink_volume",
               return_value=100) as setter:
        assert apply_configured_volume(
            {"bluetooth_speaker": {"volume_percent": 140}}
        ) == 100

    setter.assert_called_once_with(140)


def test_volume_setter_targets_default_pulse_sink():
    completed = __import__("subprocess").CompletedProcess([], 0, "", "")
    with patch("core.startup_config.subprocess.run",
               return_value=completed) as run:
        assert set_default_sink_volume(70) == 70

    assert run.call_args.args[0] == [
        "pactl", "set-sink-volume", "@DEFAULT_SINK@", "70%",
    ]


def test_failed_forced_wifi_change_restores_previous_profile():
    setup = IRWifiSetup(display=lambda *_: None, config={"stt": {}})
    previous = ("Home WiFi", "old-profile-uuid")
    completed = __import__("subprocess").CompletedProcess([], 0, "", "")

    with patch.object(setup, "_run_nmcli", return_value=completed) as run_nmcli, \
            patch.object(setup, "is_connected_to", return_value=True):
        assert setup.restore_connection(previous)

    assert run_nmcli.call_args.args[0] == [
        "--wait", "30", "connection", "up", "uuid", "old-profile-uuid",
        "ifname", "wlan0",
    ]
