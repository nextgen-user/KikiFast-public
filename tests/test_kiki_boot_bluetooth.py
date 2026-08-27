from types import SimpleNamespace
from unittest.mock import patch

import kiki_boot


def completed(stdout="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


def test_find_paired_speaker_name_is_case_insensitive():
    devices = (
        "Device 02:00:00:38:FB:DD Headphones\n"
        "Device 02:00:00:38:FB:DD EXAMPLE SPEAKER\n"
    )
    with patch.object(kiki_boot, "_run_quiet", return_value=completed(devices)):
        assert (
            kiki_boot._find_paired_bluetooth_device("Example Speaker")
            == "02:00:00:38:FB:DD"
        )


def test_connects_configured_speaker_and_selects_sink():
    settings = {
        "name": "Example Speaker",
        "mac": "02:00:00:38:FB:DD",
        "connect_timeout_seconds": 5,
        "retry_interval_seconds": 0.25,
    }
    with (
        patch.object(kiki_boot, "_run_quiet", return_value=completed()),
        patch.object(kiki_boot, "_bluetooth_is_connected",
                     side_effect=[False, True]),
        patch.object(kiki_boot, "_select_bluetooth_audio_sink",
                     return_value=True) as select_sink,
    ):
        assert kiki_boot.connect_bluetooth_speaker(settings) is True
        select_sink.assert_called_once()


def test_already_connected_speaker_does_not_reconnect():
    settings = {
        "name": "Example Speaker",
        "mac": "02:00:00:38:FB:DD",
        "connect_timeout_seconds": 5,
    }
    with (
        patch.object(kiki_boot, "_run_quiet", return_value=completed()) as run,
        patch.object(kiki_boot, "_bluetooth_is_connected", return_value=True),
        patch.object(kiki_boot, "_select_bluetooth_audio_sink",
                     return_value=True),
    ):
        assert kiki_boot.connect_bluetooth_speaker(settings) is True
        assert ["bluetoothctl", "connect", settings["mac"]] not in [
            call.args[0] for call in run.call_args_list
        ]


def test_failed_stale_mac_falls_back_to_paired_name():
    old_mac = "02:00:00:38:FB:DD"
    new_mac = "02:00:00:38:FB:DD"
    settings = {
        "name": "Example Speaker",
        "mac": old_mac,
        "connect_timeout_seconds": 5,
        "retry_interval_seconds": 0.25,
    }
    with (
        patch.object(kiki_boot, "_run_quiet", return_value=completed()),
        patch.object(kiki_boot, "_bluetooth_is_connected",
                     side_effect=[False, False, True]),
        patch.object(kiki_boot, "_find_paired_bluetooth_device",
                     return_value=new_mac),
        patch.object(kiki_boot, "_select_bluetooth_audio_sink",
                     return_value=True) as select_sink,
        patch.object(kiki_boot.time, "sleep"),
    ):
        assert kiki_boot.connect_bluetooth_speaker(settings) is True
        select_sink.assert_called_once()
        assert select_sink.call_args.args[0] == new_mac


def test_connected_without_a2dp_sink_is_not_reported_as_ready():
    settings = {
        "name": "Example Speaker",
        "mac": "02:00:00:38:FB:DD",
        "connect_timeout_seconds": 0.01,
        "retry_interval_seconds": 0.01,
    }
    with (
        patch.object(kiki_boot, "_run_quiet", return_value=completed()),
        patch.object(kiki_boot, "_bluetooth_is_connected", return_value=True),
        patch.object(kiki_boot, "_select_bluetooth_audio_sink",
                     return_value=False),
        patch.object(kiki_boot.time, "sleep"),
    ):
        assert kiki_boot.connect_bluetooth_speaker(settings) is False
