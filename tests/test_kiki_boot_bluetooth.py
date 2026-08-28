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
        patch.object(kiki_boot, "_persist_speaker_mac") as persist,
        patch.object(kiki_boot.time, "sleep"),
    ):
        assert kiki_boot.connect_bluetooth_speaker(settings) is True
        select_sink.assert_called_once()
        assert select_sink.call_args.args[0] == new_mac
        # The new address is kept for the next boot, in memory and on disk.
        assert settings["mac"] == new_mac
        persist.assert_called_once_with(new_mac)


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


def test_speaker_name_matches_a_renamed_paired_device():
    # config.json says "Example Speaker"; the speaker broadcasts "Example Speaker Pro".
    devices = (
        "Device 02:00:00:38:FB:DD OnePlus Nord Buds 3\n"
        "Device 02:00:00:38:FB:DD Example Speaker Pro\n"
    )
    with patch.object(kiki_boot, "_run_quiet", return_value=completed(devices)):
        assert (
            kiki_boot._find_paired_bluetooth_device("Example Speaker")
            == "02:00:00:38:FB:DD"
        )


def test_failed_connection_unpairs_and_pairs_the_speaker_again():
    mac = "02:00:00:38:FB:DD"
    settings = {
        "name": "Example Speaker",
        "mac": mac,
        "connect_timeout_seconds": 0.01,
        "retry_interval_seconds": 0.01,
        "repair_scan_seconds": 4,
    }
    issued = []
    # The speaker only answers once the stale bond is gone and it is paired
    # again -- the exact state a plain `connect` can never recover from.
    paired_again = {"done": False}

    def run_quiet(command, timeout=8):
        issued.append(command)
        if command[:2] == ["bluetoothctl", "pair"]:
            paired_again["done"] = True
        return completed(f"Device {mac} Example Speaker\n")

    with (
        patch.object(kiki_boot, "_run_quiet", side_effect=run_quiet),
        patch.object(kiki_boot, "_bluetooth_is_connected",
                     side_effect=lambda _mac: paired_again["done"]),
        patch.object(kiki_boot, "_bluetooth_is_present", return_value=True),
        patch.object(kiki_boot, "_bluetooth_info_says", return_value=True),
        patch.object(kiki_boot, "_select_bluetooth_audio_sink", return_value=True),
        patch.object(kiki_boot, "_persist_speaker_mac"),
        patch.object(kiki_boot.time, "sleep"),
    ):
        assert kiki_boot.connect_bluetooth_speaker(settings) is True

    assert ["bluetoothctl", "remove", mac] in issued
    assert ["bluetoothctl", "pair", mac] in issued
    assert ["bluetoothctl", "trust", mac] in issued
    assert issued.index(["bluetoothctl", "remove", mac]) < issued.index(
        ["bluetoothctl", "pair", mac])


def test_absent_speaker_keeps_its_pairing():
    # A speaker that is merely switched off must never be unpaired: re-pairing
    # needs someone standing next to it holding the pairing button.
    mac = "02:00:00:38:FB:DD"
    settings = {
        "name": "Example Speaker",
        "mac": mac,
        "connect_timeout_seconds": 0.01,
        "retry_interval_seconds": 0.01,
    }
    with (
        patch.object(kiki_boot, "_run_quiet",
                     return_value=completed(f"Device {mac} Example Speaker\n")) as run,
        patch.object(kiki_boot, "_bluetooth_is_connected", return_value=False),
        patch.object(kiki_boot, "_bluetooth_is_present", return_value=False),
        patch.object(kiki_boot.time, "sleep"),
    ):
        assert kiki_boot.connect_bluetooth_speaker(settings) is False
        issued = [call.args[0] for call in run.call_args_list]
        assert ["bluetoothctl", "remove", mac] not in issued
        assert ["bluetoothctl", "pair", mac] not in issued


def test_repair_rediscovers_the_speaker_after_removing_the_bond():
    # `bluetoothctl remove` drops the device from BlueZ entirely, so `pair`
    # answers "not available" unless a fresh inquiry runs after the removal.
    mac = "02:00:00:38:FB:DD"
    issued = []
    known = {"listed": True}

    def run_quiet(command, timeout=8):
        issued.append(command)
        if command[:2] == ["bluetoothctl", "remove"]:
            known["listed"] = False       # BlueZ forgets the device
        if command[:3] == ["bluetoothctl", "--timeout", str(12)]:
            known["listed"] = True        # the scan finds it again
        listing = f"Device {mac} Example Speaker Pro\n" if known["listed"] else ""
        return completed(listing)

    with (
        patch.object(kiki_boot, "_run_quiet", side_effect=run_quiet),
        patch.object(kiki_boot, "_bluetooth_info_says", return_value=True),
        patch.object(kiki_boot, "_bluetooth_connect", return_value=(True, True)),
    ):
        assert kiki_boot._repair_bluetooth_pairing(
            "Example Speaker", mac, {"repair_scan_seconds": 12}, answered=True) == mac

    scans = [i for i, c in enumerate(issued) if c[:2] == ["bluetoothctl", "--timeout"]]
    removed = issued.index(["bluetoothctl", "remove", mac])
    paired = issued.index(["bluetoothctl", "pair", mac])
    assert any(removed < scan < paired for scan in scans)


def test_a_refused_profile_still_counts_as_a_reachable_speaker():
    # br-connection-refused: the ACL completed, the speaker then refused audio.
    # That is a switched-on speaker with a stale bond, not an absent one.
    mac = "02:00:00:38:FB:DD"
    refused = completed(
        "Attempting to connect to 02:00:00:38:FB:DD\n"
        "Failed to connect: org.bluez.Error.Failed br-connection-refused\n")
    with (
        patch.object(kiki_boot, "_run_quiet", return_value=refused),
        patch.object(kiki_boot, "_bluetooth_is_connected", return_value=False),
    ):
        assert kiki_boot._bluetooth_connect(mac) == (False, True)


def test_a_page_timeout_means_the_speaker_is_switched_off():
    mac = "02:00:00:38:FB:DD"
    timed_out = completed(
        "Attempting to connect to 02:00:00:38:FB:DD\n"
        "Failed to connect: org.bluez.Error.Failed br-connection-page-timeout\n")
    with (
        patch.object(kiki_boot, "_run_quiet", return_value=timed_out),
        patch.object(kiki_boot, "_bluetooth_is_connected", return_value=False),
    ):
        assert kiki_boot._bluetooth_connect(mac) == (False, False)
