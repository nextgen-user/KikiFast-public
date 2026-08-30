from unittest.mock import patch

from core.wifi_setup import (
    IRWifiSetup,
    WifiNetwork,
    apply_password_dictation,
    parse_nmcli_networks,
)


def test_parse_nmcli_networks_unescapes_sorts_and_deduplicates():
    output = (
        r":Cafe\: Upstairs:61:WPA2" "\n"
        r":Home\\Lab:40:WPA2" "\n"
        r":Cafe\: Upstairs:83:WPA2" "\n"
        r":Guest:70:--" "\n"
        r"::99:--" "\n"
    )

    networks = parse_nmcli_networks(output)

    assert [(item.ssid, item.signal, item.is_open) for item in networks] == [
        ("Cafe: Upstairs", 83, False),
        ("Guest", 70, True),
        (r"Home\Lab", 40, False),
    ]


def test_parse_nmcli_network_keeps_bssid_for_reliable_connection():
    networks = parse_nmcli_networks(
        r":02\:00\:00\:38\:FB\:DD:Galaxy:100:WPA2" "\n"
    )

    assert networks == [
        WifiNetwork("Galaxy", signal=100, security="WPA2", bssid="02:00:00:38:FB:DD")
    ]


def test_preferred_hidden_hotspot_is_kept_selectable():
    setup = IRWifiSetup(
        display=lambda *_args: None,
        config={
            "stt": {},
            "wifi_setup": {"preferred_ssids": ["ExampleNet"]},
        },
    )
    completed = __import__("subprocess").CompletedProcess

    with patch.object(
        setup,
        "_run_nmcli",
        side_effect=[
            completed([], 0, "", ""),  # directed probe
            completed([], 0, ":Home:75:WPA2\n", ""),  # visible list
        ],
    ) as run_nmcli:
        networks = setup.scan()

    assert networks[-1] == WifiNetwork(
        "ExampleNet", signal=0, security="secured", hidden=True
    )
    assert run_nmcli.call_args_list[0].args[0] == [
        "device", "wifi", "rescan", "ifname", "wlan0", "ssid", "ExampleNet",
    ]


def test_visible_preferred_hotspot_is_not_duplicated():
    setup = IRWifiSetup(
        display=lambda *_args: None,
        config={
            "stt": {},
            "wifi_setup": {"preferred_ssids": ["ExampleNet"]},
        },
    )
    completed = __import__("subprocess").CompletedProcess

    with patch.object(
        setup,
        "_run_nmcli",
        side_effect=[
            completed([], 0, "", ""),
            completed([], 0, ":02\\:00\\:00\\:38\\:FB\\:DD:ExampleNet:88:WPA2\n", ""),
        ],
    ):
        networks = setup.scan()

    assert networks == [
        WifiNetwork(
            "ExampleNet",
            signal=88,
            security="WPA2",
            bssid="02:00:00:38:FB:DD",
        )
    ]


def test_letter_by_letter_dictation_defaults_to_lowercase():
    password, edits, unknown = apply_password_dictation("", "K, I, K, I.")

    assert password == "kiki"
    assert edits == 4
    assert unknown == []


def test_capital_before_or_after_letter_is_supported():
    password, _, _ = apply_password_dictation("", "capital k, i, k capital, i")

    assert password == "KiKi"


def test_spoken_clear_and_backspace_edit_existing_password():
    password, _, _ = apply_password_dictation(
        "wrong", "clear, kay eye kay eye, back space, capital eye"
    )

    assert password == "kikI"


def test_whisper_joined_spelling_is_split_into_characters():
    password, _, unknown = apply_password_dictation("", "the password is kiki")

    assert password == "kiki"
    assert unknown == ["kiki"]


def test_password_is_sent_over_stdin_not_process_arguments():
    setup = IRWifiSetup(display=lambda *_args: None, config={"stt": {}})
    completed = __import__("subprocess").CompletedProcess([], 0, "", "")

    with patch.object(setup, "_run_nmcli", return_value=completed) as run_nmcli, \
            patch.object(setup, "is_connected_to", return_value=True):
        assert setup._try_password(WifiNetwork("Home", 90, "WPA2"), "VerySecret7")

    args = run_nmcli.call_args.args[0]
    kwargs = run_nmcli.call_args.kwargs
    assert "VerySecret7" not in args
    assert kwargs["input_text"] == "VerySecret7\n"


def test_hidden_hotspot_connection_uses_networkmanager_hidden_mode():
    setup = IRWifiSetup(display=lambda *_args: None, config={"stt": {}})
    completed = __import__("subprocess").CompletedProcess([], 0, "", "")

    with patch.object(setup, "_run_nmcli", return_value=completed) as run_nmcli, \
            patch.object(setup, "is_connected_to", return_value=True):
        assert setup._try_password(
            WifiNetwork("ExampleNet", security="secured", hidden=True),
            "VerySecret7",
        )

    assert run_nmcli.call_args.args[0][-2:] == ["hidden", "yes"]


def test_missing_hotspot_is_distinguished_from_wrong_password():
    setup = IRWifiSetup(display=lambda *_args: None, config={"stt": {}})
    failed = __import__("subprocess").CompletedProcess(
        [],
        10,
        "",
        "Error: No network with SSID 'ExampleNet' found.",
    )

    with patch.object(setup, "_run_nmcli", return_value=failed):
        assert not setup._try_password(
            WifiNetwork("ExampleNet", security="secured", hidden=True),
            "not-logged",
        )

    assert setup._last_connection_error == "not_found"


def test_password_is_visible_across_both_lcd_rows():
    frames = []
    setup = IRWifiSetup(display=lambda *lines: frames.append(lines), config={"stt": {}})

    setup._render_password("VisiblePassword12345")

    assert frames == [("VisiblePassword1", "2345")]


def test_configured_fallback_raises_saved_connection_by_name():
    setup = IRWifiSetup(
        display=lambda *_args: None,
        config={
            "stt": {},
            "wifi_setup": {
                "fallback_connection": "Marina_5GHz",
                "fallback_connection_uuid": "fallback-uuid",
            },
        },
    )
    completed = __import__("subprocess").CompletedProcess([], 0, "", "")

    with patch.object(setup, "_run_nmcli", return_value=completed) as run_nmcli, \
            patch.object(setup, "is_connected_to", side_effect=[False, True]):
        assert setup.restore_fallback()

    assert run_nmcli.call_args_list[-1].args[0] == [
        "--wait", "30", "connection", "up", "uuid", "fallback-uuid",
        "ifname", "wlan0",
    ]


def test_fallback_does_not_bounce_an_already_active_connection():
    setup = IRWifiSetup(
        display=lambda *_args: None,
        config={"stt": {}, "wifi_setup": {"fallback_connection": "Marina_5GHz"}},
    )

    with patch.object(setup, "is_connected_to", return_value=True), \
            patch.object(setup, "_run_nmcli") as run_nmcli:
        assert setup.restore_fallback()

    run_nmcli.assert_not_called()


def test_nmcli_operations_use_passwordless_sudo_for_boot_service():
    completed = __import__("subprocess").CompletedProcess([], 0, "", "")

    with patch("core.wifi_setup.subprocess.run", return_value=completed) as run:
        IRWifiSetup._run_nmcli(["general", "status"])

    assert run.call_args.args[0] == ["sudo", "-n", "nmcli", "general", "status"]
