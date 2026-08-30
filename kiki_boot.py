#!/usr/bin/env python3
"""Kiki boot orchestrator.

Runs at rpi boot (kikifast.service) and from the settings-menu Restart:

  1. If config.json "auto_start" is false (and --force not given) -> exit, start NOTHING.
  2. Find and connect the configured Bluetooth speaker, then select its PulseAudio sink.
  3. Offer the LCD + IR startup-config wizard (Wi-Fi, speaking brain, volume,
     language, mode, and whether to re-run the LCD/audio sync calibration).
  4. Ensure Wi-Fi is connected. If not, use the LCD + IR sensors to select a network
     and dictate its password with the Pi-local whisper.cpp model.
  5. Loop the boot sound; show live status on the 16x2 LCD + OLED boot animation.
  6. Start the local WhatsApp bridge asynchronously (best effort; never delays boot).
  7. Ask the laptop server-manager (http://127.0.0.1:8079) to (re)start the three
     inference servers: llama-server :8080, tts-server :8082, whisper-server :5555.
     The manager always terminates existing instances first, per spec.
  8. (Re)start the local Hailo face/webcam server (hailo_follower.service).
  9. Wait until ALL FOUR are live (3 laptop servers healthy + hailo ZMQ ports up).
 10. If the wizard asked for it, stop the boot sound and run the LCD/audio sync
     calibration (needs TTS + Whisper live, and the speaker/mic to itself).
 11. Stop the boot sound and exec main.py in the kiki venv (same PID, same service).

Any failure: stop boot sound, play error.mp3, show the error on the LCD, exit 1.

Usage:
  kiki_boot.py            # boot path (honours auto_start)
  kiki_boot.py --force    # settings-menu restart / manual: ignore auto_start
"""

import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parent
KIKIFAST = str(Path(os.environ.get("KIKIFAST_HOME", _DEFAULT_ROOT)).expanduser().resolve())
VENV_PY = os.environ.get("KIKIFAST_PYTHON", sys.executable)
CONFIG = os.environ.get(
    "KIKIFAST_CONFIG", os.path.join(KIKIFAST, "tools_and_config", "config.json")
)
CONFIG_TEMPLATE = os.path.join(KIKIFAST, "tools_and_config", "config.example.json")
SFX = os.path.join(KIKIFAST, "sound_effects", "soundeffects")
BOOT_MP3 = os.path.join(SFX, "color-440-sounds-nr-norm-24860.mp3")
ERROR_MP3 = os.path.join(SFX, "error.mp3")

LAPTOP = os.environ.get("KIKIFAST_SERVER_MANAGER_URL", "http://127.0.0.1:8079").rstrip("/")
LAPTOP_REACH_TIMEOUT = 180   # laptop may still be booting up
ALL_LIVE_TIMEOUT = 360       # llama model load dominates
HAILO_PORTS = (5555, 5557)   # ZMQ cmd REP + motor REP, bound by hailo_follower

# Audio/session defaults required for mpv/aplay access under systemd.
os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
os.environ.setdefault("PULSE_SERVER", "unix:/run/user/1000/pulse/native")
os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")
os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("XAUTHORITY", os.path.expanduser("~/.Xauthority"))

os.chdir(KIKIFAST)
sys.path.insert(0, KIKIFAST)

from core.audio_output import ensure_bluetooth_sink


def log(msg):
    print(f"[boot {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _config_read_path():
    """Use checked-in defaults until the deployment creates config.json."""
    return CONFIG if os.path.exists(CONFIG) else CONFIG_TEMPLATE


# ---------------------------------------------------------------- displays --
lcd = None
oled = None
try:
    from core.lcd_display import lcd_manager as lcd
except Exception as e:  # display is best-effort; never block boot on it
    log(f"LCD unavailable: {e}")
try:
    from core.oled_display import oled_manager as oled
except Exception as e:
    log(f"OLED unavailable: {e}")


def show(line1, line2=""):
    log(f"{line1} | {line2}")
    if lcd:
        try:
            lcd.write(line1, line2)
        except Exception:
            pass
    if oled:
        try:
            oled.set_state("boot")
            oled.push_status(f"{line1} {line2}".strip())
        except Exception:
            pass


def show_password(line1, line2=""):
    """Show literal password characters on the LCD without putting them in logs."""
    log("WiFi password visible on LCD")
    if lcd:
        try:
            lcd.write_raw(line1, line2)
        except Exception:
            pass


# ------------------------------------------------------------------- sound --
_snd = None


def boot_sound_start():
    global _snd
    if os.path.exists(BOOT_MP3):
        try:
            _snd = subprocess.Popen(
                ["mpv", "--no-video", "--really-quiet", "--loop=inf", BOOT_MP3],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            log(f"boot sound failed: {e}")


def boot_sound_stop():
    global _snd
    if _snd and _snd.poll() is None:
        _snd.terminate()
        try:
            _snd.wait(timeout=2)
        except Exception:
            _snd.kill()
    _snd = None


def play_error_sound():
    if os.path.exists(ERROR_MP3):
        subprocess.run(["play", "-q", ERROR_MP3], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# -------------------------------------------------------------- bluetooth --
_DEVICE_RE = re.compile(
    r"^\s*Device\s+([0-9A-Fa-f:]{17})\s+(.+?)\s*$", re.MULTILINE)


def _run_quiet(command, timeout=8):
    """Run a small boot command without allowing it to abort the orchestrator."""
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
        log(f"{command[0]} failed: {e}")
        return None


def _bluetooth_devices(paired_only=False):
    """Return [(MAC, normalized name)] from a bluetoothctl device listing."""
    if paired_only:
        # bluetoothctl renamed paired-devices to "devices Paired"; support both
        # versions because Raspberry Pi OS images may carry either BlueZ CLI.
        commands = (
            ["bluetoothctl", "devices", "Paired"],
            ["bluetoothctl", "paired-devices"],
        )
    else:
        commands = (["bluetoothctl", "devices"],)
    for command in commands:
        result = _run_quiet(command)
        if not result or result.returncode != 0:
            continue
        devices = [
            (mac.upper(), " ".join(name.casefold().split()))
            for mac, name in _DEVICE_RE.findall(result.stdout)
        ]
        if devices:
            return devices
    return []


def _match_device_name(devices, target_name):
    """Best MAC for target_name: exact, then substring, then word overlap."""
    wanted = " ".join(str(target_name).casefold().split())
    if not wanted or not devices:
        return None
    exact = next((mac for mac, name in devices if name == wanted), None)
    if exact:
        return exact
    partial = next(
        (mac for mac, name in devices if wanted in name or name in wanted),
        None,
    )
    if partial:
        return partial
    # A vendor can ship the same speaker under a slightly different broadcast
    # name than the one in config.json ("Example Speaker" vs "Example Speaker Pro").
    # The distinctive words still overlap, so fall back to the best overlap.
    wanted_words = set(wanted.split())
    best_mac, best_score = None, 0
    for mac, name in devices:
        score = len(wanted_words & set(name.split()))
        if score > best_score:
            best_mac, best_score = mac, score
    if best_mac:
        log(f"matched Bluetooth speaker {target_name!r} loosely to {best_mac}")
    return best_mac


def _find_paired_bluetooth_device(target_name):
    """Return the MAC for a paired device whose name matches target_name."""
    return _match_device_name(_bluetooth_devices(paired_only=True), target_name)


def _bluetooth_info_says(mac, field):
    """True when `bluetoothctl info` reports "<field>: yes" for the device."""
    result = _run_quiet(["bluetoothctl", "info", mac])
    return bool(
        result
        and result.returncode == 0
        and re.search(rf"^\s*{field}:\s*yes\s*$", result.stdout, re.MULTILINE)
    )


def _bluetooth_is_connected(mac):
    return _bluetooth_info_says(mac, "Connected")


# BlueZ only reports these when nothing answered the page at all, i.e. the
# speaker is switched off or out of range. Every other failure means something
# did answer, so the pairing is the suspect and may be rebuilt.
_ABSENT_MARKERS = (
    "page-timeout",
    "not available",
    "unreachable",
    "host is down",
    "no route",
)


def _bluetooth_is_present(mac):
    """True when the controller can currently see the device.

    A bonded BR/EDR speaker stops answering inquiry once it is paired, so RSSI
    alone would report every working speaker as absent. The connection attempt
    itself is the reliable probe, and this only backstops it.
    """
    if _bluetooth_is_connected(mac):
        return True
    result = _run_quiet(["bluetoothctl", "info", mac])
    return bool(
        result
        and result.returncode == 0
        and re.search(r"^\s*RSSI:", result.stdout, re.MULTILINE)
    )


def _bluetooth_connect(mac):
    """Attempt one connection. Returns (connected, the speaker answered)."""
    result = _run_quiet(["bluetoothctl", "connect", mac], timeout=12)
    output = ""
    if result is not None:
        output = f"{result.stdout}\n{result.stderr}".casefold()
    connected = _bluetooth_is_connected(mac)
    # A speaker with a stale bond still completes the ACL ("Connected: yes")
    # and then refuses the audio profile -- br-connection-refused. That is a
    # switched-on speaker, and exactly the state re-pairing repairs.
    answered = connected or (
        "failed to connect" in output
        and not any(marker in output for marker in _ABSENT_MARKERS)
    )
    if not connected and output.strip():
        reason = next(
            (line.strip() for line in reversed(output.splitlines())
             if "failed to connect" in line),
            "",
        )
        if reason:
            log(f"connect refused by {mac}: {reason}")
    return connected, answered


def _select_bluetooth_audio_sink(mac, timeout_seconds=10):
    """Wait for PulseAudio to expose the device and make its sink the default."""
    token = mac.replace(":", "_").casefold()
    deadline = time.monotonic() + max(0, timeout_seconds)
    while time.monotonic() <= deadline:
        # Also handles the state where BlueZ is connected and a card exists,
        # but Pulse/PipeWire has not activated its A2DP profile (the desktop
        # otherwise shows only auto_null / Dummy Output).
        repaired = ensure_bluetooth_sink(
            {"enabled": True, "mac": mac},
            reconnect=False,
            timeout_seconds=0,
        )
        if repaired:
            log(f"PulseAudio default sink set to {repaired}")
            return True
        result = _run_quiet(["pactl", "list", "short", "sinks"], timeout=3)
        if result and result.returncode == 0:
            sink_names = []
            for line in result.stdout.splitlines():
                fields = line.split()
                if len(fields) >= 2:
                    sink_names.append(fields[1])
            sink = next(
                (name for name in sink_names
                 if token in name.casefold() and "a2dp" in name.casefold()),
                None,
            )
            if sink:
                selected = _run_quiet(
                    ["pactl", "set-default-sink", sink], timeout=3)
                if selected and selected.returncode == 0:
                    log(f"PulseAudio default sink set to {sink}")
                    return True
        time.sleep(1)
    log("Bluetooth connected, but its A2DP sink did not appear in time")
    return False


def _connect_bluetooth_loop(name, mac, deadline, retry_seconds):
    """Connect and route the speaker until deadline.

    Returns (ok, mac, answered) where ``answered`` records whether the speaker
    ever responded, so a failure can be told apart from a powered-off speaker.
    """
    answered = False
    while time.monotonic() < deadline:
        if not mac:
            mac = _find_paired_bluetooth_device(name)
        if not mac:
            log(f"paired Bluetooth speaker not found: {name}")
        elif _bluetooth_is_connected(mac):
            answered = True
            log(f"Bluetooth speaker already connected: {name} ({mac})")
            if _select_bluetooth_audio_sink(
                    mac, min(10, max(0, deadline - time.monotonic()))):
                return True, mac, answered
            # BlueZ can say Connected while Pulse/PipeWire has not created an
            # A2DP sink. Reconnect once instead of reporting false success and
            # letting the rest of Kiki start on auto_null (Dummy Output).
            log("Bluetooth is connected but has no selectable A2DP sink; reconnecting")
            _run_quiet(["bluetoothctl", "disconnect", mac], timeout=8)
            connected, _ = _bluetooth_connect(mac)
            if connected and _select_bluetooth_audio_sink(
                    mac, min(10, max(0, deadline - time.monotonic()))):
                return True, mac, answered
        else:
            log(f"connecting Bluetooth speaker: {name} ({mac})")
            connected, replied = _bluetooth_connect(mac)
            answered = answered or replied
            if connected:
                log(f"Bluetooth speaker connected: {name} ({mac})")
                if _select_bluetooth_audio_sink(
                        mac, min(10, max(0, deadline - time.monotonic()))):
                    return True, mac, answered
                log("Bluetooth connected, but A2DP routing is not ready")
            # If the configured address became stale after re-pairing, fall
            # back to the paired device name for the next attempt.
            paired_mac = _find_paired_bluetooth_device(name)
            if paired_mac and paired_mac != mac:
                log(f"found {name} at a new Bluetooth address: {paired_mac}")
                mac = paired_mac
        time.sleep(min(retry_seconds, max(0, deadline - time.monotonic())))
    return False, mac, answered


def _bluetooth_scan(seconds):
    """Run one inquiry so BlueZ (re)discovers nearby devices."""
    _run_quiet(["bluetoothctl", "--timeout", str(int(seconds)), "scan", "on"],
               timeout=seconds + 15)


def _repair_bluetooth_pairing(name, mac, settings, answered=False):
    """Unpair the speaker and pair it from scratch. Returns its MAC or None.

    BlueZ can hold a bond the speaker itself has forgotten -- after a factory
    reset, or after it was paired to a phone. The ACL then still completes
    ("Connected: yes") while the speaker refuses the audio profile with
    br-connection-refused, so no A2DP sink is ever created and every plain
    `connect` retries forever. Deleting the link key and pairing again is the
    only way out.
    """
    scan_seconds = max(4.0, float(settings.get("repair_scan_seconds", 12)))

    # Prove the speaker is really there before deleting its pairing: a
    # switched-off speaker must keep the bond, because rebuilding one needs
    # somebody standing next to it holding the pairing button.
    if not answered:
        log(f"re-pairing {name}: scanning {scan_seconds:g}s for the speaker")
        _bluetooth_scan(scan_seconds)
    known = _bluetooth_devices()
    target = mac if mac and any(m == mac for m, _ in known) else None
    if not target:
        target = _match_device_name(known, name)
    if not target:
        log(f"re-pair skipped: {name} was not found; leaving its pairing intact")
        return None
    if not answered and not _bluetooth_is_present(target):
        log(f"re-pair skipped: {name} never answered; leaving its pairing intact")
        return None

    stale = [old for old in (mac, target) if old]
    named = _match_device_name(_bluetooth_devices(paired_only=True), name)
    if named and named not in stale:
        stale.append(named)
    for old in dict.fromkeys(stale):
        log(f"removing stale Bluetooth pairing: {old}")
        _run_quiet(["bluetoothctl", "disconnect", old], timeout=8)
        _run_quiet(["bluetoothctl", "remove", old], timeout=8)

    # `remove` drops the device from BlueZ entirely -- `pair` answers "not
    # available" until a fresh inquiry finds it again.
    log(f"re-pairing {name}: scanning {scan_seconds:g}s for the speaker")
    _bluetooth_scan(scan_seconds)
    found = _bluetooth_devices()
    if not any(m == target for m, _ in found):
        rediscovered = _match_device_name(found, name)
        if not rediscovered:
            log(f"re-pair failed: {name} did not reappear after unpairing "
                f"(hold its pairing button, then reboot)")
            return None
        target = rediscovered

    log(f"pairing Bluetooth speaker: {name} ({target})")
    _run_quiet(["bluetoothctl", "pair", target], timeout=40)
    _run_quiet(["bluetoothctl", "trust", target], timeout=8)
    if not _bluetooth_info_says(target, "Paired"):
        log(f"re-pair failed: {name} ({target}) did not pair "
            f"(put the speaker in pairing mode, then reboot)")
        return None
    _bluetooth_connect(target)
    log(f"re-paired Bluetooth speaker: {name} ({target})")
    return target


def _persist_speaker_mac(mac):
    """Save a re-paired speaker's address so the next boot finds it first."""
    try:
        with open(_config_read_path(), encoding="utf-8") as f:
            saved = json.load(f)
        saved.setdefault("bluetooth_speaker", {})["mac"] = mac
        temp_path = CONFIG + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(saved, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, CONFIG)
        log(f"saved the speaker's new Bluetooth address to config.json: {mac}")
    except Exception as e:
        log(f"could not save the new Bluetooth address: {e}")


def connect_bluetooth_speaker(settings):
    """Connect the configured paired speaker before the audible boot sequence."""
    if not settings.get("enabled", True):
        log("Bluetooth speaker startup connection disabled")
        return True

    name = str(settings.get("name", "Example Speaker")).strip()
    configured_mac = str(settings.get("mac", "")).strip().upper()
    mac = configured_mac if re.fullmatch(
        r"[0-9A-F]{2}(?::[0-9A-F]{2}){5}", configured_mac) else None
    timeout_seconds = max(1, float(settings.get("connect_timeout_seconds", 30)))
    retry_seconds = max(0.25, float(settings.get("retry_interval_seconds", 2)))

    powered = _run_quiet(["bluetoothctl", "power", "on"])
    if not powered or powered.returncode != 0:
        log("could not power on the Bluetooth controller")
        return False

    ok, mac, answered = _connect_bluetooth_loop(
        name, mac, time.monotonic() + timeout_seconds, retry_seconds)

    # Every ordinary reconnect failed. Before giving up, treat the bond itself
    # as the suspect: unpair and pair again, then retry the same connect loop.
    if not ok and settings.get("repair_pairing", True):
        repaired = _repair_bluetooth_pairing(name, mac, settings, answered)
        if repaired:
            mac = repaired
            ok, mac, _ = _connect_bluetooth_loop(
                name, mac, time.monotonic() + min(timeout_seconds, 20.0),
                retry_seconds)

    if ok:
        if mac and mac != configured_mac:
            settings["mac"] = mac
            _persist_speaker_mac(mac)
        return True

    log(f"Bluetooth speaker unavailable after {timeout_seconds:g}s: {name}")
    return False


# ------------------------------------------------------------------ checks --
def http_json(url, method="GET", timeout=5):
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tcp_alive(host, port, timeout=1.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def hailo_alive():
    return all(tcp_alive("127.0.0.1", p) for p in HAILO_PORTS)


def fail(reason):
    log(f"BOOT FAILED: {reason}")
    boot_sound_stop()
    show("BOOT FAILED", reason[:16])
    if oled:
        try:
            oled.push_status(f"BOOT FAILED: {reason}")
        except Exception:
            pass
    play_error_sound()
    sys.exit(1)


# -------------------------------------------------------------------- main --
def main():
    force = "--force" in sys.argv

    try:
        with open(_config_read_path(), encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        fail(f"config: {e}")

    if not cfg.get("auto_start", True) and not force:
        log("auto_start is false in config.json -> starting nothing.")
        show("Auto start off", "sleeping...")
        return

    # Connect audio before Wi-Fi setup or the audible boot sequence. The
    # configured MAC avoids ambiguity; name discovery keeps this working if
    # the speaker is re-paired and receives a new address.
    speaker_cfg = cfg.get("bluetooth_speaker", {})
    speaker_name = str(speaker_cfg.get("name", "Example Speaker"))
    show("Audio setup...", speaker_name[:16])
    speaker_ready = connect_bluetooth_speaker(speaker_cfg)
    if not speaker_ready:
        if speaker_cfg.get("required", False):
            fail("speaker offline")
        log("continuing boot without the Bluetooth speaker")

    # Restore a volume previously confirmed in the startup wizard. The setting
    # is optional, so existing installations retain their current sink volume
    # until the user visits the new volume step once.
    try:
        from core.startup_config import apply_configured_volume
        if speaker_ready:
            apply_configured_volume(cfg)
    except Exception as e:
        log(f"saved volume skipped: {e}")

    # Offer a short settings-style wizard on every startup. It runs after audio
    # setup so its IR feedback beep is audible, and before the normal Wi-Fi gate
    # so a requested network change can happen first.
    calibrate_sync = False
    try:
        from core.startup_config import offer_startup_config
        cfg, calibrate_sync = offer_startup_config(
            cfg, CONFIG, show, password_display=show_password
        )
    except Exception as e:
        # Configuration is optional; missing display/GPIO never blocks boot.
        log(f"startup config skipped: {e}")

    # Wi-Fi must exist before the Tailscale laptop services (including Whisper)
    # are reachable. Password dictation therefore uses the local whisper.cpp
    # binary/model and releases the mic + GPIO lines before normal startup.
    try:
        from core.wifi_setup import ensure_wifi_connected
        if not ensure_wifi_connected(cfg, show, password_display=show_password):
            fail("wifi not connected")
    except SystemExit:
        raise
    except Exception as e:
        fail(f"wifi: {e}")

    # Start the Go WhatsApp bridge while the much slower laptop model is loading.
    # This is deliberately best-effort and fire-and-forget: Go compilation,
    # WhatsApp reconnect/authentication, or failure can never extend Kiki's boot
    # or time-to-first-word. main.py starts/owns the corresponding MCP stdio
    # server later and also has a duplicate-safe bridge fallback.
    try:
        from core.self_extend.whatsapp_mcp import launch_whatsapp_bridge_background
        launch_whatsapp_bridge_background(cfg.get("whatsapp", {}))
    except Exception as e:
        log(f"WhatsApp bridge startup skipped: {e}")

    boot_sound_start()
    show("Kiki starting...", "laptop servers")

    # 1) Tell the laptop to (re)start llama/tts/whisper. Retry while the
    #    laptop itself may still be booting.
    deadline = time.time() + LAPTOP_REACH_TIMEOUT
    triggered = False
    while time.time() < deadline:
        try:
            http_json(f"{LAPTOP}/restart", method="POST")
            triggered = True
            break
        except Exception:
            show("Kiki starting...", "wait laptop...")
            time.sleep(5)
    if not triggered:
        fail("laptop offline")
    log("laptop restart triggered")

    # 2) (Re)start the local Hailo face/webcam server.
    show("Kiki starting...", "hailo webcam")
    r = subprocess.run(["sudo", "-n", "systemctl", "restart", "hailo_follower.service"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        fail(f"hailo: {r.stderr.strip()[:40] or 'restart failed'}")

    # 3) Wait for ALL FOUR services.
    deadline = time.time() + ALL_LIVE_TIMEOUT
    last_pending = None
    while time.time() < deadline:
        pending = []
        try:
            st = http_json(f"{LAPTOP}/status")
            if st.get("state") == "error":
                fail(f"laptop: {st.get('detail', '')[:32]}")
            for name, svc in st.get("services", {}).items():
                if not svc.get("healthy"):
                    pending.append(name)
        except Exception:
            pending.append("laptop")
        if not hailo_alive():
            pending.append("hailo")

        if not pending:
            break
        if pending != last_pending:
            show("Waiting for:", ",".join(pending)[:16])
            last_pending = pending
        time.sleep(3)
    else:
        fail("timeout: " + ",".join(last_pending or ["?"]))

    # 4) Optional LCD/audio sync calibration, requested in the startup wizard.
    #    It needs the TTS + Whisper servers (only live now) and the speaker and
    #    microphone to itself, so the looping boot sound stops first. main.py is
    #    not running yet, so nothing else holds the mic.
    if calibrate_sync:
        boot_sound_stop()
        log("running LCD/audio sync calibration")
        try:
            from core.startup_config import run_lcd_sync_calibration
            run_lcd_sync_calibration(cfg, show, python_executable=VENV_PY)
        except Exception as e:
            # Calibration is a convenience; a failure must never block boot.
            log(f"calibration skipped: {e}")

    # 5) All live -> hand over to main.py (exec: same PID, same systemd unit).
    log("all 4 services live, starting main.py")
    show("All systems go", "starting Kiki")
    boot_sound_stop()
    os.execv(VENV_PY, [VENV_PY, os.path.join(KIKIFAST, "main.py")])


if __name__ == "__main__":
    main()
