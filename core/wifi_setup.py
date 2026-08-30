"""Boot-time Wi-Fi provisioning for Kiki's 16x2 LCD and IR sensors.

The laptop Whisper service is unavailable before Wi-Fi exists, so this module records
the held-to-talk password locally and transcribes it with the repository's Pi-local
``whisper-cli`` and ``ggml-base.en.bin``.  It intentionally has no dependency on
``main.py`` or :mod:`core.stt`; the normal runtime owns those resources only after boot.

Controls
--------
Network list:
  * tap LEFT / RIGHT: next / previous network
  * hold either sensor: select the displayed network

Password screen:
  * hold LEFT, dictate the password character by character, then release: transcribe
    and try the password
  * tap RIGHT: backspace
  * hold RIGHT: clear the password
  * hold BOTH: return to the network list

Spoken ``clear`` and ``backspace`` edit the current password too.  ``capital k`` and
``k capital`` both produce ``K``; letters without a case instruction are lowercase.
Passwords are shown in full on the LCD, but are never printed or placed on a subprocess
command line.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

try:
    import gpiod
    from gpiod.line import Direction, Value

    GPIOD_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only off the robot
    gpiod = None
    Direction = Value = None
    GPIOD_AVAILABLE = False


CHIP_PATH = "/dev/gpiochip4"
LEFT_PIN = 22
RIGHT_PIN = 17

POLL_S = 0.03
MIN_TAP_S = 0.05
SELECT_HOLD_S = 1.0
MIN_DICTATE_S = 0.35
CLEAR_HOLD_S = 1.0
BACK_HOLD_S = 1.2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHISPER_CLI = os.environ.get("KIKIFAST_WHISPER_CLI", "whisper-cli")
WHISPER_MODEL = os.environ.get("KIKIFAST_WHISPER_MODEL", "ggml-base.en.bin")
WHISPER_PROMPT = (
    "A Wi-Fi password dictated one character at a time. "
    "Example: k i k i. Say capital before or after a capital letter. "
    "Editing commands are clear and backspace."
)


class WifiSetupError(RuntimeError):
    """Raised when boot-time Wi-Fi setup cannot operate."""


@dataclass(frozen=True)
class WifiNetwork:
    ssid: str
    signal: int = 0
    security: str = ""
    bssid: str = ""
    hidden: bool = False

    @property
    def is_open(self) -> bool:
        return self.security.strip() in ("", "--")


def _nmcli_fields(line: str) -> list[str]:
    """Split nmcli terse output while honoring its backslash escapes."""
    fields: list[str] = []
    current: list[str] = []
    escaped = False
    for char in line.rstrip("\n"):
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ":":
            fields.append("".join(current))
            current = []
        else:
            current.append(char)
    if escaped:
        current.append("\\")
    fields.append("".join(current))
    return fields


def parse_nmcli_networks(output: str) -> list[WifiNetwork]:
    """Parse and de-duplicate nmcli terse Wi-Fi scan output."""
    by_ssid: dict[str, WifiNetwork] = {}
    for line in output.splitlines():
        fields = _nmcli_fields(line)
        if len(fields) < 4:
            continue
        if len(fields) >= 5:
            _in_use, bssid, ssid, signal_text = fields[:4]
            security = ":".join(fields[4:])
        else:  # backward-compatible parser for scans/tests without BSSID
            _in_use, ssid, signal_text = fields[:3]
            bssid = ""
            security = ":".join(fields[3:])
        ssid = ssid.strip()
        if not ssid:
            continue  # hidden networks cannot be selected meaningfully on a 16x2 UI
        try:
            signal = int(signal_text)
        except ValueError:
            signal = 0
        candidate = WifiNetwork(
            ssid=ssid, signal=signal, security=security, bssid=bssid
        )
        previous = by_ssid.get(ssid)
        if previous is None or candidate.signal > previous.signal:
            by_ssid[ssid] = candidate
    return sorted(by_ssid.values(), key=lambda item: (-item.signal, item.ssid.casefold()))


_LETTER_NAMES = {
    "a": "a", "ay": "a",
    "b": "b", "bee": "b", "be": "b",
    "c": "c", "cee": "c", "sea": "c", "see": "c",
    "d": "d", "dee": "d",
    "e": "e", "ee": "e",
    "f": "f", "eff": "f",
    "g": "g", "gee": "g",
    "h": "h", "aitch": "h", "hitch": "h",
    "i": "i", "eye": "i",
    "j": "j", "jay": "j",
    "k": "k", "kay": "k",
    "l": "l", "el": "l", "ell": "l",
    "m": "m", "em": "m",
    "n": "n", "en": "n",
    "o": "o", "oh": "o",
    "p": "p", "pee": "p",
    "q": "q", "cue": "q", "queue": "q",
    "r": "r", "are": "r",
    "s": "s", "ess": "s",
    "t": "t", "tee": "t", "tea": "t",
    "u": "u", "you": "u",
    "v": "v", "vee": "v",
    "w": "w",
    "x": "x", "ex": "x",
    "y": "y", "why": "y",
    "z": "z", "zee": "z", "zed": "z",
}

_DIGIT_NAMES = {
    "zero": "0", "ohzero": "0", "one": "1", "two": "2", "too": "2",
    "three": "3", "four": "4", "for": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "ate": "8", "nine": "9",
}

_SYMBOL_NAMES = {
    "space": " ", "dash": "-", "hyphen": "-", "underscore": "_",
    "at": "@", "hash": "#", "pound": "#", "dollar": "$", "percent": "%",
    "ampersand": "&", "star": "*", "asterisk": "*", "bang": "!",
    "exclamation": "!", "question": "?", "dot": ".", "period": ".",
    "comma": ",", "colon": ":", "semicolon": ";", "slash": "/",
    "backslash": "\\", "plus": "+", "equals": "=", "equal": "=",
}

_FILLER_WORDS = {
    "password", "wifi", "wi-fi", "wi", "fi", "is", "my", "the", "and", "then",
    "letter", "letters", "please", "type", "enter", "say",
}


def apply_password_dictation(current: str, transcript: str) -> tuple[str, int, list[str]]:
    """Apply a character dictation/edit command without ever logging its result.

    Returns ``(new_password, edit_count, unknown_words)``.  Unknown alphabetic words
    are split into letters because Whisper often joins a carefully spelled sequence
    (``k i k i``) into one word (``kiki``).
    """
    normalized = re.sub(r"back\s+space", "backspace", transcript, flags=re.I)
    normalized = re.sub(r"upper\s+case", "uppercase", normalized, flags=re.I)
    normalized = re.sub(r"lower\s+case", "lowercase", normalized, flags=re.I)
    normalized = re.sub(r"double\s+u", "w", normalized, flags=re.I)
    tokens = re.findall(r"[A-Za-z]+|\d+", normalized)

    chars = list(current)
    edits = 0
    unknown: list[str] = []
    pending_case: Optional[str] = None
    repeat_next = False
    last_added_index: Optional[int] = None

    def add_character(char: str) -> None:
        nonlocal edits, pending_case, repeat_next, last_added_index
        if char.isalpha():
            char = char.upper() if pending_case == "upper" else char.lower()
        count = 2 if repeat_next else 1
        for _ in range(count):
            chars.append(char)
            last_added_index = len(chars) - 1
            edits += 1
        pending_case = None
        repeat_next = False

    index = 0
    while index < len(tokens):
        token = tokens[index].lower()
        if token in _FILLER_WORDS:
            index += 1
            continue
        if token == "clear":
            edits += len(chars) or 1
            chars.clear()
            last_added_index = None
            pending_case = None
            repeat_next = False
        elif token in ("backspace", "delete"):
            if chars:
                chars.pop()
            edits += 1
            last_added_index = None
        elif token in ("capital", "uppercase", "upper"):
            if last_added_index is not None and chars[last_added_index].isalpha():
                chars[last_added_index] = chars[last_added_index].upper()
                edits += 1
                last_added_index = None
            else:
                pending_case = "upper"
        elif token in ("lowercase", "lower"):
            if last_added_index is not None and chars[last_added_index].isalpha():
                chars[last_added_index] = chars[last_added_index].lower()
                edits += 1
                last_added_index = None
            else:
                pending_case = "lower"
        elif token == "double":
            repeat_next = True
        elif token in _LETTER_NAMES:
            add_character(_LETTER_NAMES[token])
        elif token in _DIGIT_NAMES:
            add_character(_DIGIT_NAMES[token])
        elif token in _SYMBOL_NAMES:
            add_character(_SYMBOL_NAMES[token])
        elif token.isdigit():
            for char in token:
                add_character(char)
        elif token.isalpha():
            # Whisper commonly collapses spelled letters into the resulting word.
            unknown.append(token)
            for char in token:
                add_character(char)
        index += 1

    return "".join(chars), edits, unknown


class LocalPasswordDictation:
    """Capture from Kiki's configured microphone and run offline whisper.cpp."""

    def __init__(self, device_index: Optional[int], channels: int = 1):
        self.device_index = device_index
        self.channels = channels

    def record_while(self, held: Callable[[], bool]) -> tuple[str, float]:
        try:
            import pyaudio
        except ImportError as exc:  # pragma: no cover - robot dependency
            raise WifiSetupError("PyAudio is not installed") from exc

        audio = pyaudio.PyAudio()
        kwargs = {
            "format": pyaudio.paInt16,
            "channels": self.channels,
            "rate": 16000,
            "input": True,
            "frames_per_buffer": 512,
        }
        if self.device_index is not None:
            try:
                audio.get_device_info_by_index(self.device_index)
                kwargs["input_device_index"] = self.device_index
            except Exception:
                pass

        frames: list[bytes] = []
        started = 0.0
        stream = None
        try:
            stream = audio.open(**kwargs)
            started = time.monotonic()
            while held():
                frames.append(stream.read(512, exception_on_overflow=False))
        except Exception as exc:
            raise WifiSetupError(f"microphone: {exc}") from exc
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            audio.terminate()

        duration = time.monotonic() - started if started else 0.0
        handle = tempfile.NamedTemporaryFile(prefix="kiki-wifi-", suffix=".wav", delete=False)
        wav_path = handle.name
        handle.close()
        try:
            with wave.open(wav_path, "wb") as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"".join(frames))
            return wav_path, duration
        except Exception:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
            raise

    def transcribe(self, wav_path: str) -> str:
        if not os.path.isfile(WHISPER_CLI) or not os.access(WHISPER_CLI, os.X_OK):
            raise WifiSetupError("local whisper-cli is missing")
        if not os.path.isfile(WHISPER_MODEL):
            raise WifiSetupError("local Whisper model is missing")
        try:
            result = subprocess.run(
                [
                    WHISPER_CLI, "-m", WHISPER_MODEL, "-f", wav_path,
                    "-t", "4", "-l", "en", "-nt", "-np", "-nf",
                    "--prompt", WHISPER_PROMPT,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise WifiSetupError("local transcription timed out") from exc
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
        if result.returncode != 0:
            raise WifiSetupError("local transcription failed")
        return result.stdout.strip()


class IRWifiSetup:
    """Synchronous boot UI that owns the IR GPIO lines until Wi-Fi is ready."""

    def __init__(self, display, config: dict, password_display=None, feedback=None):
        self.display = display
        self.password_display = password_display or display
        self.config = config
        self.feedback = feedback
        self.request = None
        self._force = False
        self._last_connection_error = None
        stt_config = config.get("stt", {})
        self.dictation = LocalPasswordDictation(
            device_index=stt_config.get("device_index", 2),
            channels=int(stt_config.get("channels", 1)),
        )

    def _show(self, line1: str, line2: str = "") -> None:
        self.display(line1, line2)

    def _feedback(self) -> None:
        if self.feedback is None:
            return
        try:
            self.feedback()
        except Exception as exc:
            print(f"[WiFi] IR feedback failed: {exc}", flush=True)

    @staticmethod
    def _run_nmcli(args: Iterable[str], *, input_text: Optional[str] = None,
                   timeout: float = 30) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                ["sudo", "-n", "nmcli", *args],
                input=input_text, capture_output=True, text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WifiSetupError(f"NetworkManager: {exc}") from exc

    def is_connected(self) -> bool:
        result = self._run_nmcli(["-t", "-f", "TYPE,STATE", "device", "status"], timeout=8)
        if result.returncode != 0:
            return False
        return any(
            fields[:2] == ["wifi", "connected"]
            for fields in (_nmcli_fields(line) for line in result.stdout.splitlines())
        )

    def is_connected_to(self, ssid: str) -> bool:
        """Return whether the active Wi-Fi access point has this exact SSID."""
        result = self._run_nmcli(
            ["-t", "--escape", "yes", "-f", "IN-USE,SSID",
             "device", "wifi", "list", "--rescan", "no"],
            timeout=8,
        )
        if result.returncode != 0:
            return False
        for line in result.stdout.splitlines():
            fields = _nmcli_fields(line)
            if len(fields) >= 2 and fields[0] == "*" and fields[1] == ssid:
                return True
        return False

    def restore_fallback(self) -> bool:
        """Reconnect the configured known-good profile after a setup failure."""
        wifi_config = self.config.get("wifi_setup", {})
        fallback = wifi_config.get("fallback_connection")
        if not fallback:
            return False
        if self.is_connected_to(str(fallback)):
            print("[WiFi] Fallback connection is already active", flush=True)
            return True
        self._show("WiFi recovery", str(fallback))
        self._run_nmcli(["radio", "wifi", "on"], timeout=8)
        fallback_uuid = wifi_config.get("fallback_connection_uuid")
        selector = ["uuid", str(fallback_uuid)] if fallback_uuid else ["id", str(fallback)]
        restored = False
        for _attempt in range(3):
            result = self._run_nmcli(
                ["--wait", "30", "connection", "up", *selector, "ifname", "wlan0"],
                timeout=40,
            )
            if result.returncode == 0 and self.is_connected_to(str(fallback)):
                restored = True
                break
            time.sleep(2)
        print(f"[WiFi] Fallback connection {'restored' if restored else 'failed'}", flush=True)
        return restored

    def active_wifi_connection(self) -> Optional[tuple[str, str]]:
        """Return the active Wi-Fi profile's (name, UUID), if available."""
        result = self._run_nmcli(
            ["-t", "--escape", "yes", "-f", "NAME,UUID,TYPE",
             "connection", "show", "--active"],
            timeout=8,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            fields = _nmcli_fields(line)
            if len(fields) >= 3 and fields[2] in ("802-11-wireless", "wifi"):
                return fields[0], fields[1]
        return None

    def restore_connection(self, connection: Optional[tuple[str, str]]) -> bool:
        """Reconnect the exact profile active before a forced Wi-Fi change."""
        if not connection:
            return self.is_connected()
        name, uuid = connection
        self._show("WiFi recovery", name)
        result = self._run_nmcli(
            ["--wait", "30", "connection", "up", "uuid", uuid,
             "ifname", "wlan0"],
            timeout=40,
        )
        restored = result.returncode == 0 and self.is_connected_to(name)
        print(
            f"[WiFi] Previous connection {'restored' if restored else 'failed'}: {name}",
            flush=True,
        )
        return restored

    def scan(self) -> list[WifiNetwork]:
        wifi_config = self.config.get("wifi_setup", {})
        preferred = []
        configured_preferred = wifi_config.get("preferred_ssids", [])
        if isinstance(configured_preferred, str):
            configured_preferred = [configured_preferred]
        for value in configured_preferred:
            ssid = str(value).strip()
            if ssid and ssid.casefold() not in {
                existing.casefold() for existing in preferred
            }:
                preferred.append(ssid)

        # A phone hotspot may suppress ordinary beacon discovery while idle.
        # Directed probes make NetworkManager actively ask for configured
        # preferred names before collecting the normal visible-network list.
        for ssid in preferred:
            try:
                self._run_nmcli(
                    ["device", "wifi", "rescan", "ifname", "wlan0", "ssid", ssid],
                    timeout=12,
                )
            except WifiSetupError as exc:
                # A directed probe is an enhancement; never let it prevent the
                # ordinary scan or the configured placeholder from appearing.
                print(f"[WiFi] Directed scan failed for {ssid}: {exc}", flush=True)

        result = self._run_nmcli(
            ["-t", "--escape", "yes", "-f", "IN-USE,BSSID,SSID,SIGNAL,SECURITY",
             "device", "wifi", "list", "--rescan", "yes"],
            timeout=25,
        )
        if result.returncode != 0:
            return []
        networks = parse_nmcli_networks(result.stdout)
        discovered = {network.ssid.casefold() for network in networks}
        for ssid in preferred:
            if ssid.casefold() not in discovered:
                # Keep explicitly configured mobile/hidden hotspots selectable
                # even when they do not answer a broadcast scan. Selecting the
                # entry uses NetworkManager's `hidden yes` connection path.
                networks.append(
                    WifiNetwork(
                        ssid=ssid,
                        signal=0,
                        security="secured",
                        hidden=True,
                    )
                )
        return networks

    def _read(self) -> tuple[bool, bool]:
        try:
            return (
                self.request.get_value(LEFT_PIN) == Value.INACTIVE,
                self.request.get_value(RIGHT_PIN) == Value.INACTIVE,
            )
        except Exception as exc:
            raise WifiSetupError(f"IR read: {exc}") from exc

    def _wait_clear(self) -> None:
        while True:
            left, right = self._read()
            if not left and not right:
                return
            time.sleep(POLL_S)

    def _render_network(self, networks: list[WifiNetwork], index: int) -> None:
        network = networks[index]
        lock = "hidden" if network.hidden else "open" if network.is_open else "locked"
        if network.hidden:
            heading = f"WiFi {index + 1}/{len(networks)} target"
        else:
            heading = f"WiFi {index + 1}/{len(networks)} {network.signal}%"
        self._show(heading, network.ssid)
        # The LCD shows one SSID at a time; the first line conveys signal and position.
        print(f"[WiFi] Showing network {index + 1}/{len(networks)} ({lock}, {network.signal}%)",
              flush=True)

    def _choose_network(self, networks: list[WifiNetwork]) -> Optional[WifiNetwork]:
        self._show("WiFi networks", "L next R prev")
        time.sleep(1.2)
        self._wait_clear()
        index = 0
        last_scan = time.monotonic()
        self._render_network(networks, index)

        while True:
            if time.monotonic() - last_scan >= 10:
                selected_ssid = networks[index].ssid
                refreshed = self.scan()
                last_scan = time.monotonic()
                if refreshed:
                    networks = refreshed
                    index = next(
                        (position for position, item in enumerate(networks)
                         if item.ssid == selected_ssid),
                        0,
                    )
                    self._render_network(networks, index)
            left, right = self._read()
            if left and right:
                if not self._force:
                    self._wait_clear()
                    continue
                started = time.monotonic()
                while all(self._read()):
                    if time.monotonic() - started >= BACK_HOLD_S:
                        self._feedback()
                        self._wait_clear()
                        return None
                    time.sleep(POLL_S)
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
                    return networks[index]
                if not held:
                    if elapsed >= MIN_TAP_S:
                        self._feedback()
                        index = ((index + 1) if pin == LEFT_PIN else (index - 1)) % len(networks)
                        self._render_network(networks, index)
                    break
                time.sleep(POLL_S)

    def _try_saved_or_open(self, network: WifiNetwork) -> bool:
        self._show("Connecting WiFi", network.ssid)
        if network.is_open:
            args = ["--wait", "25", "device", "wifi", "connect", network.ssid]
            if network.bssid:
                args.extend(["bssid", network.bssid])
        else:
            # Only raise a saved profile here. A passwordless `device wifi connect`
            # against an unknown secured SSID can invalidate its scan-cache entry,
            # making the immediately following password attempt say "not found".
            args = ["--wait", "25", "connection", "up", "id", network.ssid,
                    "ifname", "wlan0"]
        result = self._run_nmcli(args, timeout=30)
        return result.returncode == 0 and self.is_connected_to(network.ssid)

    def _try_password(self, network: WifiNetwork, password: str) -> bool:
        self._show("Trying password", f"{len(password)} characters")
        self._last_connection_error = None
        # --ask reads the secret from stdin, avoiding a password in argv/process listings.
        args = ["--ask", "--wait", "30", "device", "wifi", "connect", network.ssid]
        if network.bssid:
            args.extend(["bssid", network.bssid])
        if network.hidden:
            args.extend(["hidden", "yes"])
        result = self._run_nmcli(args, input_text=password + "\n", timeout=40)
        if result.returncode != 0:
            error_text = (result.stderr or result.stdout).strip()
            lowered = error_text.casefold()
            if "no network with ssid" in lowered or "network could not be found" in lowered:
                self._last_connection_error = "not_found"
            elif (
                "secrets were required" in lowered
                or "invalid password" in lowered
                or "authentication" in lowered
            ):
                self._last_connection_error = "authentication"
            else:
                self._last_connection_error = "other"
            detail = error_text.splitlines()
            if detail:
                print(f"[WiFi] NetworkManager: {detail[-1][:160]}", flush=True)
        return result.returncode == 0 and self.is_connected_to(network.ssid)

    def _render_password(self, password: str) -> None:
        if password:
            # The LCD has 32 character cells. Keep the newest characters visible
            # while the user dictates or edits a password longer than the panel.
            visible = password[-32:]
            if len(visible) <= 16:
                self.password_display("Password", visible)
            else:
                self.password_display(visible[:16], visible[16:])
        else:
            self._show("L hold to talk", "R tap backspace")

    def _password_loop(self, network: WifiNetwork) -> bool:
        password = ""
        self._show("WiFi password", "Both hold menu")
        time.sleep(1.2)
        self._wait_clear()
        self._render_password(password)
        both_since: Optional[float] = None

        while True:
            left, right = self._read()
            if left and right:
                both_since = both_since or time.monotonic()
                if time.monotonic() - both_since >= BACK_HOLD_S:
                    self._feedback()
                    self._wait_clear()
                    return False
                time.sleep(POLL_S)
                continue
            both_since = None

            if left:
                self._show("Listening...", "release to try")
                wav_path, duration = self.dictation.record_while(
                    lambda: self._read()[0]
                )
                # Play after recording so the feedback tone cannot become part
                # of the dictated password audio.
                self._feedback()
                if duration < MIN_DICTATE_S:
                    try:
                        os.unlink(wav_path)
                    except OSError:
                        pass
                    self._render_password(password)
                    continue
                self._show("Understanding...", "offline Whisper")
                transcript = self.dictation.transcribe(wav_path)
                password, edits, _unknown = apply_password_dictation(password, transcript)
                if not edits:
                    self._show("Did not hear", "hold L and retry")
                    time.sleep(1.5)
                    self._render_password(password)
                    continue
                if password and self._try_password(network, password):
                    return True
                if self._last_connection_error == "not_found":
                    self._show("Hotspot not found", network.ssid)
                    time.sleep(2)
                    return False
                if self._last_connection_error == "authentication":
                    self._show("Wrong password?", "hold L and retry")
                else:
                    self._show("Not connected", "Both hold menu")
                time.sleep(1.5)
                self._render_password(password)
                continue

            if right:
                started = time.monotonic()
                cleared = False
                while self._read()[1]:
                    if not cleared and time.monotonic() - started >= CLEAR_HOLD_S:
                        password = ""
                        cleared = True
                        self._feedback()
                        self._show("Password clear", "L hold to talk")
                    time.sleep(POLL_S)
                if not cleared and time.monotonic() - started >= MIN_TAP_S:
                    self._feedback()
                    password = password[:-1]
                self._render_password(password)
                continue

            time.sleep(POLL_S)

    def run(self, force: bool = False) -> bool:
        self._force = force
        if self.is_connected() and not force:
            return True
        if not GPIOD_AVAILABLE:
            raise WifiSetupError("gpiod is not installed")

        # Give NetworkManager a short chance to reconnect a known network on
        # normal boot. A forced run records the active profile below and restores
        # that exact UUID if the requested change is canceled or raises.
        self._run_nmcli(["radio", "wifi", "on"], timeout=8)
        if not force:
            grace = float(
                self.config.get("wifi_setup", {}).get("connection_grace_seconds", 10)
            )
            deadline = time.monotonic() + max(0, grace)
            while time.monotonic() < deadline:
                self._show("Checking WiFi...", "auto connect")
                if self.is_connected():
                    return True
                time.sleep(1)

        try:
            self.request = gpiod.request_lines(
                CHIP_PATH,
                consumer="kiki-wifi-setup",
                config={
                    LEFT_PIN: gpiod.LineSettings(direction=Direction.INPUT),
                    RIGHT_PIN: gpiod.LineSettings(direction=Direction.INPUT),
                },
            )
        except Exception as exc:
            raise WifiSetupError(f"IR sensors unavailable: {exc}") from exc

        previous_connection = self.active_wifi_connection() if force else None
        try:
            while True:
                self._show("Scanning WiFi...", "please wait")
                networks = self.scan()
                if not networks:
                    self._show("No WiFi found", "rescanning...")
                    time.sleep(4)
                    continue
                network = self._choose_network(networks)
                if network is None:
                    return self.restore_connection(previous_connection)
                if self._try_saved_or_open(network):
                    return True
                if not network.is_open and self._password_loop(network):
                    return True
                # Open-network failure or BOTH-hold from password entry returns here.
        except Exception:
            if force:
                try:
                    self.restore_connection(previous_connection)
                except Exception as restore_exc:
                    print(
                        f"[WiFi] Could not restore previous connection: {restore_exc}",
                        flush=True,
                    )
            raise
        finally:
            if self.request is not None:
                try:
                    self.request.release()
                except Exception:
                    pass
                self.request = None
        return self.is_connected()


def ensure_wifi_connected(config: dict, display, password_display=None) -> bool:
    """Ensure a Wi-Fi device is connected, invoking the IR/LCD setup if needed."""
    if not config.get("wifi_setup", {}).get("enabled", True):
        return True
    setup = IRWifiSetup(
        display=display, config=config, password_display=password_display
    )
    try:
        return setup.run()
    except Exception:
        try:
            setup.restore_fallback()
        except Exception:
            pass
        raise


def _manual_main() -> int:
    """Run only the Wi-Fi selector; intended for an NM checkpoint wrapper."""
    import json

    config_path = os.path.join(ROOT, "tools_and_config", "config.json")
    with open(config_path) as config_file:
        config = json.load(config_file)

    from kiki_boot import show, show_password

    setup = IRWifiSetup(display=show, config=config, password_display=show_password)
    try:
        return 0 if setup.run(force=True) else 1
    except KeyboardInterrupt:
        setup.restore_fallback()
        return 130
    except Exception as exc:
        print(f"[WiFi] Manual setup failed: {exc}", flush=True)
        setup.restore_fallback()
        return 1


if __name__ == "__main__":
    sys.exit(_manual_main())
