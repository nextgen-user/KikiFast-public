"""Stateful music playback and reliable countdown alarms for Kiki.

The speaking-tool wrappers live in ``tools_and_config/tools.py``.  This module
owns the state that must survive between separate tool calls: the exact
YouTube video currently playing, liked songs, playlist position, history, and
the active mpv subprocess.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.audio_output import ensure_bluetooth_sink, playback_environment


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LIBRARY_PATH = _PROJECT_ROOT / "liked_songs.json"
_DEFAULT_ALARM_PATH = _PROJECT_ROOT / "sound_effects" / "soundeffects" / "timer.mp3"

# yt-dlp enables only deno by default and deprecates YouTube extraction with no
# JavaScript runtime at all. Without one, signature deciphering silently fails
# and every googlevideo stream URL it hands back answers 403, so mpv exits at
# once and music "just doesn't play".
_JS_RUNTIMES = ("node", "quickjs", "bun")


@functools.lru_cache(maxsize=4)
def _yt_dlp_js_flags(binary: str) -> Tuple[str, ...]:
    """Flags naming every installed JS runtime this yt-dlp build accepts."""
    try:
        help_text = subprocess.run(
            [binary, "--help"], capture_output=True, text=True,
            timeout=20, check=False,
        ).stdout
    except Exception:
        return ()
    if "--js-runtimes" not in help_text:
        return ()  # older yt-dlp: the option does not exist and would abort
    flags: List[str] = []
    for runtime in _JS_RUNTIMES:
        if shutil.which(runtime):
            flags.extend(["--js-runtimes", runtime])
    return tuple(flags)


def _playback_environment() -> Dict[str, str]:
    """Environment pinning a player process to Kiki's Bluetooth speaker.

    PulseAudio leaves ``auto_null`` as the default sink after the A2DP sink
    drops and returns, so an unpinned mpv plays into the dummy output while
    speech -- which pins its own sink in ``core/tts.py`` -- still comes out of
    the speaker. That looks exactly like a song that refuses to play.

    ``reconnect=False`` on purpose: activating the A2DP profile repairs the
    ordinary auto_null case in milliseconds, while a full BlueZ reconnect would
    stall "play some music" for the length of a `bluetoothctl connect` timeout
    whenever the speaker is switched off. Reconnecting belongs to boot and to
    ``core/tts.py``, which re-checks on every utterance. An empty sink returns
    an unpinned environment so music still plays rather than failing closed.
    """
    try:
        from tools_and_config.config_loader import get_full_config
        settings = get_full_config().get("bluetooth_speaker", {}) or {}
        sink = ensure_bluetooth_sink(
            settings, reconnect=False, timeout_seconds=0.0)
    except Exception as exc:
        print(f"[Music] Could not confirm the Bluetooth sink: {exc}")
        sink = ""
    return playback_environment(sink)


def _atomic_json_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


class MusicManager:
    """One mpv player with persistent exact-video likes and playlist controls."""

    def __init__(self, library_path: Optional[str] = None):
        self.library_path = Path(library_path or _DEFAULT_LIBRARY_PATH)
        self._lock = threading.RLock()
        self._transition_lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._generation = 0
        self._queue: List[dict] = []
        self._index = -1
        self._auto_advance = False
        self._paused = False
        self._current: Optional[dict] = None
        self._liked: List[dict] = []
        self._history: List[dict] = []
        self._load()

    def configure(self, config: dict) -> None:
        configured = config.get("liked_songs_path")
        if not configured:
            return
        path = Path(configured)
        with self._lock:
            if path == self.library_path:
                return
            self.library_path = path
            self._load_locked()

    def _load(self) -> None:
        with self._lock:
            self._load_locked()

    def _load_locked(self) -> None:
        try:
            raw = json.loads(self.library_path.read_text(encoding="utf-8"))
            self._liked = [x for x in raw.get("liked_songs", []) if self._valid_entry(x)]
            self._history = [x for x in raw.get("history", []) if self._valid_entry(x)][-100:]
        except FileNotFoundError:
            self._liked, self._history = [], []
        except Exception as exc:
            print(f"[Music] Could not read {self.library_path}: {exc}")
            self._liked, self._history = [], []

    @staticmethod
    def _valid_entry(entry: Any) -> bool:
        return (
            isinstance(entry, dict)
            and isinstance(entry.get("title"), str)
            and isinstance(entry.get("webpage_url"), str)
            and entry["webpage_url"].startswith(("http://", "https://"))
        )

    def _save_locked(self) -> None:
        _atomic_json_write(
            self.library_path,
            {"liked_songs": self._liked, "history": self._history[-100:]},
        )

    @staticmethod
    def _yt_dlp(config: dict) -> str:
        configured = config.get("yt_dlp_path", "yt-dlp")
        if os.path.isfile(configured) and os.access(configured, os.X_OK):
            return configured
        return shutil.which("yt-dlp") or configured

    def _resolve(self, query_or_url: str, config: dict) -> Tuple[Optional[dict], str]:
        target = query_or_url if query_or_url.startswith(("http://", "https://")) else f"ytsearch1:{query_or_url}"
        timeout = max(5.0, min(float(config.get("search_timeout_seconds", 20)), 25.0))
        binary = self._yt_dlp(config)
        command = [
            binary,
            *_yt_dlp_js_flags(binary),
            "--dump-single-json",
            "--no-playlist",
            "-f",
            "ba/b",
            target,
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return None, "YouTube lookup timed out"
        except Exception as exc:
            return None, str(exc)

        if result.returncode != 0:
            errors = (result.stderr or "").strip().splitlines()
            return None, errors[-1] if errors else f"yt-dlp exited with {result.returncode}"
        try:
            info = json.loads(result.stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            return None, f"yt-dlp returned invalid metadata: {exc}"

        # yt-dlp represents ``ytsearch1:query`` as a one-entry playlist even
        # with --no-playlist.  The permanent YouTube URL and signed audio URL
        # live on entries[0], while the wrapper has no playable ``url`` at all.
        # Direct YouTube URLs already return the video object at top level.
        entries = info.get("entries") if isinstance(info, dict) else None
        if isinstance(entries, list):
            video = next((item for item in entries if isinstance(item, dict)), None)
            if video is None:
                return None, "YouTube search returned no videos"
            info = video

        stream_url = info.get("url")
        if not stream_url:
            downloads = info.get("requested_downloads") or []
            if downloads and isinstance(downloads[0], dict):
                stream_url = downloads[0].get("url")
        webpage_url = info.get("webpage_url") or info.get("original_url")
        if not isinstance(stream_url, str) or not stream_url.startswith("http"):
            return None, "YouTube did not return a playable audio URL"
        if not isinstance(webpage_url, str) or not webpage_url.startswith("http"):
            return None, "YouTube did not return the video's permanent URL"

        return {
            "title": str(info.get("title") or query_or_url).strip(),
            "webpage_url": webpage_url,
            "video_id": str(info.get("id") or ""),
            "stream_url": stream_url,
        }, ""

    @staticmethod
    def _oled_on(title: str) -> None:
        try:
            from core.oled_display import oled_manager
            oled_manager.music_on(title)
        except Exception:
            pass

    @staticmethod
    def _oled_off() -> None:
        try:
            from core.oled_display import oled_manager
            oled_manager.music_off()
        except Exception:
            pass

    @staticmethod
    def _player_failure(player: str, return_code: int, errors) -> str:
        """Explain an instant player exit using the player's own last words."""
        detail = ""
        try:
            errors.seek(0)
            lines = [line.strip() for line in errors.read().splitlines() if line.strip()]
            if lines:
                detail = f": {lines[-1][:160]}"
        except Exception:
            pass
        finally:
            try:
                errors.close()
            except Exception:
                pass
        return f"{player} exited immediately (code {return_code}){detail}"

    def _stop_locked(self) -> None:
        proc = self._process
        self._process = None
        self._paused = False
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _play_index(self, queue: List[dict], index: int, auto_advance: bool,
                    config: dict, should_cancel: Optional[Callable[[], bool]] = None
                    ) -> Tuple[bool, str]:
        if not queue or index < 0 or index >= len(queue):
            return False, "That playlist position does not exist."

        with self._transition_lock:
            entry = dict(queue[index])
            if isinstance(entry.get("stream_url"), str):
                resolved, error = entry, ""
            else:
                resolved, error = self._resolve(entry["webpage_url"], config)
                if resolved is None:
                    return False, error
            stream_url = resolved["stream_url"]
            entry.update({k: v for k, v in resolved.items() if k != "stream_url"})
            entry.pop("stream_url", None)  # signed stream URLs expire; never persist them
            if should_cancel and should_cancel():
                return False, "stopped by hand gesture"

            player = config.get("player_command", "mpv")
            player_path = shutil.which(player) or player
            # Keep the player's own diagnostics: an expired or 403 stream URL
            # kills mpv within a second, and the reason is the only thing that
            # tells the difference from a dead speaker.
            errors = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
            try:
                proc = subprocess.Popen(
                    [player_path, "--no-video", "--really-quiet", stream_url],
                    stdout=subprocess.DEVNULL,
                    stderr=errors,
                    start_new_session=True,
                    env=_playback_environment(),
                )
                time.sleep(0.6)
                if should_cancel and should_cancel():
                    proc.kill()
                    return False, "stopped by hand gesture"
                if proc.poll() is not None:
                    return False, self._player_failure(player, proc.returncode, errors)
            except Exception as exc:
                return False, str(exc)

            with self._lock:
                self._generation += 1
                generation = self._generation
                self._stop_locked()
                self._process = proc
                self._queue = [dict(item) for item in queue]
                self._queue[index] = dict(entry)
                self._index = index
                self._auto_advance = auto_advance
                self._paused = False
                self._current = dict(entry)
                self._history.append(dict(entry))
                self._history = self._history[-100:]
                try:
                    self._save_locked()
                except Exception as exc:
                    print(f"[Music] Could not persist playback history: {exc}")

            self._oled_on(entry["title"])
            threading.Thread(
                target=self._watch_process,
                args=(proc, generation, config),
                daemon=True,
                name="music-end-watcher",
            ).start()
            print(f"Playing music: {entry['title']} ({entry['webpage_url']})")
            return True, entry["title"]

    def _watch_process(self, proc: subprocess.Popen, generation: int, config: dict) -> None:
        return_code = proc.wait()
        next_index = None
        queue: List[dict] = []
        with self._lock:
            if generation != self._generation or proc is not self._process:
                return
            self._process = None
            self._paused = False
            self._current = None
            if return_code == 0 and self._auto_advance and self._index + 1 < len(self._queue):
                next_index = self._index + 1
                queue = [dict(item) for item in self._queue]
        self._oled_off()
        if next_index is not None:
            ok, error = self._play_index(queue, next_index, True, config)
            if not ok:
                print(f"[Music] Playlist could not advance: {error}")

    def play_query(self, song: str, config: dict,
                   should_cancel: Optional[Callable[[], bool]] = None) -> Tuple[bool, str]:
        query = (song or "").strip()
        if not query:
            return False, "Please say which song to play."
        resolved, error = self._resolve(query, config)
        if resolved is None:
            return False, error
        if should_cancel and should_cancel():
            return False, "stopped by hand gesture"
        return self._play_index([resolved], 0, False, config, should_cancel)

    def like_current(self) -> str:
        with self._lock:
            if self._current is None:
                return "No song is currently playing, so there is nothing to like."
            entry = dict(self._current)
            identity = entry.get("video_id") or entry["webpage_url"]
            if any((x.get("video_id") or x["webpage_url"]) == identity for x in self._liked):
                return f"{entry['title']} is already in your liked songs."
            self._liked.append(entry)
            self._save_locked()
            return f"Added {entry['title']} to your liked songs."

    def play_liked(self, config: dict,
                   should_cancel: Optional[Callable[[], bool]] = None) -> Tuple[bool, str]:
        with self._lock:
            queue = [dict(entry) for entry in self._liked]
        if not queue:
            return False, "Your liked songs playlist is empty."
        return self._play_index(queue, 0, True, config, should_cancel)

    def play_last(self, config: dict,
                  should_cancel: Optional[Callable[[], bool]] = None) -> Tuple[bool, str]:
        with self._lock:
            if not self._history:
                return False, "There is no previously played song yet."
            entry = dict(self._history[-1])
        return self._play_index([entry], 0, False, config, should_cancel)

    def control(self, action: str, config: dict) -> str:
        normalized = (action or "").strip().lower().replace(" ", "_")
        if normalized in {"play", "resume", "unpause"}:
            with self._lock:
                if self._process is None or self._process.poll() is not None:
                    return "No song is paused right now."
                if not self._paused:
                    return f"{self._current['title']} is already playing."
                os.kill(self._process.pid, signal.SIGCONT)
                self._paused = False
                return f"Resumed {self._current['title']}."

        if normalized == "pause":
            with self._lock:
                if self._process is None or self._process.poll() is not None:
                    return "No song is playing right now."
                if self._paused:
                    return f"{self._current['title']} is already paused."
                os.kill(self._process.pid, signal.SIGSTOP)
                self._paused = True
                return f"Paused {self._current['title']}."

        if normalized in {"next", "next_song", "previous", "previous_song", "prev"}:
            with self._lock:
                queue = [dict(entry) for entry in self._queue]
                index = self._index
                auto = self._auto_advance
            delta = 1 if normalized in {"next", "next_song"} else -1
            target = index + delta
            if target < 0:
                return "This is already the first song in the playlist."
            if target >= len(queue):
                return "This is already the last song in the playlist."
            ok, result = self._play_index(queue, target, auto, config)
            return f"Now playing {result}." if ok else f"Could not change songs: {result}"

        return "Music action must be pause, resume, next, or previous."

    def snapshot(self) -> dict:
        """Small read-only state view used by focused tests and diagnostics."""
        with self._lock:
            return {
                "current": dict(self._current) if self._current else None,
                "liked_songs": [dict(x) for x in self._liked],
                "history": [dict(x) for x in self._history],
                "queue_index": self._index,
                "queue_size": len(self._queue),
                "paused": self._paused,
            }


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60,
}


def parse_duration_seconds(duration: Any) -> int:
    """Accept schema-correct numbers plus common small-model string mistakes."""
    if isinstance(duration, bool):
        raise ValueError("Timer duration must be a positive number.")
    if isinstance(duration, (int, float)):
        seconds = float(duration)
    elif isinstance(duration, str):
        text = duration.strip().lower().replace("-", " ")
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)?", text)
        if match:
            value = float(match.group(1))
            unit = match.group(2) or "seconds"
        else:
            unit_match = re.fullmatch(r"([a-z ]+)\s+(seconds?|secs?|minutes?|mins?|hours?|hrs?)", text)
            if not unit_match:
                raise ValueError("Use a duration such as 30 seconds or 5 minutes.")
            words = [w for w in unit_match.group(1).split() if w != "and"]
            if not words or any(word not in _NUMBER_WORDS for word in words):
                raise ValueError("Use a duration such as 30 seconds or 5 minutes.")
            value = float(sum(_NUMBER_WORDS[word] for word in words))
            unit = unit_match.group(2)
        multiplier = 3600 if unit.startswith(("h", "hour")) else 60 if unit.startswith(("m", "min")) else 1
        seconds = value * multiplier
    else:
        raise ValueError("Timer duration must be a positive number.")

    rounded = int(round(seconds))
    if rounded < 1:
        raise ValueError("Timer duration must be at least one second.")
    if rounded > 7 * 24 * 60 * 60:
        raise ValueError("Timer duration cannot be longer than seven days.")
    return rounded


class TimerManager:
    """In-process countdown timers that launch a validated alarm command."""

    def __init__(self):
        self._lock = threading.Lock()
        self._next_id = 0
        self._timers: Dict[int, threading.Timer] = {}

    @staticmethod
    def resolve_sound_path(configured: Optional[str]) -> Path:
        path = Path(configured) if configured else _DEFAULT_ALARM_PATH
        if path.is_file():
            return path
        # Recover from the repository's old, easy-to-miss `soundeffects` typo.
        fallback = _DEFAULT_ALARM_PATH
        if fallback.is_file():
            print(f"[Timer] Configured alarm does not exist: {path}; using {fallback}")
            return fallback
        raise FileNotFoundError(f"Timer alarm sound not found: {path}")

    def set_timer(self, duration: Any, config: dict) -> int:
        seconds = parse_duration_seconds(duration)
        sound_path = self.resolve_sound_path(config.get("sound_effect_path"))
        player = config.get("player_command", "mpv")
        player_path = shutil.which(player)
        if not player_path:
            raise FileNotFoundError(f"Timer audio player not found: {player}")

        with self._lock:
            self._next_id += 1
            timer_id = self._next_id

        def fire() -> None:
            try:
                command = (
                    [player_path, "--no-video", "--really-quiet", str(sound_path)]
                    if Path(player_path).name == "mpv"
                    else [player_path, "-q", str(sound_path)]
                )
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    env=_playback_environment(),
                )
                print(f"[Timer] Alarm {timer_id} firing via PID {proc.pid}")
            except Exception as exc:
                print(f"[Timer] Alarm {timer_id} playback failed: {exc}")
            finally:
                with self._lock:
                    self._timers.pop(timer_id, None)

        timer = threading.Timer(seconds, fire)
        timer.daemon = True
        with self._lock:
            self._timers[timer_id] = timer
        timer.start()
        print(f"[Timer] Set alarm {timer_id} for {seconds} seconds")
        return seconds


music_manager = MusicManager()
timer_manager = TimerManager()
