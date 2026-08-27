import json
import signal
import asyncio
from types import SimpleNamespace

import pytest

from core import media_manager as media


class FakeProcess:
    next_pid = 9000

    def __init__(self, command):
        self.command = command
        self.returncode = None
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if timeout is not None and self.returncode is None:
            self.returncode = -signal.SIGTERM
        return self.returncode

    def terminate(self):
        self.returncode = -signal.SIGTERM

    def kill(self):
        self.returncode = -signal.SIGKILL


def test_ytdlp_search_playlist_wrapper_is_unwrapped(tmp_path, monkeypatch):
    instance = media.MusicManager(str(tmp_path / "liked.json"))
    actual_video = {
        "id": "dQw4w9WgXcQ",
        "title": "Rick Astley - Never Gonna Give You Up",
        "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "url": "https://googlevideo.invalid/signed-audio",
        "requested_downloads": [{
            "url": "https://googlevideo.invalid/signed-audio",
        }],
    }
    wrapper = {
        "_type": "playlist",
        "id": "never gonna give you up",
        "title": "never gonna give you up",
        "webpage_url": "ytsearch1:never gonna give you up",
        "entries": [actual_video],
    }
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(wrapper), stderr=""
        ),
    )

    resolved, error = instance._resolve(
        "never gonna give you up",
        {"yt_dlp_path": "/fake/yt-dlp", "search_timeout_seconds": 20},
    )

    assert error == ""
    assert resolved == {
        "title": "Rick Astley - Never Gonna Give You Up",
        "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "video_id": "dQw4w9WgXcQ",
        "stream_url": "https://googlevideo.invalid/signed-audio",
    }


def test_ytdlp_empty_search_has_clear_error(tmp_path, monkeypatch):
    instance = media.MusicManager(str(tmp_path / "liked.json"))
    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"_type": "playlist", "entries": [None]}),
            stderr="",
        ),
    )

    resolved, error = instance._resolve("no result", {})

    assert resolved is None
    assert error == "YouTube search returned no videos"


def test_play_music_tool_returns_exact_permanent_youtube_link(monkeypatch):
    from tools_and_config import tools

    expected_url = "https://www.youtube.com/watch?v=8WVXk0Gz66E"
    monkeypatch.setattr(tools, "is_output_muted", lambda: False)
    monkeypatch.setattr(media.music_manager, "configure", lambda _config: None)
    monkeypatch.setattr(
        media.music_manager,
        "play_query",
        lambda *_args: (
            True,
            "10 Hours of Relaxing Music - Calm Piano & Guitar, Sleep Music, Study Music.",
        ),
    )
    monkeypatch.setattr(
        media.music_manager,
        "snapshot",
        lambda: {
            "current": {
                "title": "10 Hours of Relaxing Music",
                "webpage_url": expected_url,
                "video_id": "8WVXk0Gz66E",
            },
        },
    )

    result = asyncio.run(tools.play_music("soft instrumental music"))

    assert result == (
        "Now playing 10 Hours of Relaxing Music - Calm Piano & Guitar, "
        "Sleep Music, Study Music. - "
        "https://www.youtube.com/watch?v=8WVXk0Gz66E"
    )


@pytest.fixture
def manager(tmp_path, monkeypatch):
    instance = media.MusicManager(str(tmp_path / "liked.json"))
    processes = []

    class DormantThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

    def fake_resolve(query, _config):
        video_id = query.rsplit("=", 1)[-1] if query.startswith("http") else query.replace(" ", "_")
        return {
            "title": f"Title {video_id}",
            "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
            "video_id": video_id,
            "stream_url": f"https://stream.invalid/{video_id}",
        }, ""

    def fake_popen(command, **_kwargs):
        proc = FakeProcess(command)
        processes.append(proc)
        return proc

    monkeypatch.setattr(instance, "_resolve", fake_resolve)
    monkeypatch.setattr(media.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(media.threading, "Thread", DormantThread)
    monkeypatch.setattr(media.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(instance, "_oled_on", lambda _title: None)
    monkeypatch.setattr(instance, "_oled_off", lambda: None)
    return instance, processes


def test_like_keeps_exact_current_youtube_video_and_persists(manager):
    instance, _ = manager
    ok, title = instance.play_query("blue monday", {"player_command": "mpv"})

    assert ok is True
    assert title == "Title blue_monday"
    assert "Added Title blue_monday" in instance.like_current()
    assert "already in your liked songs" in instance.like_current()

    stored = json.loads(instance.library_path.read_text(encoding="utf-8"))
    assert stored["liked_songs"] == [{
        "title": "Title blue_monday",
        "webpage_url": "https://www.youtube.com/watch?v=blue_monday",
        "video_id": "blue_monday",
    }]


def test_liked_songs_are_a_queue_with_next_and_previous(manager):
    instance, processes = manager
    for query in ("one", "two"):
        assert instance.play_query(query, {"player_command": "mpv"})[0]
        assert "Added" in instance.like_current()

    ok, first = instance.play_liked({"player_command": "mpv"})
    assert ok is True
    assert first == "Title one"
    assert instance.snapshot()["queue_size"] == 2

    assert instance.control("next", {"player_command": "mpv"}) == "Now playing Title two."
    assert instance.snapshot()["queue_index"] == 1
    assert instance.control("previous", {"player_command": "mpv"}) == "Now playing Title one."
    assert instance.snapshot()["queue_index"] == 0
    assert len(processes) >= 5


def test_pause_resume_controls_the_current_process(manager, monkeypatch):
    instance, _ = manager
    signals = []
    monkeypatch.setattr(media.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    assert instance.play_query("one", {"player_command": "mpv"})[0]
    pid = instance._process.pid

    assert instance.control("pause", {}) == "Paused Title one."
    assert instance.snapshot()["paused"] is True
    assert instance.control("resume", {}) == "Resumed Title one."
    assert signals == [(pid, signal.SIGSTOP), (pid, signal.SIGCONT)]


def test_stopped_gesture_during_lookup_never_starts_mpv(manager):
    instance, processes = manager

    ok, reason = instance.play_query(
        "one", {"player_command": "mpv"}, should_cancel=lambda: True
    )

    assert ok is False
    assert reason == "stopped by hand gesture"
    assert processes == []


@pytest.mark.parametrize(("value", "seconds"), [
    (5, 5),
    ("5 seconds", 5),
    ("five seconds", 5),
    ("two minutes", 120),
    ("1.5 hours", 5400),
])
def test_timer_duration_parser_accepts_model_and_voice_shapes(value, seconds):
    assert media.parse_duration_seconds(value) == seconds


def test_timer_uses_real_alarm_path_and_no_shell(tmp_path, monkeypatch):
    sound = tmp_path / "timer.mp3"
    sound.write_bytes(b"alarm")
    scheduled = {}
    launched = []

    class FakeTimer:
        def __init__(self, duration, callback):
            scheduled["duration"] = duration
            scheduled["callback"] = callback
            self.daemon = False

        def start(self):
            scheduled["started"] = True

    monkeypatch.setattr(media.threading, "Timer", FakeTimer)
    monkeypatch.setattr(media.shutil, "which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(
        media.subprocess,
        "Popen",
        lambda command, **kwargs: launched.append((command, kwargs)) or FakeProcess(command),
    )

    manager = media.TimerManager()
    assert manager.set_timer("five seconds", {
        "sound_effect_path": str(sound),
        "player_command": "mpv",
    }) == 5
    assert scheduled == {
        "duration": 5,
        "callback": scheduled["callback"],
        "started": True,
    }

    scheduled["callback"]()
    command, kwargs = launched[0]
    assert command == ["/usr/bin/mpv", "--no-video", "--really-quiet", str(sound)]
    assert "shell" not in kwargs
