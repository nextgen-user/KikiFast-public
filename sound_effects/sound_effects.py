"""
Thinking sound effects player for KikiFast voice assistant.
Plays a single thinking sound while the LLM is generating a response.
"""

import os
import random
import subprocess
import threading
from tools_and_config.config_loader import get_sfx_config


class ThinkingSoundPlayer:
    """
    Plays a single thinking/waiting sound on a background thread.
    A sound is played once and does not loop.
    """

    def __init__(self):
        cfg = get_sfx_config()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sfx_dir = os.path.join(base_dir, cfg.get("directory", "soundeffects"))
        fillers_dir = os.path.join(sfx_dir, cfg.get("fillers_directory", "fillers"))

        self._files = []
        if os.path.exists(fillers_dir) and os.path.isdir(fillers_dir):
            self._files = [
                os.path.join(fillers_dir, f)
                for f in os.listdir(fillers_dir)
                if f.lower().endswith(".wav")
            ]

        if not self._files:
            print("[SFX] Warning: No thinking sound files found!")

        self._stop_event = threading.Event()
        self._thread = None
        self._process = None
        self._lock = threading.Lock()

    def start(self):
        """Start playing thinking sounds in a loop."""
        if self._thread and self._thread.is_alive():
            return  # Already playing

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()
        print("[SFX] 🎵 Thinking sounds started")

    def stop(self):
        """Stop playing thinking sounds."""
        self._stop_event.set()

        # Hot path: this runs the instant Kiki's first real audio plays.
        # SIGKILL, not terminate — mpv's graceful shutdown drains ALSA and took
        # 300-500ms, delaying the caller AND letting the filler bleed over the
        # first spoken words. Reaping happens off-thread.
        with self._lock:
            if self._process and self._process.poll() is None:
                try:
                    self._process.kill()
                except Exception:
                    pass
                proc = self._process
                threading.Thread(target=proc.wait, daemon=True).start()

        thread = self._thread
        self._thread = None
        if thread and thread.is_alive():
            threading.Thread(target=lambda: thread.join(timeout=2), daemon=True).start()

        print("[SFX] 🔇 Thinking sounds stopped")

    def _play_loop(self):
        """Plays a single random filler sound and exits."""
        if not self._files:
            return

        sound_file = random.choice(self._files)

        try:
            with self._lock:
                if self._stop_event.is_set():
                    return
                self._process = subprocess.Popen(
                    ["mpv", "--no-video", "--audio-device=alsa", sound_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            # Wait for playback to finish or stop signal
            while self._process.poll() is None:
                if self._stop_event.is_set():
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                    return
                self._stop_event.wait(timeout=0.1)

        except Exception as e:
            print(f"[SFX] Playback error: {e}")

    @property
    def is_playing(self):
        return self._thread is not None and self._thread.is_alive()


if __name__ == "__main__":
    import time
    print("=== Sound Effects Test ===")
    player = ThinkingSoundPlayer()
    player.start()
    time.sleep(5)
    player.stop()
    print("Done.")
