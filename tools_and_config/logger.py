"""
Console → file logging for KikiFast.

When `logging.enabled` is true in config.json, everything printed to
stdout/stderr (by ANY module or thread) is also appended to the log file with
a timestamp per line. Simple size-based rotation keeps the SD card safe.

    "logging": {
      "enabled": true,
      "file": "/srv/kikifast/logs/kiki.log",
      "max_bytes": 5000000,
      "debug": true          // verbose background prompts/responses and stream stats
    }

debug(tag, msg) is a helper for verbose diagnostics that should only appear
when logging.debug is on.
"""

import os
import sys
import threading
from datetime import datetime

_cfg = {}
_enabled = False
_debug = False
_lock = threading.Lock()


class _Tee:
    """Wraps a console stream; mirrors every write to the log file."""

    def __init__(self, stream, file_path, max_bytes):
        self._stream = stream
        self._file_path = file_path
        self._max_bytes = max_bytes
        self._fh = open(file_path, "a", buffering=1)
        self._at_line_start = True

    def write(self, data):
        try:
            self._stream.write(data)
        except Exception:
            pass
        try:
            with _lock:
                out = []
                for ch in data:
                    if self._at_line_start and ch != "\n":
                        out.append(datetime.now().strftime("%H:%M:%S.%f")[:-3] + " ")
                        self._at_line_start = False
                    out.append(ch)
                    if ch == "\n":
                        self._at_line_start = True
                self._fh.write("".join(out))
                self._rotate_if_needed()
        except Exception:
            pass

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._fh.flush()
        except Exception:
            pass

    def _rotate_if_needed(self):
        try:
            if self._fh.tell() > self._max_bytes:
                self._fh.close()
                old = self._file_path + ".1"
                if os.path.exists(old):
                    os.remove(old)
                os.replace(self._file_path, old)
                self._fh = open(self._file_path, "a", buffering=1)
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def setup_logging(full_config: dict):
    """Install the stdout/stderr tee. Call once, early in main.py."""
    global _cfg, _enabled, _debug
    _cfg = full_config.get("logging", {})
    _enabled = _cfg.get("enabled", False)
    _debug = _cfg.get("debug", False)
    if not _enabled:
        return
    file_path = _cfg.get("file", "/srv/kikifast/logs/kiki.log")
    max_bytes = _cfg.get("max_bytes", 5_000_000)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sys.stdout = _Tee(sys.stdout, file_path, max_bytes)
    sys.stderr = _Tee(sys.stderr, file_path, max_bytes)
    print(f"[Logger] Logging to {file_path} (debug={_debug})")


def debug_enabled() -> bool:
    return _enabled and _debug


def debug(tag: str, msg: str):
    """Verbose diagnostics — printed (and therefore logged) only when
    logging.enabled AND logging.debug are true."""
    if debug_enabled():
        print(f"[{tag}:debug] {msg}")
