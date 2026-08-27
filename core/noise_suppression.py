"""Real-time microphone noise suppression for Kiki's STT pipeline.

RNNoise operates on 10 ms, 48 kHz frames.  Keeping the denoiser here (at
capture time) means all enhanced audio is ready before Silero declares an
endpoint; there is no utterance-sized enhancement job on the response path.

The small ctypes wrapper intentionally does not import :mod:`pyrnnoise`.
Importing that package also imports its file/plotting stack, none of which is
needed by the live microphone path.  We only use the bundled native RNNoise
library and its stable C API.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import os
import time
from pathlib import Path

import numpy as np


RNNOISE_SAMPLE_RATE = 48_000
RNNOISE_FRAME = 480  # 10 ms at 48 kHz


class RNNoiseUnavailable(RuntimeError):
    """Raised when the native RNNoise runtime cannot be loaded."""


def _find_rnnoise_library(explicit_path: str | os.PathLike | None = None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.is_file():
            return path
        raise RNNoiseUnavailable(f"RNNoise library does not exist: {path}")

    try:
        distribution = importlib.metadata.distribution("pyrnnoise")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RNNoiseUnavailable(
            "pyrnnoise is not installed (install requirements.txt)"
        ) from exc

    # Normal wheel installation location.  The files scan is a defensive
    # fallback for installers that relocate the wheel's ``.data/purelib`` tree.
    direct = Path(distribution.locate_file("pyrnnoise/librnnoise.so"))
    if direct.is_file():
        return direct
    for installed_file in distribution.files or ():
        if installed_file.name == "librnnoise.so":
            candidate = Path(distribution.locate_file(installed_file))
            if candidate.is_file():
                return candidate
    raise RNNoiseUnavailable("pyrnnoise is installed but librnnoise.so is missing")


class RNNoiseSuppressor:
    """Stateful RNNoise processor with a strict per-frame real-time guard.

    If native processing repeatedly consumes too much of its 10 ms frame
    budget, suppression bypasses itself for the rest of the session.  The
    caller continues receiving raw audio, so an overloaded denoiser cannot
    increase speech endpoint or response latency.
    """

    def __init__(
        self,
        *,
        library_path: str | os.PathLike | None = None,
        max_process_ms: float = 5.0,
        slow_frame_limit: int = 3,
    ):
        path = _find_rnnoise_library(library_path)
        try:
            self._lib = ctypes.CDLL(str(path))
        except OSError as exc:
            raise RNNoiseUnavailable(f"could not load RNNoise: {exc}") from exc

        self._lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        self._lib.rnnoise_create.restype = ctypes.c_void_p
        self._lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        self._lib.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.rnnoise_process_frame.restype = ctypes.c_float

        self.max_process_ms = max(0.1, float(max_process_ms))
        self.slow_frame_limit = max(1, int(slow_frame_limit))
        self._slow_frames = 0
        self._state = None
        self.active = True
        self.bypass_reason = None
        self.reset()

    def reset(self):
        """Reset recurrent noise history at a mute/query boundary."""
        if self._state:
            self._lib.rnnoise_destroy(self._state)
        self._state = self._lib.rnnoise_create(None)
        if not self._state:
            raise RNNoiseUnavailable("rnnoise_create returned NULL")
        self._slow_frames = 0

    def process(self, samples_i16: np.ndarray) -> tuple[np.ndarray, float | None]:
        """Return normalized, denoised float32 audio and speech probability."""
        samples_i16 = np.asarray(samples_i16, dtype=np.int16)
        if samples_i16.size != RNNOISE_FRAME:
            raise ValueError(
                f"RNNoise needs {RNNOISE_FRAME} samples, got {samples_i16.size}"
            )
        raw = samples_i16.astype(np.float32)
        if not self.active:
            return raw / 32768.0, None

        work = raw.copy()
        ptr = work.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        started = time.perf_counter()
        try:
            speech_probability = float(
                self._lib.rnnoise_process_frame(self._state, ptr, ptr)
            )
        except Exception as exc:
            self.active = False
            self.bypass_reason = f"runtime error: {exc}"
            return raw / 32768.0, None

        process_ms = (time.perf_counter() - started) * 1000.0
        if process_ms > self.max_process_ms:
            self._slow_frames += 1
            if self._slow_frames >= self.slow_frame_limit:
                self.active = False
                self.bypass_reason = (
                    f"{self._slow_frames} frames exceeded "
                    f"{self.max_process_ms:.1f} ms"
                )
        else:
            self._slow_frames = 0

        return np.clip(work / 32768.0, -1.0, 1.0), speech_probability

    def close(self):
        if self._state:
            self._lib.rnnoise_destroy(self._state)
            self._state = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class StreamingDecimator3:
    """Anti-aliased streaming 48 kHz -> 16 kHz converter.

    RNNoise frames contain exactly 480 samples, so every call returns exactly
    160 samples.  The causal 63-tap filter has under 0.7 ms group delay and
    performs no post-utterance work and never waits for a completed utterance.
    """

    def __init__(self, taps: int = 63, cutoff_hz: float = 7_200.0):
        taps = int(taps)
        if taps < 7 or taps % 2 == 0:
            raise ValueError("taps must be an odd integer >= 7")
        if not 0 < cutoff_hz < 8_000:
            raise ValueError("cutoff_hz must be between 0 and 8000")

        midpoint = (taps - 1) / 2.0
        positions = np.arange(taps, dtype=np.float64) - midpoint
        normalized_cutoff = cutoff_hz / RNNOISE_SAMPLE_RATE
        kernel = (
            2.0
            * normalized_cutoff
            * np.sinc(2.0 * normalized_cutoff * positions)
            * np.hamming(taps)
        )
        kernel /= np.sum(kernel)
        self._kernel = kernel.astype(np.float32)
        self._history = np.zeros(taps - 1, dtype=np.float32)

    def reset(self):
        self._history.fill(0.0)

    def process(self, samples_48k: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples_48k, dtype=np.float32)
        if samples.size != RNNOISE_FRAME:
            raise ValueError(
                f"decimator needs {RNNOISE_FRAME} samples, got {samples.size}"
            )
        combined = np.concatenate((self._history, samples))
        filtered = np.convolve(combined, self._kernel, mode="valid")
        self._history = combined[-self._history.size :].copy()
        return np.ascontiguousarray(filtered[::3], dtype=np.float32)
