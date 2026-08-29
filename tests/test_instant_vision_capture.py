"""Bounded clean-snapshot contracts for live/care vision."""

import base64

import cv2
import numpy as np

from core.vision import instant_vision


class _Response:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None


def _jpeg(pattern=False):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    if pattern:
        frame[:, ::2] = 255
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def test_capture_pulls_clean_snapshots_and_selects_a_decodable_frame(monkeypatch):
    seen = []

    def get(url, timeout):
        seen.append((url, timeout))
        return _Response(_jpeg(pattern=len(seen) == 2))

    monkeypatch.setattr(instant_vision.requests, "get", get)
    result = instant_vision.capture_best_frame_b64({
        "capture_frames": 3,
        "jpeg_quality": 90,
        "clean_snapshot_url": "http://camera/clean",
        "snapshot_fallback_url": "http://camera/fallback",
        "snapshot_timeout_seconds": 0.4,
    })

    assert len(seen) == 3
    assert {row[0] for row in seen} == {"http://camera/clean"}
    assert all(row[1] == (0.4, 0.4) for row in seen)
    assert base64.b64decode(result).startswith(b"\xff\xd8")


def test_capture_uses_one_bounded_fallback_snapshot(monkeypatch):
    seen = []

    def get(url, timeout):
        seen.append(url)
        if url.endswith("clean"):
            raise TimeoutError("camera restarting")
        return _Response(_jpeg(pattern=True))

    monkeypatch.setattr(instant_vision.requests, "get", get)
    result = instant_vision.capture_best_frame_b64({
        "capture_frames": 4,
        "clean_snapshot_url": "http://camera/clean",
        "snapshot_fallback_url": "http://camera/fallback",
        "snapshot_timeout_seconds": 0.2,
    })

    assert seen == ["http://camera/clean"] * 4 + ["http://camera/fallback"]
    assert result
