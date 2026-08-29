"""Bounded snapshot contracts for live/care vision.

Two things are under test: that capture stays bounded (no MJPEG backlog, one
fallback attempt), and that it refuses a frame the camera server admits is
stale. The second one is not hypothetical — a wedged Hailo pipeline answered
200 with the same frame for thirteen minutes, and the care agent praised a
person's exercise form while they were out of the room.
"""

import base64

import cv2
import numpy as np

from core.vision import instant_vision


class _Response:
    def __init__(self, content, status_code=200, headers=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
        return None


def _jpeg(pattern=False):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    if pattern:
        frame[:, ::2] = 255
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return encoded.tobytes()


def test_capture_pulls_raw_snapshots_and_selects_a_decodable_frame(monkeypatch):
    seen = []

    def get(url, timeout):
        seen.append((url, timeout))
        return _Response(_jpeg(pattern=len(seen) == 2))

    monkeypatch.setattr(instant_vision.requests, "get", get)
    result = instant_vision.capture_best_frame_b64({
        "capture_frames": 3,
        "jpeg_quality": 90,
        "raw_snapshot_url": "http://camera/raw",
        "clean_snapshot_url": "http://camera/clean",
        "snapshot_fallback_url": "http://camera/fallback",
        "snapshot_timeout_seconds": 0.4,
    })

    # The raw tap satisfies the request on its own; nothing falls through to
    # the slower annotated sources.
    assert len(seen) == 3
    assert {row[0] for row in seen} == {"http://camera/raw"}
    assert all(row[1] == (0.4, 0.4) for row in seen)
    assert base64.b64decode(result).startswith(b"\xff\xd8")


def test_capture_uses_one_bounded_fallback_snapshot(monkeypatch):
    seen = []

    def get(url, timeout):
        seen.append(url)
        if url.endswith("raw") or url.endswith("clean"):
            raise TimeoutError("camera restarting")
        return _Response(_jpeg(pattern=True))

    monkeypatch.setattr(instant_vision.requests, "get", get)
    result = instant_vision.capture_best_frame_b64({
        "capture_frames": 4,
        "raw_snapshot_url": "http://camera/raw",
        "clean_snapshot_url": "http://camera/clean",
        "snapshot_fallback_url": "http://camera/fallback",
        "snapshot_timeout_seconds": 0.2,
    })

    # Each live source gets its full sample count; the annotated fallback is
    # tried exactly once so a restarting camera cannot stall a spoken turn.
    assert seen == (["http://camera/raw"] * 4
                    + ["http://camera/clean"] * 4
                    + ["http://camera/fallback"])
    assert result


def test_a_stale_frame_is_refused_rather_than_used(monkeypatch):
    """An old frame must reach the caller as "no frame", not as pixels."""
    seen = []

    def get(url, timeout):
        seen.append(url)
        # Every source reports a frame from thirteen minutes ago.
        return _Response(_jpeg(pattern=True),
                         headers={"X-Frame-Age-Ms": "800000"})

    monkeypatch.setattr(instant_vision.requests, "get", get)
    result = instant_vision.capture_best_frame_b64({
        "capture_frames": 3,
        "max_frame_age_ms": 2000,
        "raw_snapshot_url": "http://camera/raw",
        "clean_snapshot_url": "http://camera/clean",
        "snapshot_fallback_url": "http://camera/fallback",
        "snapshot_timeout_seconds": 0.2,
    })

    assert result is None
    # One look per source is enough to know it is stale — no re-sampling a
    # source that just said its frame is old.
    assert seen == ["http://camera/raw", "http://camera/clean",
                    "http://camera/fallback"]


def test_a_503_from_a_stalled_frame_server_moves_on(monkeypatch):
    """503 means "the pipeline stalled"; take it at its word and move on."""
    seen = []

    def get(url, timeout):
        seen.append(url)
        if url.endswith("raw"):
            return _Response(b'{"error":"stale_frame"}', status_code=503)
        return _Response(_jpeg(pattern=True), headers={"X-Frame-Age-Ms": "40"})

    monkeypatch.setattr(instant_vision.requests, "get", get)
    result = instant_vision.capture_best_frame_b64({
        "capture_frames": 3,
        "max_frame_age_ms": 2000,
        "raw_snapshot_url": "http://camera/raw",
        "clean_snapshot_url": "http://camera/clean",
        "snapshot_fallback_url": "http://camera/fallback",
        "snapshot_timeout_seconds": 0.2,
    })

    assert result
    # The 503'd tap is abandoned after a single request, then the fresh clean
    # frame serves the turn.
    assert seen == ["http://camera/raw"] + ["http://camera/clean"] * 3


def test_a_fresh_frame_is_still_accepted(monkeypatch):
    """The freshness check must not reject a healthy camera."""

    def get(url, timeout):
        return _Response(_jpeg(pattern=True), headers={"X-Frame-Age-Ms": "18"})

    monkeypatch.setattr(instant_vision.requests, "get", get)
    result = instant_vision.capture_best_frame_b64({
        "capture_frames": 2,
        "max_frame_age_ms": 2000,
        "raw_snapshot_url": "http://camera/raw",
        "clean_snapshot_url": "http://camera/clean",
        "snapshot_fallback_url": "http://camera/fallback",
        "snapshot_timeout_seconds": 0.2,
    })

    assert base64.b64decode(result).startswith(b"\xff\xd8")


def test_a_server_without_an_age_header_is_still_usable(monkeypatch):
    """Age reporting is an improvement, not a requirement.

    An older frame server (or the annotated Flask snapshot) sends no
    X-Frame-Age-Ms. Refusing those would leave Kiki blind against a camera
    stack that is working fine.
    """

    def get(url, timeout):
        return _Response(_jpeg(pattern=True))

    monkeypatch.setattr(instant_vision.requests, "get", get)
    result = instant_vision.capture_best_frame_b64({
        "capture_frames": 2,
        "max_frame_age_ms": 2000,
        "raw_snapshot_url": "http://camera/raw",
        "clean_snapshot_url": "http://camera/clean",
        "snapshot_fallback_url": "http://camera/fallback",
        "snapshot_timeout_seconds": 0.2,
    })

    assert result
