"""
Regression test for the ZMQ context/thread leak that exhausted the process.

Symptom: web-search tool calls (and both their timeouts) hung forever with
`Resource temporarily unavailable` (EAGAIN) spammed across the log. Root cause
was NOT search_web — it was per-operation ZMQ leaks:
  * core.hardware.controller.quick_command() (one per neck gesture) created a fresh
    zmq.asyncio.Context()+socket but only closed them on the SUCCESS path; when
    the controller was down, recv raised and the context (which owns an OS I/O
    thread + FDs) leaked.
  * KikiController.connect() (retried every 10s by the face handler) created a
    context+2 sockets and returned False on failure without closing them.
Once enough contexts leaked, the process hit its thread limit and pthread_create
failed process-wide (EAGAIN) — so ThreadPoolExecutor could no longer spawn the
workers that run search_web's timeout and main.py's tool-wait timeout. Both hung.

These tests drive the real functions with a fake zmq that always fails on recv
(controller-down) and assert every created context is terminated and every
socket closed — i.e. zero leak on the failure path.
"""

import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hardware import controller as kcc


class _Tracker:
    def __init__(self):
        self.created = 0
        self.termed = 0
        self.closed = 0


class _FakeSocket:
    def __init__(self, tracker):
        self._t = tracker

    def setsockopt(self, *a, **k):
        pass

    def setsockopt_string(self, *a, **k):
        pass

    def connect(self, *a, **k):
        pass

    async def send_json(self, *a, **k):
        pass

    async def recv_json(self, *a, **k):
        # Simulate the controller being down (RCVTIMEO / EAGAIN).
        raise RuntimeError("Resource temporarily unavailable")

    def close(self, *a, **k):
        self._t.closed += 1


class _FakeContext:
    def __init__(self, tracker):
        self._t = tracker
        tracker.created += 1

    def socket(self, *a, **k):
        return _FakeSocket(self._t)

    def term(self, *a, **k):
        self._t.termed += 1


def _install_fake_zmq(monkeypatch, tracker):
    monkeypatch.setattr(kcc.zmq.asyncio, "Context",
                        lambda *a, **k: _FakeContext(tracker))


def test_quick_command_does_not_leak_context_on_failure(monkeypatch):
    tracker = _Tracker()
    _install_fake_zmq(monkeypatch, tracker)

    N = 25
    for _ in range(N):
        try:
            asyncio.run(kcc.quick_command(host="10.0.0.1", look="center"))
        except Exception:
            pass  # caller's problem; we only care that cleanup ran

    assert tracker.created == N, "fake context not exercised"
    assert tracker.termed == N, (
        f"leaked {tracker.created - tracker.termed} contexts (I/O threads) — "
        "quick_command must term() the context even when recv raises"
    )
    assert tracker.closed >= N, "sockets must be closed on the failure path too"


def test_controller_connect_does_not_leak_on_failed_retries(monkeypatch):
    tracker = _Tracker()
    _install_fake_zmq(monkeypatch, tracker)

    ctrl = kcc.KikiController(host="10.0.0.1")
    N = 25
    for _ in range(N):
        ok = asyncio.run(ctrl.connect())   # get_state fails -> returns False
        assert ok is False

    assert tracker.created == N
    assert tracker.termed == N, (
        f"leaked {tracker.created - tracker.termed} contexts across reconnect "
        "attempts — connect() must release context/sockets on failure"
    )


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
