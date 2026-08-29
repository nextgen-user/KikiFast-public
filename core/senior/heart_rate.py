"""Thread-safe MAX30102 measurement controller for Kiki's care agent.

The controller emits structured sensor states only. It deliberately contains no
spoken prompts: the complex care agent owns every word of the interaction.
"""

import json
import threading
import time
from typing import Any, Callable, Dict, Optional

from max30102_read import capture_heart_rate, prepare_heart_rate

PREPARATION_TTL_SECONDS = 180.0


class HeartRateController:
    def __init__(self, prepare_fn: Callable = prepare_heart_rate,
                 capture_fn: Callable = capture_heart_rate,
                 progress_fn: Optional[Callable[[Dict[str, Any]], None]] = None):
        self._prepare_fn = prepare_fn
        self._capture_fn = capture_fn
        self._progress_fn = progress_fn or self._display_progress
        self._operation_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._cancel = threading.Event()
        self._preparation: Optional[Dict[str, Any]] = None
        self._prepared_at = 0.0

    @staticmethod
    def _display_progress(event: Dict[str, Any]) -> None:
        try:
            from core.lcd_display import lcd_manager
            phase = str(event.get("phase", "measuring"))
            if phase == "capturing":
                detail = (f"{int(event.get('elapsed_seconds', 0))}/"
                          f"{int(event.get('duration_seconds', 0))} sec")
            else:
                detail = phase.replace("_", " ")[:16]
            lcd_manager.update_status("Heart rate", detail)
        except Exception:
            pass

    def prepare(self, site: str = "finger") -> Dict[str, Any]:
        if not self._operation_lock.acquire(blocking=False):
            return {"status": "busy", "reason": "measurement_already_running"}
        try:
            self._cancel.clear()
            result = self._prepare_fn(
                site=site, cancel_event=self._cancel, progress=self._progress_fn)
            with self._state_lock:
                self._preparation = (
                    json.loads(json.dumps(result))
                    if result.get("status") == "ready_for_contact" else None)
                self._prepared_at = time.monotonic() if self._preparation else 0.0
            return result
        finally:
            self._operation_lock.release()

    def capture(self, seconds: Optional[float] = None) -> Dict[str, Any]:
        if not self._operation_lock.acquire(blocking=False):
            return {"status": "busy", "reason": "measurement_already_running"}
        try:
            self._cancel.clear()
            with self._state_lock:
                if (self._preparation and
                        time.monotonic() - self._prepared_at > PREPARATION_TTL_SECONDS):
                    self._preparation = None
                    self._prepared_at = 0.0
                preparation = json.loads(json.dumps(self._preparation)) \
                    if self._preparation else None
            if preparation is None:
                return {"status": "not_prepared", "reason": "run_prepare_first"}
            result = self._capture_fn(
                preparation, seconds=seconds, cancel_event=self._cancel,
                progress=self._progress_fn)
            # A completed/failed capture consumes its baseline. Retrying starts
            # from a fresh ambient reading rather than trusting stale light data.
            with self._state_lock:
                self._preparation = None
                self._prepared_at = 0.0
            return result
        finally:
            self._operation_lock.release()

    def cancel(self) -> Dict[str, Any]:
        self._cancel.set()
        with self._state_lock:
            self._preparation = None
            self._prepared_at = 0.0
        return {"status": "cancel_requested"}

    def state(self) -> Dict[str, Any]:
        with self._state_lock:
            if (self._preparation and
                    time.monotonic() - self._prepared_at > PREPARATION_TTL_SECONDS):
                self._preparation = None
                self._prepared_at = 0.0
            prepared = bool(self._preparation)
            site = (self._preparation or {}).get("site")
        return {
            "status": "busy" if self._operation_lock.locked()
            else ("ready_for_contact" if prepared else "idle"),
            "site": site,
        }


_controller: Optional[HeartRateController] = None
_controller_lock = threading.Lock()


def get_heart_rate_controller() -> HeartRateController:
    global _controller
    with _controller_lock:
        if _controller is None:
            _controller = HeartRateController()
        return _controller
