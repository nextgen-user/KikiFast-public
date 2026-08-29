"""The ephemeral `CARE NOW` line: what Kiki should know before the next turn.

One short system row assembled from the environment provider, the care plan's
schedule, and the trusted vitals history. It is the health-companion equivalent
of the time anchor -- and it is governed by the same hard constraint, for the
same reason (Codestructure section 4): **every character lives in the warm KV
prefix and is re-prefilled on every subsequent turn.**

That constraint drives the whole design:

  * **Silence is the default.** Only noteworthy facts earn a place. An ordinary
    afternoon with clean air, nothing due, and no readings produces "" and
    nothing is injected at all.
  * **Injected only on CHANGE, never on a timer.** Re-stating an unchanged AQI
    every five minutes would both nag the person and grow the prompt without
    bound. A row goes in when the rendered line differs from the last one, and
    not before its cooldown expires.
  * **Append-only.** The row is appended and never rewritten or removed, so the
    warm prefix stays a valid byte-prefix of the next request. "Ephemeral" here
    means "not repeated", not "retracted" -- retracting it would invalidate the
    cache and cost a full reprefill on the next voice turn.

The bands underneath are deterministic (`core/health/environment.py`), so the
AI chooses the wording while code decides whether there is anything to say.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Only these count as worth spending prefix characters on.
_NOTEWORTHY_HEAT = ("caution", "high", "very high", "extreme")
_NOTEWORTHY_AQI = ("moderate", "poor", "very poor", "severe")

# How soon a due item is worth mentioning. Beyond this it is not "now".
_UPCOMING_MINUTES = 90

# Minimum gap between two injections, even when the line genuinely changed.
# Without it a reading oscillating across a band edge would inject on every
# turn, which is exactly the nagging the deterministic bands exist to prevent.
_DEFAULT_COOLDOWN_SECONDS = 900

_MAX_LINE_CHARS = 320


def _environment_part(provider) -> str:
    try:
        return provider.compact_line() if provider else ""
    except Exception:
        return ""


def _next_due_part(manager, now: Optional[datetime] = None) -> str:
    """The single next care item, when it is close enough to matter."""
    if manager is None:
        return ""
    try:
        receipt = manager.schedule_receipt()
    except Exception:
        return ""
    if not receipt.get("scheduled"):
        return ""
    now = now or datetime.now()
    horizon = now + timedelta(minutes=_UPCOMING_MINUTES)

    soonest, soonest_at = None, None
    for worker in receipt.get("workers", []):
        raw = worker.get("next_trigger_at")
        if not raw:
            continue
        try:
            when = datetime.fromisoformat(str(raw))
        except (TypeError, ValueError):
            continue
        if when < now or when > horizon:
            continue
        if soonest_at is None or when < soonest_at:
            soonest, soonest_at = worker, when
    if soonest_at is None:
        return ""

    # `senior:routine_event:evt123` -> the title is not in the worker name, so
    # the name's middle segment is the most specific thing available here
    # without a second care-plan read on the speaking path.
    name = str(soonest.get("worker_name", ""))
    label = name.split(":")[1].replace("_", " ") if ":" in name else "care item"
    minutes = max(1, round((soonest_at - now).total_seconds() / 60))
    return f"NEXT {label} in {minutes}min"


def _vitals_part(plan) -> str:
    """The most recent trusted reading, only while it is still current."""
    if plan is None:
        return ""
    try:
        readings = plan.get_section("health_measurements") or []
    except Exception:
        return ""
    if not readings:
        return ""
    latest = readings[-1]
    try:
        measured = datetime.fromisoformat(str(latest.get("measured_at")))
    except (TypeError, ValueError):
        return ""
    age_hours = (datetime.now(measured.tzinfo) - measured).total_seconds() / 3600
    if age_hours > 12:
        # An older reading belongs to a trend question, not to "right now".
        return ""
    return (f"LAST {latest.get('measurement', 'reading')} "
            f"{latest.get('value')}{latest.get('unit', '')} "
            f"{round(age_hours)}h ago")


def _session_part(plan) -> str:
    if plan is None:
        return ""
    try:
        state = plan.care_session_state()
    except Exception:
        return ""
    if state.get("status") != "active":
        return ""
    return f"IN SESSION: {state.get('event_title', 'care routine')}"


def build_care_now(provider=None, manager=None, plan=None,
                   now: Optional[datetime] = None) -> str:
    """Assemble the compact line, or "" when nothing is noteworthy.

    Every part fails soft and independently: a broken care plan must not cost
    the person the air-quality warning, and vice versa.
    """
    parts = [
        _environment_part(provider),
        _next_due_part(manager, now),
        _vitals_part(plan),
        _session_part(plan),
    ]
    body = " | ".join(part for part in parts if part)
    if not body:
        return ""
    return f"CARE NOW: {body}"[:_MAX_LINE_CHARS]


class CareNowInjector:
    """Decides WHETHER to put a `CARE NOW` row into the conversation.

    Deliberately separate from building the line: the wording is the model's
    business, but whether the person hears about it again is a deterministic
    decision about bands and cooldowns.
    """

    def __init__(self, cooldown_seconds: int = _DEFAULT_COOLDOWN_SECONDS):
        self.cooldown_seconds = int(cooldown_seconds)
        self._lock = threading.RLock()
        self._last_line = ""
        self._last_injected_at = 0.0

    def should_inject(self, line: str, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        with self._lock:
            if not line:
                return False
            if line == self._last_line:
                return False
            if self._last_line and (now - self._last_injected_at) < self.cooldown_seconds:
                return False
            return True

    def mark_injected(self, line: str, now: Optional[float] = None) -> None:
        with self._lock:
            self._last_line = line
            self._last_injected_at = now if now is not None else time.time()

    def maybe_inject(self, message_history: List[Dict[str, Any]],
                     rewarm: bool = False, **sources) -> str:
        """Append one `CARE NOW` row when there is something new to say.

        Returns the injected line, or "". Never raises: a health snapshot must
        never be the reason a voice turn fails.
        """
        try:
            line = build_care_now(**sources)
            if not self.should_inject(line):
                return ""
            message_history.append({"role": "system", "content": line})
            self.mark_injected(line)
            if rewarm:
                from core.llm import register_history
                register_history(message_history)
            print(f"[CareNow] {line}")
            return line
        except Exception as exc:
            print(f"[CareNow] snapshot skipped: {exc}")
            return ""


_injector: Optional[CareNowInjector] = None
_injector_lock = threading.RLock()


def get_care_now_injector(cooldown_seconds: Optional[int] = None) -> CareNowInjector:
    global _injector
    with _injector_lock:
        if _injector is None:
            if cooldown_seconds is None:
                try:
                    from tools_and_config.config_loader import get_full_config
                    cooldown_seconds = int(
                        get_full_config().get("environment", {})
                        .get("care_now_cooldown_seconds", _DEFAULT_COOLDOWN_SECONDS))
                except Exception:
                    cooldown_seconds = _DEFAULT_COOLDOWN_SECONDS
            _injector = CareNowInjector(cooldown_seconds)
        return _injector


def reset_care_now_injector() -> None:
    global _injector
    with _injector_lock:
        _injector = None


def current_sources() -> Dict[str, Any]:
    """The live provider/manager/plan trio, each independently optional."""
    sources: Dict[str, Any] = {"provider": None, "manager": None, "plan": None}
    try:
        from core.health.environment import get_environment_provider
        sources["provider"] = get_environment_provider()
    except Exception:
        pass
    try:
        from core.senior.senior_care_manager import get_senior_care_manager
        sources["manager"] = get_senior_care_manager()
    except Exception:
        pass
    try:
        from core.senior.care_plan import get_care_plan_store
        sources["plan"] = get_care_plan_store()
    except Exception:
        pass
    return sources
