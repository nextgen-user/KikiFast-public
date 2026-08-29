"""SeniorCareManager — bridges the care plan onto the existing workers scheduler.

When the ``senior`` assistant mode is entered, every enabled routine event,
legacy reminder, and guided exercise in ``care_plan.json`` is materialized into
a real background *worker* (§5.13), plus a daily caregiver-summary worker.
Leaving the mode cancels them. Schedule edits re-sync immediately; progress
inside an active multi-turn session deliberately does not rebuild its worker.

Nothing here re-implements scheduling or speaking. It drives
``WorkerManager.create_worker``; when any care item becomes due, the worker only
opens persisted session state and queues ``main.py``. The foreground microphone
lifecycle and conversational care agent own every spoken turn.

Schedule mapping:
  recurring {value=<sec>}  -> recurring worker, first fire <sec> from now
  daily     {value=HH:MM}  -> recurring worker (interval 24h), back-dated
                             last_fired_at so the first fire lands at HH:MM
  once      {value=ISO}    -> one-shot scheduled_time worker
"""

import json
from datetime import datetime, timedelta
from typing import List, Optional

from core.workers.worker_engine import TriggerType

_DAY_SECONDS = 86400
_WORKER_PREFIX = "senior:"


async def execute_scheduled_routine(worker) -> tuple[bool, str, Optional[str]]:
    """Open a due session; main.py owns its first real voice turn."""
    from core.senior.care_plan import get_care_plan_store
    event_id = str(worker.name or "").rsplit(":", 1)[-1]
    try:
        state = get_care_plan_store().start_care_session(event_id)
        result = json.dumps({
            "status": "care_session_ready", "event_id": event_id,
            "session_id": state.get("id"),
        }, ensure_ascii=False, separators=(",", ":"))
        # A worker never speaks or creates a fake participant response.
        return True, result, None
    except Exception as exc:
        return False, f"Could not start care session {event_id}: {exc}", None


class SeniorCareManager:
    def __init__(self, worker_manager, care_plan, config: Optional[dict] = None):
        self.wm = worker_manager
        self.plan = care_plan
        self.config = config or {}
        self._active = False

    # ---------------------------------------------------------------- public
    def is_active(self) -> bool:
        return self._active

    def activate(self) -> int:
        """(Re)build all senior workers from the current care plan. Idempotent."""
        self._clear_existing()
        count = 0
        for item in self.plan.all_active_schedules():
            try:
                if self._materialize(item):
                    count += 1
            except Exception as e:
                print(f"[SeniorCare] Failed to schedule {item.get('id')}: {e}")
        try:
            if self._materialize_daily_summary():
                count += 1
        except Exception as e:
            print(f"[SeniorCare] Failed to schedule daily summary: {e}")
        self._active = True
        print(f"[SeniorCare] Activated — {count} care worker(s) scheduled.")
        return count

    def deactivate(self) -> None:
        self._clear_existing()
        self._active = False
        print("[SeniorCare] Deactivated — care workers cancelled.")

    def sync_workers(self) -> int:
        """Re-materialize after a care-plan edit (only if currently active)."""
        if not self._active:
            return 0
        return self.activate()

    def is_item_scheduled(self, item_id: str) -> bool:
        """Return whether an active worker exists for one care-plan item."""
        suffix = f":{item_id}"
        try:
            return any(
                worker.name.startswith(_WORKER_PREFIX)
                and worker.name.endswith(suffix)
                and worker.is_active()
                for worker in self.wm.list_workers(include_completed=True)
            )
        except Exception:
            return False

    def schedule_receipt(self, item_id: str = "") -> dict:
        """Verified runtime receipt for one or all materialized care workers."""
        suffix = f":{item_id}" if item_id else ""
        receipts = []
        now = datetime.now()
        try:
            workers = self.wm.list_workers(include_completed=True)
        except Exception as exc:
            return {"status": "error", "scheduled": False, "reason": str(exc)}
        for worker in workers:
            if not worker.name.startswith(_WORKER_PREFIX):
                continue
            if suffix and not worker.name.endswith(suffix):
                continue
            if not worker.is_active():
                continue
            trigger = worker.trigger
            next_at = None
            if trigger.trigger_type == TriggerType.SCHEDULED_TIME.value:
                next_at = trigger.scheduled_time
            elif (trigger.trigger_type == TriggerType.RECURRING.value
                  and trigger.interval_seconds):
                try:
                    last = datetime.fromisoformat(trigger.last_fired_at)
                    target = last + timedelta(seconds=int(trigger.interval_seconds))
                    while target <= now:
                        target += timedelta(seconds=int(trigger.interval_seconds))
                    next_at = target.isoformat()
                except (TypeError, ValueError):
                    next_at = None
            receipts.append({
                "worker_id": worker.id,
                "worker_name": worker.name,
                "status": worker.status,
                "trigger_type": trigger.trigger_type,
                "next_trigger_at": next_at,
            })
        return {
            "status": "scheduled" if receipts else "not_scheduled",
            "scheduled": bool(receipts),
            "item_id": item_id,
            "timezone": "Asia/Kolkata",
            "workers": receipts,
        }

    def continuous_vision_required(self) -> bool:
        """Whether the currently active guided session requests every-turn vision."""
        try:
            session = self.plan.care_session_state()
            return (session.get("status") == "active"
                    and bool(session.get("continuous_vision", False)))
        except Exception:
            return False

    # --------------------------------------------------------------- internal
    def _clear_existing(self) -> None:
        try:
            for w in self.wm.list_workers(include_completed=True):
                if w.name.startswith(_WORKER_PREFIX) and w.is_active():
                    self.wm.cancel_worker(w.id)
        except Exception as e:
            print(f"[SeniorCare] Clear existing failed: {e}")

    def _materialize(self, item: dict) -> bool:
        kind = item.get("_kind")  # "reminder" | "exercise"
        sched = item.get("schedule") or {}
        skind = sched.get("kind")
        name = f"{_WORKER_PREFIX}{kind}:{item.get('id')}"
        task = self._task_for(item)

        if skind == "recurring":
            worker = self.wm.create_worker(
                name=name, task_description=task,
                trigger_type=TriggerType.RECURRING.value,
                trigger_value=str(int(sched.get("value", 3600))),
                created_by="senior_care",
            )
            # Start the clock now so the first fire is one interval away
            worker.trigger.last_fired_at = datetime.now().isoformat()
            self.wm._save()
            return True

        if skind == "daily":
            worker = self.wm.create_worker(
                name=name, task_description=task,
                trigger_type=TriggerType.RECURRING.value,
                trigger_value=str(_DAY_SECONDS),
                created_by="senior_care",
            )
            first = _next_daily(str(sched.get("value", "09:00")))
            worker.trigger.last_fired_at = (first - timedelta(seconds=_DAY_SECONDS)).isoformat()
            self.wm._save()
            return True

        if skind == "once":
            self.wm.create_worker(
                name=name, task_description=task,
                trigger_type=TriggerType.SCHEDULED_TIME.value,
                trigger_value=str(sched.get("value", "")),
                created_by="senior_care",
            )
            return True

        return False

    def _materialize_daily_summary(self) -> bool:
        contacts = self.plan.contacts_for("daily_summary")
        if not contacts:
            return False
        hhmm = str(self.config.get("daily_summary_time", "20:00"))
        worker = self.wm.create_worker(
            name=f"{_WORKER_PREFIX}daily_summary",
            task_description=self._daily_summary_task(contacts),
            trigger_type=TriggerType.RECURRING.value,
            trigger_value=str(_DAY_SECONDS),
            created_by="senior_care",
        )
        first = _next_daily(hhmm)
        worker.trigger.last_fired_at = (first - timedelta(seconds=_DAY_SECONDS)).isoformat()
        self.wm._save()
        return True

    # ----------------------------------------------------------- task prompts
    def _task_for(self, item: dict) -> str:
        label = (item.get("title") or item.get("name") or item.get("message")
                 or "scheduled care item")
        return (
            f"Scheduled foreground care-session trigger for {item.get('_kind', 'care')} "
            f"id {item.get('id', '')}, titled '{label}'. The worker only opens "
            "persisted session state and queues the normal voice loop. It must not "
            "generate dialogue, task instructions, or simulate the person."
        )

    def _daily_summary_task(self, contacts: List[dict]) -> str:
        emails = ", ".join(c.get("email", "") for c in contacts)
        today = datetime.now().strftime("%Y-%m-%d")
        return (
            "Compose and email the caregiver's DAILY SUMMARY for the senior. Steps: "
            "1) call get_care_plan with section 'care_log' to review what happened today; "
            "2) call recall_memory if useful for the day's mood/topics or dated background research; "
            f"3) write a warm, concise summary in English covering the day ({today}): reminders "
            "delivered and adherence, mood and engagement, any concerns or notable moments, and "
            "whether anything needs the family's attention; "
            f"4) call send_care_email once to send it to: {emails} with a clear subject like "
            f"'Kiki daily summary — {today}'. Do NOT speak this aloud. Respond with final JSON "
            "status 'completed', speak=false, summary describing what you emailed."
        )


# ---------------------------------------------------------------------- utils
def _next_daily(hhmm: str) -> datetime:
    """Next datetime matching HH:MM (today if still ahead, else tomorrow)."""
    try:
        hh, mm = [int(x) for x in hhmm.split(":")[:2]]
    except Exception:
        hh, mm = 9, 0
    now = datetime.now()
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


# --------------------------------------------------------------------- singleton
_manager: Optional[SeniorCareManager] = None


def get_senior_care_manager(worker_manager=None, care_plan=None, config=None) -> Optional[SeniorCareManager]:
    """Get/create the singleton. Requires worker_manager + care_plan on first call."""
    global _manager
    if _manager is None:
        if worker_manager is None or care_plan is None:
            return None
        _manager = SeniorCareManager(worker_manager, care_plan, config)
    return _manager
