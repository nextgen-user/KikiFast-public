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
    from core.workers.worker_engine import WorkerDeferred
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
        # Kiki is already mid-conversation with the person about another
        # routine. That is a scheduling collision, not a broken worker: wait
        # and try again rather than spending a retry and reporting a failure
        # the person never needed to hear about.
        if "another care session is already active" in str(exc):
            raise WorkerDeferred(
                f"care session {event_id} is waiting for the active session "
                f"to finish") from exc
        return False, f"Could not start care session {event_id}: {exc}", None


# Set by main.py at wiring time: hands an event id to the foreground voice
# lifecycle so a care session opens and speaks on the NEXT turn. Same route the
# scheduler uses — a session must never be spoken by a worker thread.
_foreground_hook = None


def set_foreground_hook(callback) -> None:
    global _foreground_hook
    _foreground_hook = callback


def start_care_session_now(event_ref: str = "") -> str:
    """Begin a care routine immediately instead of waiting for its schedule.

    ``event_ref`` is a routine-event id, or part of its title ("neck", "waist").
    With nothing at all, the single enabled routine is used when there is only
    one, because "start my exercise" is unambiguous with one routine and
    dangerous to guess at with five.
    """
    from core.senior.care_plan import get_care_plan_store

    plan = get_care_plan_store()
    events = [e for e in (plan.data.get("routine_events") or [])
              if e.get("enabled", True)]
    if not events:
        return ("CARE_ACTION_FAILED: there are no enabled routines in the care "
                "plan to start.")

    ref = str(event_ref or "").strip().lower()
    if not ref:
        if len(events) != 1:
            titles = ", ".join(str(e.get("title", "")) for e in events)
            return (f"Which routine should I start? The care plan has: {titles}.")
        match = events[0]
    else:
        match = next((e for e in events if str(e.get("id", "")).lower() == ref), None)
        if match is None:
            hits = [e for e in events
                    if ref in str(e.get("title", "")).lower()
                    or ref in str(e.get("objective", "")).lower()]
            if not hits:
                titles = ", ".join(str(e.get("title", "")) for e in events)
                return (f"CARE_ACTION_FAILED: no care routine matches "
                        f"{event_ref!r}. The plan has: {titles}.")
            if len(hits) > 1:
                titles = ", ".join(str(e.get("title", "")) for e in hits)
                return f"Which one did you mean: {titles}?"
            match = hits[0]

    event_id = str(match.get("id", ""))
    try:
        state = plan.start_care_session(event_id)
    except Exception as exc:
        return f"CARE_ACTION_FAILED: could not start {match.get('title')}: {exc}"

    if _foreground_hook is None:
        # The session is open but nothing can voice it; say so rather than
        # letting the person wait for a routine that will never speak.
        plan.finish_care_session("cancelled")
        return ("CARE_ACTION_FAILED: the foreground care route is not wired, so "
                "the session was not started.")
    try:
        _foreground_hook(event_id)
    except Exception as exc:
        plan.finish_care_session("cancelled")
        return f"CARE_ACTION_FAILED: could not open the care turn: {exc}"

    print(f"[SeniorCare] Immediate session requested for "
          f"{match.get('title')!r} ({event_id})")
    return json.dumps({
        "status": "care_session_starting",
        "event_id": event_id,
        "title": match.get("title", ""),
        "session_id": state.get("id"),
        "note": ("The session is open and Kiki will begin guiding it on this "
                 "turn. Do not also describe the routine yourself."),
    }, ensure_ascii=False)


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
        # activate() rebuilds every care worker from the plan, so the old set
        # is deleted outright rather than cancelled in place. Cancelling left
        # the rows behind, and because each care-plan edit re-activates, they
        # piled up — five real events had grown into twenty-four workers.
        try:
            self.wm.remove_workers_by_prefix(_WORKER_PREFIX)
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
