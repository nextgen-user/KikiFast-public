"""SeniorCareManager — bridges the care plan onto the existing workers scheduler.

When the ``senior`` assistant mode is entered, every enabled routine event,
legacy reminder, and guided exercise in ``care_plan.json`` is materialized into
a real background *worker* (§5.13), plus a daily caregiver-summary worker.
Leaving the mode cancels them. Schedule edits re-sync immediately; progress
inside an active multi-turn session deliberately does not rebuild its worker.

Nothing here re-implements scheduling or speaking — it drives
``WorkerManager.create_worker`` and lets the normal worker path speak the
reminder aloud (``execute_worker`` → ``_speak_text`` → ``TTSStreamer``) under
the usual single-slot preempt/rewarm discipline.

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
    """Run a due routine through the sole care brain, then return direct TTS text."""
    from core.brain.action_agent import run_complex_query

    event_id = str(worker.name or "").rsplit(":", 1)[-1]
    request = (
        "CARE_PLAN_DELEGATION: A scheduled daily routine event is due now. "
        f"Read event id {event_id}, start or idempotently resume its care_session, "
        "and conduct the returned turn_actions adaptively. Produce the exact warm "
        "voice-ready words Kiki should say now, then stop if a response is needed.")
    result = str(await run_complex_query(request, context=worker.task_description))
    # Only CARE_ACTION_FAILED carries voice-ready words after its marker. The
    # generic markers are written FOR the speaking model ("Tell Alex it did
    # not finish…"), so splitting them on the first colon would read a relay
    # instruction out loud to the person the routine is caring for.
    marker = "CARE_ACTION_FAILED:"
    if result.startswith(marker):
        return False, result, result[len(marker):].strip()
    if result.startswith(("ACTION FAILED", "ACTION INCOMPLETE")):
        return False, result, None
    return True, result, result


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
        lang = self.plan.language()
        lang_note = ("Speak in natural, clear Hindi (Devanagari)."
                     if lang == "hi" else "Speak in simple, clear English.")
        if item.get("_kind") == "routine_event":
            actions = item.get("actions", [])
            interactive = any(action.get("needs_response") for action in actions)
            action_json = json.dumps(actions, ensure_ascii=False)
            session_rule = (
                f"First call update_care_plan(section='care_session', action='start', "
                f"data={{'event_id':'{item.get('id', '')}'}}). Read the returned turn_actions, "
                "conduct ALL of them in order, then stop at the response-required action so "
                "the person can answer. The foreground "
                "care agent will continue the persisted session on their next turn."
                if interactive else
                "Carry out the actions in order now. Use tools for tool actions, combine the "
                "spoken actions into one warm concise interaction, and add a care_log entry."
            )
            return (
                f"A daily care-routine event is due: '{item.get('title', '')}' "
                f"(event id {item.get('id', '')}, category {item.get('category', 'other')}). "
                f"{lang_note} This is an ordered care sequence, not a generic reminder. "
                f"Actions JSON: {action_json}. Continuous per-turn vision feedback is "
                f"{'enabled' if item.get('continuous_vision') else 'disabled'} for this event. "
                f"{session_rule} The actions are an adaptable goal outline, not fixed dialogue: "
                "the voice AI must phrase them naturally and accept skip/repeat/slower/change/stop "
                "requests. Adapt warmly to the person's response and never claim an action was "
                "completed if it was not. Respond ONLY "
                "with final JSON status, speak=true, and the exact speak_text to say now."
            )
        if item.get("_kind") == "exercise":
            steps = " | ".join(item.get("steps", []))
            return (
                f"It is time for the guided exercise '{item.get('name','')}' "
                f"prescribed by {item.get('prescribed_by') or 'the caregiver'}. "
                f"{lang_note} Gently and warmly lead the senior through these steps one at a "
                f"time, encouraging them and pausing between steps: {steps}. "
                "Respond ONLY with final JSON: status 'completed', speak=true, and speak_text "
                "containing your warm spoken guidance. After delivering, if the senior clearly "
                "declined or seemed unwell, call update_care_plan to add a care_log note."
            )
        # reminder
        return (
            f"It is time for a {item.get('category','')} reminder for the senior. "
            f"{lang_note} Warmly and briefly remind them: \"{item.get('message','')}\". "
            "Sound caring, not robotic; add a light friendly touch. Respond ONLY with final "
            "JSON: status 'completed', speak=true, speak_text containing the spoken reminder. "
            "Then call update_care_plan with section 'care_log' action 'add' to record that "
            "you delivered this reminder (and the senior's response if you heard one)."
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
