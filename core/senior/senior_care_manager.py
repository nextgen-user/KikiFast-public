"""SeniorCareManager — bridges the care plan onto the existing workers scheduler.

When the ``senior`` assistant mode is entered, every enabled reminder / guided
exercise in ``care_plan.json`` is materialized into a real background *worker*
(§5.13) plus a daily caregiver-summary worker. Leaving the mode cancels them.
Editing the care plan by voice re-syncs (cancel + rebuild) so changes take
effect immediately.

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

from datetime import datetime, timedelta
from typing import List, Optional

from core.workers.worker_engine import TriggerType

_DAY_SECONDS = 86400
_WORKER_PREFIX = "senior:"


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
