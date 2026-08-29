"""Care Plan store for Senior Citizen Mode.

A small JSON-backed store (mirrors ``core/brain/knowledge_base.py``) holding the
caregiver-defined care plan: the senior's profile, family contacts, scheduled
reminders (medicine / hydration / meal / appointment / exercise), guided
exercise routines, family-approved music/topics, and a rolling ``care_log`` of
what actually happened during the day (used for the daily caregiver summary).

Voice-first: Kiki sends caregiver/senior requests to the complex care agent,
which owns ``get_care_plan``/``update_care_plan`` and verifies scheduling. The
store remains plain JSON that a caregiver Web UI can render and edit through a
future API.

Saves are atomic (tmp + ``os.replace``) so a crash mid-write never corrupts it.
"""

import json
import os
import statistics
import threading
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from tools_and_config.config_loader import get_full_config


# ============================================================================
# Configuration / paths
# ============================================================================

VALID_REMINDER_CATEGORIES = {
    "medicine", "hydration", "meal", "appointment", "exercise", "other",
}

# A schedule is one of:
#   {"kind": "recurring", "value": <seconds:int>}   -> fires every N seconds
#   {"kind": "daily",     "value": "HH:MM"}         -> fires each day at HH:MM
#   {"kind": "once",      "value": "<ISO datetime>"}-> fires once at that time
VALID_SCHEDULE_KINDS = {"recurring", "daily", "once"}
VALID_ROUTINE_CATEGORIES = {
    "morning", "medicine", "hydration", "meal", "exercise", "sleep",
    "appointment", "wellbeing", "memory", "social", "safety", "vitals", "other",
}
VALID_ROUTINE_ACTION_TYPES = {
    "speak", "check_in", "guided_step", "memory_activity", "play_music",
    "observe", "measure_vital", "log", "notify_caregiver",
}


def _default_care_plan_path() -> Path:
    return Path(__file__).parent.parent.parent / "care_plan.json"


def get_care_plan_path() -> Path:
    cfg = get_full_config().get("senior_mode", {})
    return Path(cfg.get("care_plan_file", str(_default_care_plan_path())))


DEFAULT_STRUCTURE: Dict[str, Any] = {
    "senior": {"name": "", "language": "hi", "notes": "", "health_conditions": []},
    "family_contacts": [],      # {name, relationship, email, notify_on:[alert,daily_summary]}
    "reminders": [],            # {id, category, message, schedule, enabled}
    "exercises": [],            # {id, name, steps:[...], schedule, prescribed_by, enabled}
    # A person's day: each scheduled event is a sequence Kiki carries out.
    "routine_events": [],       # {id,title,category,schedule,actions,continuous_vision,...}
    "active_session": None,     # persisted multi-turn routine currently being conducted
    "approved_music": [],       # ["song / query", ...]
    "approved_topics": [],      # ["cricket", "old bollywood", ...]
    "care_log": [],             # {ts, kind, text}
    "health_measurements": [], # trusted numeric readings for trend analysis
    "metadata": {"created": "", "last_updated": ""},
}

_MAX_CARE_LOG = 500


class CarePlan:
    """JSON-backed care plan with atomic saves."""

    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = file_path or get_care_plan_path()
        self._lock = threading.RLock()
        self.data = self._load()

    # ------------------------------------------------------------------ io
    def _load(self) -> Dict[str, Any]:
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"[CarePlan] Loaded from {self.file_path}")
                return self._migrate(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"[CarePlan] Error loading ({e}); starting fresh.")
        data = json.loads(json.dumps(DEFAULT_STRUCTURE))
        data["metadata"]["created"] = datetime.now().isoformat()
        return data

    def _migrate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key, default in DEFAULT_STRUCTURE.items():
            if key not in data:
                data[key] = json.loads(json.dumps(default))
        return data

    def save(self) -> bool:
        with self._lock:
            try:
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                self.data["metadata"]["last_updated"] = datetime.now().isoformat()
                tmp = self.file_path.with_suffix(".json.tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, indent=2, ensure_ascii=False)
                os.replace(tmp, self.file_path)
                return True
            except Exception as e:
                print(f"[CarePlan] Error saving: {e}")
                return False

    # -------------------------------------------------------------- getters
    def get_section(self, name: str) -> Any:
        with self._lock:
            return json.loads(json.dumps(self.data.get(name)))

    def all_active_schedules(self) -> List[Dict[str, Any]]:
        """Reminders + exercises that are enabled and have a valid schedule.

        Returns a flat list of dicts each carrying ``_kind`` (``reminder`` or
        ``exercise``) so the care manager can materialize one worker per item.
        """
        with self._lock:
            items: List[Dict[str, Any]] = []
            for r in self.data.get("reminders", []):
                if r.get("enabled", True) and _valid_schedule(r.get("schedule")):
                    items.append({**r, "_kind": "reminder"})
            for ex in self.data.get("exercises", []):
                if ex.get("enabled", True) and _valid_schedule(ex.get("schedule")):
                    items.append({**ex, "_kind": "exercise"})
            for event in self.data.get("routine_events", []):
                if event.get("enabled", True) and _valid_schedule(event.get("schedule")):
                    items.append({**event, "_kind": "routine_event"})
            return items

    def contacts_for(self, purpose: str) -> List[Dict[str, Any]]:
        """Family contacts whose ``notify_on`` includes ``purpose`` and have an email."""
        with self._lock:
            out = []
            for c in self.data.get("family_contacts", []):
                if c.get("email") and purpose in (c.get("notify_on") or []):
                    out.append(json.loads(json.dumps(c)))
            return out

    def language(self) -> str:
        with self._lock:
            return str(self.data.get("senior", {}).get("language") or "hi")

    def daily_routine(self) -> List[Dict[str, Any]]:
        """Enabled routine events in the order Kiki will encounter each day."""
        with self._lock:
            events = [json.loads(json.dumps(event))
                      for event in self.data.get("routine_events", [])
                      if event.get("enabled", True)]

        def sort_key(event):
            schedule = event.get("schedule") or {}
            kind = schedule.get("kind")
            value = str(schedule.get("value", ""))
            if kind == "daily":
                return (0, value, event.get("title", ""))
            if kind == "once":
                return (1, value, event.get("title", ""))
            return (2, value, event.get("title", ""))

        return sorted(events, key=sort_key)

    # -------------------------------------------------------------- mutators
    def set_senior_profile(self, **fields) -> bool:
        with self._lock:
            prof = self.data.setdefault("senior", {})
            for k, v in fields.items():
                if v is not None:
                    prof[k] = v
        return self.save()

    def add_reminder(self, category: str, message: str, schedule: Dict[str, Any],
                     enabled: bool = True) -> Dict[str, Any]:
        category = (category or "other").strip().lower()
        if category not in VALID_REMINDER_CATEGORIES:
            category = "other"
        message = str(message or "").strip()
        if not message:
            raise ValueError("reminder message is required")
        schedule = _normalize_schedule(schedule)
        if not _valid_schedule(schedule):
            raise ValueError(
                "a valid reminder schedule is required: daily HH:MM, "
                "recurring positive seconds, or once ISO datetime")
        item = {
            "id": uuid.uuid4().hex[:8],
            "category": category,
            "message": message,
            "schedule": schedule,
            "enabled": bool(enabled),
        }
        with self._lock:
            self.data.setdefault("reminders", []).append(item)
        if not self.save():
            with self._lock:
                self.data["reminders"] = [
                    existing for existing in self.data.get("reminders", [])
                    if existing.get("id") != item["id"]
                ]
            raise IOError("could not persist reminder")
        return item

    def edit_reminder(self, reminder_id: str, **fields) -> bool:
        if "message" in fields:
            fields["message"] = str(fields["message"] or "").strip()
            if not fields["message"]:
                raise ValueError("reminder message cannot be empty")
        if "schedule" in fields:
            fields["schedule"] = _normalize_schedule(fields["schedule"])
            if not _valid_schedule(fields["schedule"]):
                raise ValueError(
                    "a valid reminder schedule is required: daily HH:MM, "
                    "recurring positive seconds, or once ISO datetime")
        return self._edit_item("reminders", reminder_id, fields)

    def remove_reminder(self, reminder_id: str) -> bool:
        return self._remove_item("reminders", reminder_id)

    def add_exercise(self, name: str, steps: List[str], schedule: Optional[Dict[str, Any]] = None,
                     prescribed_by: str = "", enabled: bool = True) -> Dict[str, Any]:
        name = str(name or "").strip()
        steps = [str(s).strip() for s in (steps or []) if str(s).strip()]
        if not name:
            raise ValueError("exercise name is required")
        if not steps:
            raise ValueError("a guided exercise requires at least one step")
        normalized_schedule = _normalize_schedule(schedule) if schedule else {}
        if schedule and not _valid_schedule(normalized_schedule):
            raise ValueError(
                "a valid exercise schedule must be daily HH:MM, recurring "
                "positive seconds, or once ISO datetime")
        item = {
            "id": uuid.uuid4().hex[:8],
            "name": name,
            "steps": steps,
            "schedule": normalized_schedule,
            "prescribed_by": str(prescribed_by or "").strip(),
            "enabled": bool(enabled),
        }
        with self._lock:
            self.data.setdefault("exercises", []).append(item)
        if not self.save():
            with self._lock:
                self.data["exercises"] = [
                    existing for existing in self.data.get("exercises", [])
                    if existing.get("id") != item["id"]
                ]
            raise IOError("could not persist exercise")
        return item

    def edit_exercise(self, exercise_id: str, **fields) -> bool:
        return self._edit_item("exercises", exercise_id, fields)

    def remove_exercise(self, exercise_id: str) -> bool:
        return self._remove_item("exercises", exercise_id)

    def add_routine_event(self, title: str, category: str,
                          schedule: Dict[str, Any], actions: List[Dict[str, Any]],
                          enabled: bool = True, source: str = "user",
                          evidence: str = "", adaptation: Optional[Dict[str, Any]] = None,
                          continuous_vision: bool = False, objective: str = ""
                          ) -> Dict[str, Any]:
        title = str(title or "").strip()
        if not title:
            raise ValueError("routine event title is required")
        category = str(category or "other").strip().lower()
        if category not in VALID_ROUTINE_CATEGORIES:
            category = "other"
        raw_schedule = schedule
        schedule = _normalize_schedule(schedule)
        if not _valid_schedule(schedule):
            raise ValueError(
                "a valid routine schedule is required: "
                '{"kind":"daily","value":"HH:MM"} or {"kind":"once","value":'
                '"<ISO datetime>"} or {"kind":"recurring","value":<seconds>}. '
                f"Received: {json.dumps(raw_schedule, ensure_ascii=False, default=str)[:200]}")
        action_problems: List[str] = []
        actions = _normalize_routine_actions(actions, action_problems)
        if not actions:
            raise ValueError(_no_valid_actions_error(action_problems))
        source = str(source or "user").strip().lower()
        evidence = str(evidence or "").strip()
        if not isinstance(continuous_vision, bool):
            raise ValueError("continuous_vision must be true or false")
        if source == "idle_mind" and len(evidence) < 20:
            raise ValueError(
                "idle-mind routine changes require concrete repeated-routine evidence")
        adaptation = adaptation if isinstance(adaptation, dict) else {}
        try:
            review_after = max(
                1, int(adaptation.get("review_after_occurrences", 3) or 3))
        except (TypeError, ValueError):
            review_after = 3
        # Agent retries and network timeouts must be idempotent. If the exact
        # event is already present, return it rather than creating a duplicate.
        with self._lock:
            duplicate = next((existing for existing in self.data.get("routine_events", [])
                              if existing.get("enabled", True)
                              and str(existing.get("title", "")).casefold() == title.casefold()
                              and existing.get("category") == category
                              and existing.get("schedule") == schedule), None)
            if duplicate is not None:
                result = json.loads(json.dumps(duplicate))
                result["_existing"] = True
                return result
        now = datetime.now().isoformat()
        item = {
            "id": uuid.uuid4().hex[:8],
            "title": title,
            "objective": str(objective or title).strip(),
            "category": category,
            "schedule": schedule,
            "actions": actions,
            # While this interactive session is active, main.py captures a
            # fresh frame after every turn through the existing Gemini vision
            # pipeline and injects its description before the next turn.
            "continuous_vision": bool(continuous_vision),
            "enabled": bool(enabled),
            "source": source,
            "evidence": evidence,
            "adaptation": {
                "allowed": bool(adaptation.get("allowed", True)),
                "strategy": str(adaptation.get("strategy", "adapt_to_response")).strip(),
                "review_after_occurrences": review_after,
            },
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self.data.setdefault("routine_events", []).append(item)
        if not self.save():
            with self._lock:
                self.data["routine_events"] = [
                    existing for existing in self.data.get("routine_events", [])
                    if existing.get("id") != item["id"]
                ]
            raise IOError("could not persist routine event")
        return item

    def edit_routine_event(self, event_id: str, **fields) -> bool:
        fields = dict(fields)
        if "title" in fields and not str(fields["title"] or "").strip():
            raise ValueError("routine event title cannot be empty")
        if "objective" in fields:
            fields["objective"] = str(fields["objective"] or "").strip()
        if "schedule" in fields:
            fields["schedule"] = _normalize_schedule(fields["schedule"])
            if not _valid_schedule(fields["schedule"]):
                raise ValueError("routine event schedule is invalid")
        if "actions" in fields:
            action_problems: List[str] = []
            fields["actions"] = _normalize_routine_actions(
                fields["actions"], action_problems)
            if not fields["actions"]:
                raise ValueError(_no_valid_actions_error(action_problems))
        if "continuous_vision" in fields:
            if not isinstance(fields["continuous_vision"], bool):
                raise ValueError("continuous_vision must be true or false")
        fields["updated_at"] = datetime.now().isoformat()
        return self._edit_item("routine_events", event_id, fields)

    def remove_routine_event(self, event_id: str) -> bool:
        with self._lock:
            active = self.data.get("active_session") or {}
            if active.get("event_id") == event_id and active.get("status") == "active":
                self.data["active_session"] = None
        return self._remove_item("routine_events", event_id)

    def start_care_session(self, event_id: str) -> Dict[str, Any]:
        with self._lock:
            event = next((item for item in self.data.get("routine_events", [])
                          if item.get("id") == event_id), None)
            if event is None:
                raise ValueError("no routine event with that id")
            active = self.data.get("active_session")
            if isinstance(active, dict) and active.get("status") == "active":
                if active.get("event_id") == event_id:
                    # A worker retry must resume, never restart, the same session.
                    return self.care_session_state()
                raise ValueError(
                    "another care session is already active; finish or cancel it first")
            session = {
                "id": uuid.uuid4().hex[:8],
                "event_id": event_id,
                "event_title": event.get("title", ""),
                "status": "active",
                "action_index": 0,
                "started_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "responses": [],
                # This is a session-local outline. The voice AI may adapt,
                # repeat, skip, or replace its remaining steps without silently
                # changing the recurring caregiver-approved routine.
                "session_actions": json.loads(json.dumps(event.get("actions", []))),
                "adaptations": [],
                "vision_override": None,
            }
            self.data["active_session"] = session
        if not self.save():
            raise IOError("could not persist active care session")
        return self.care_session_state()

    def advance_care_session(self, response: str = "") -> Dict[str, Any]:
        with self._lock:
            session = self.data.get("active_session")
            if not isinstance(session, dict) or session.get("status") != "active":
                raise ValueError("there is no active care session")
            event = next((item for item in self.data.get("routine_events", [])
                          if item.get("id") == session.get("event_id")), None)
            if event is None:
                raise ValueError("the active session's routine event no longer exists")
            actions = session.get("session_actions") or event.get("actions", [])
            current_index = int(session.get("action_index", 0))
            presented = _routine_turn_actions(actions, current_index)
            if str(response or "").strip():
                session.setdefault("responses", []).append({
                    "at": datetime.now().isoformat(),
                    "action_index": current_index,
                    "response": str(response).strip()[:500],
                })
            # One care turn may contain an introduction plus the first actual
            # check-in. Advance past everything Kiki just presented, not merely
            # one array element, so replies are attached to the right step.
            session["action_index"] = current_index + len(presented)
            session["updated_at"] = datetime.now().isoformat()
            if session["action_index"] >= len(actions):
                session["status"] = "completed"
                session["completed_at"] = datetime.now().isoformat()
            else:
                next_actions = _routine_turn_actions(
                    actions, session["action_index"])
                # No reply is needed after a trailing close/log/notification
                # batch. Return it once as completion_actions and finish the
                # session; the care agent still performs those actions.
                if next_actions and not any(
                        action.get("needs_response") for action in next_actions):
                    session["completion_actions"] = next_actions
                    session["action_index"] = len(actions)
                    session["status"] = "completed"
                    session["completed_at"] = datetime.now().isoformat()
        if not self.save():
            raise IOError("could not persist care-session progress")
        return self.care_session_state()

    def adapt_care_session(self, response: str, remaining_actions: List[Dict[str, Any]],
                           reason: str = "") -> Dict[str, Any]:
        """Replace the current-and-remaining outline for this session only."""
        replacement = _normalize_routine_actions(remaining_actions)
        if not replacement:
            raise ValueError("adaptive session change requires remaining_actions")
        with self._lock:
            session = self.data.get("active_session")
            if not isinstance(session, dict) or session.get("status") != "active":
                raise ValueError("there is no active care session")
            event = next((item for item in self.data.get("routine_events", [])
                          if item.get("id") == session.get("event_id")), None)
            if event is None:
                raise ValueError("the active session's routine event no longer exists")
            actions = session.get("session_actions") or event.get("actions", [])
            idx = max(0, int(session.get("action_index", 0)))
            session["session_actions"] = (
                json.loads(json.dumps(actions[:idx])) + replacement)
            session.setdefault("adaptations", []).append({
                "at": datetime.now().isoformat(),
                "action_index": idx,
                "response": str(response or "").strip()[:500],
                "reason": str(reason or "").strip()[:500],
            })
            session["updated_at"] = datetime.now().isoformat()
        if not self.save():
            raise IOError("could not persist care-session adaptation")
        return self.care_session_state()

    def set_care_session_vision(self, enabled: bool) -> Dict[str, Any]:
        """Override every-turn vision for this live session only."""
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be true or false")
        with self._lock:
            session = self.data.get("active_session")
            if not isinstance(session, dict) or session.get("status") != "active":
                raise ValueError("there is no active care session")
            session["vision_override"] = enabled
            session["updated_at"] = datetime.now().isoformat()
        if not self.save():
            raise IOError("could not persist care-session vision setting")
        return self.care_session_state()

    def finish_care_session(self, status: str = "completed", response: str = "") -> Dict[str, Any]:
        status = str(status or "completed").strip().lower()
        if status not in {"completed", "cancelled", "declined"}:
            raise ValueError("session status must be completed, cancelled, or declined")
        with self._lock:
            session = self.data.get("active_session")
            if not isinstance(session, dict):
                raise ValueError("there is no care session to finish")
            if str(response or "").strip():
                session.setdefault("responses", []).append({
                    "at": datetime.now().isoformat(),
                    "action_index": session.get("action_index", 0),
                    "response": str(response).strip()[:500],
                })
            session["status"] = status
            session["updated_at"] = datetime.now().isoformat()
            session["completed_at"] = datetime.now().isoformat()
        if not self.save():
            raise IOError("could not persist care-session completion")
        return self.care_session_state()

    def care_session_state(self) -> Dict[str, Any]:
        with self._lock:
            session = json.loads(json.dumps(self.data.get("active_session")))
            if not isinstance(session, dict):
                return {"status": "none"}
            event = next((item for item in self.data.get("routine_events", [])
                          if item.get("id") == session.get("event_id")), None)
            if event:
                actions = session.get("session_actions") or event.get("actions", [])
                idx = int(session.get("action_index", 0))
                session["total_actions"] = len(actions)
                turn_actions = (
                    _routine_turn_actions(actions, idx)
                    if session.get("status") == "active" else [])
                session["turn_actions"] = turn_actions
                session["current_action"] = turn_actions[-1] if turn_actions else None
                session["awaiting_response"] = any(
                    action.get("needs_response") for action in turn_actions)
                override = session.get("vision_override")
                session["continuous_vision"] = (
                    override if isinstance(override, bool)
                    else bool(event.get("continuous_vision", False)))
            return session

    def add_family_contact(self, name: str, email: str, relationship: str = "",
                           notify_on: Optional[List[str]] = None) -> Dict[str, Any]:
        item = {
            "name": str(name or "").strip(),
            "relationship": str(relationship or "").strip(),
            "email": str(email or "").strip(),
            "notify_on": notify_on or ["alert", "daily_summary"],
        }
        with self._lock:
            self.data.setdefault("family_contacts", []).append(item)
        self.save()
        return item

    def remove_family_contact(self, name: str) -> bool:
        name = (name or "").strip().lower()
        with self._lock:
            before = len(self.data.get("family_contacts", []))
            self.data["family_contacts"] = [
                c for c in self.data.get("family_contacts", [])
                if c.get("name", "").strip().lower() != name
            ]
            changed = len(self.data["family_contacts"]) != before
        return self.save() and changed

    def add_to_list(self, section: str, value: str) -> bool:
        """Append to ``approved_music`` / ``approved_topics`` (dedup, case-insensitive)."""
        if section not in ("approved_music", "approved_topics"):
            return False
        value = str(value or "").strip()
        if not value:
            return False
        with self._lock:
            lst = self.data.setdefault(section, [])
            if value.lower() not in [x.lower() for x in lst]:
                lst.append(value)
        return self.save()

    def remove_from_list(self, section: str, value: str) -> bool:
        if section not in ("approved_music", "approved_topics"):
            return False
        value = (value or "").strip().lower()
        with self._lock:
            lst = self.data.get(section, [])
            self.data[section] = [x for x in lst if x.strip().lower() != value]
        return self.save()

    def add_care_log(self, kind: str, text: str) -> None:
        entry = {"ts": datetime.now().isoformat(), "kind": str(kind or "note"),
                 "text": str(text or "").strip()}
        with self._lock:
            log = self.data.setdefault("care_log", [])
            log.append(entry)
            if len(log) > _MAX_CARE_LOG:
                del log[: len(log) - _MAX_CARE_LOG]
        self.save()

    def care_log_since(self, iso_ts: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [json.loads(json.dumps(e)) for e in self.data.get("care_log", [])
                    if e.get("ts", "") >= iso_ts]

    def add_health_measurement(self, measurement: str, value: float, unit: str,
                               quality: str, site: str = "", context: str = "",
                               signal: Optional[Dict[str, Any]] = None,
                               routine_event_id: str = "", session_id: str = ""
                               ) -> Dict[str, Any]:
        """Persist one trusted numeric vital reading for personal trends."""
        measurement = str(measurement or "").strip().lower()
        quality = str(quality or "").strip().upper()
        if measurement != "heart_rate":
            raise ValueError("unsupported health measurement")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("health measurement value must be numeric")
        if not 20 <= value <= 250:
            raise ValueError("heart-rate value is outside the sensor's supported range")
        if quality not in {"GOOD", "FAIR"}:
            raise ValueError("only trusted GOOD/FAIR readings may be trended")
        entry = {
            "id": uuid.uuid4().hex[:8],
            "measured_at": datetime.now().astimezone().isoformat(),
            "measurement": measurement,
            "value": round(value, 1),
            "unit": str(unit or "bpm").strip().lower(),
            "quality": quality,
            "site": str(site or "").strip().lower(),
            "context": str(context or "").strip()[:500],
            "signal": json.loads(json.dumps(signal or {})),
            "routine_event_id": str(routine_event_id or "").strip(),
            "session_id": str(session_id or "").strip(),
        }
        with self._lock:
            values = self.data.setdefault("health_measurements", [])
            values.append(entry)
            if len(values) > 2000:
                del values[:len(values) - 2000]
        if not self.save():
            raise IOError("could not persist health measurement")
        return json.loads(json.dumps(entry))

    def health_trend(self, measurement: str = "heart_rate", days: int = 7,
                     limit: int = 30) -> Dict[str, Any]:
        measurement = str(measurement or "heart_rate").strip().lower()
        days = max(1, min(365, int(days or 7)))
        limit = max(1, min(200, int(limit or 30)))
        cutoff = datetime.now().astimezone().timestamp() - days * 86400
        with self._lock:
            rows = json.loads(json.dumps(self.data.get("health_measurements", [])))
        trusted = []
        for row in rows:
            if row.get("measurement") != measurement:
                continue
            try:
                if datetime.fromisoformat(row.get("measured_at", "")).timestamp() < cutoff:
                    continue
                trusted.append(row)
            except (TypeError, ValueError):
                continue
        trusted.sort(key=lambda row: row.get("measured_at", ""))
        values = [float(row["value"]) for row in trusted]
        result: Dict[str, Any] = {
            "measurement": measurement,
            "period_days": days,
            "count": len(values),
            "recent": trusted[-limit:],
        }
        if values:
            result.update({
                "latest": values[-1],
                "median": round(statistics.median(values), 1),
                "minimum": round(min(values), 1),
                "maximum": round(max(values), 1),
                "change_from_median": round(values[-1] - statistics.median(values), 1),
                "unit": trusted[-1].get("unit", "bpm"),
            })
        return result

    # -------------------------------------------------------------- internals
    def _edit_item(self, section: str, item_id: str, fields: Dict[str, Any]) -> bool:
        with self._lock:
            for it in self.data.get(section, []):
                if it.get("id") == item_id:
                    for k, v in fields.items():
                        if v is None:
                            continue
                        if k == "schedule":
                            v = _normalize_schedule(v)
                        it[k] = v
                    break
            else:
                return False
        return self.save()

    def _remove_item(self, section: str, item_id: str) -> bool:
        with self._lock:
            before = len(self.data.get(section, []))
            self.data[section] = [it for it in self.data.get(section, [])
                                  if it.get("id") != item_id]
            changed = len(self.data[section]) != before
        return self.save() and changed


# ============================================================================
# Schedule helpers
# ============================================================================

def _valid_schedule(schedule: Any) -> bool:
    if not isinstance(schedule, dict):
        return False
    kind = schedule.get("kind")
    value = schedule.get("value")
    if kind == "recurring":
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if kind == "daily":
        try:
            datetime.strptime(str(value), "%H:%M")
            return True
        except (TypeError, ValueError):
            return False
    if kind == "once":
        try:
            datetime.fromisoformat(str(value))
            return True
        except (TypeError, ValueError):
            return False
    return False


# The agent invents plausible key names for a schedule the same way it does for
# actions. Live failures 2026-08-29: {"start_time":"12:39","trigger_type":
# "scheduled_time"} was rejected four times at 00:37, and "daily 00:00" four
# times at 00:08 — both unambiguous. Coerce the shape; `_valid_schedule` is
# still the gate, so a genuinely ambiguous schedule is refused, not guessed.
_SCHEDULE_KIND_KEYS = ("kind", "type", "trigger_type", "frequency",
                       "recurrence", "mode")
_SCHEDULE_VALUE_KEYS = ("value", "time", "start_time", "at", "when",
                        "datetime", "date_time", "iso", "seconds",
                        "interval_seconds", "interval")
_SCHEDULE_KIND_ALIASES = {
    "daily": "daily", "every_day": "daily", "everyday": "daily", "day": "daily",
    "scheduled_time": "daily", "time_of_day": "daily", "time": "daily",
    "once": "once", "one_time": "once", "onetime": "once", "single": "once",
    "specific_date_time": "once", "date": "once", "datetime": "once",
    "recurring": "recurring", "interval": "recurring", "repeat": "recurring",
    "every": "recurring", "periodic": "recurring",
}


def _coerce_schedule_value(value: Any) -> tuple:
    """Infer (kind, canonical_value) from the value alone, or (None, None)."""
    if isinstance(value, bool):
        return None, None
    if isinstance(value, (int, float)):
        seconds = int(value)
        return ("recurring", seconds) if seconds > 0 else (None, None)
    text = str(value or "").strip()
    if not text:
        return None, None
    for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I %p"):
        try:
            return "daily", datetime.strptime(text.upper(), fmt).strftime("%H:%M")
        except ValueError:
            pass
    try:
        datetime.fromisoformat(text)
        return "once", text
    except ValueError:
        pass
    if text.isdigit() and int(text) > 0:
        return "recurring", int(text)
    return None, None


def _normalize_schedule(schedule: Any) -> Dict[str, Any]:
    """Best-effort coercion of a spoken/loose schedule into the canonical shape."""
    kind_hint, raw_value = "", None
    if isinstance(schedule, dict):
        kind_hint = _first_str(schedule, _SCHEDULE_KIND_KEYS).lower()
        for key in _SCHEDULE_VALUE_KEYS:
            if schedule.get(key) not in (None, ""):
                raw_value = schedule[key]
                break
    elif isinstance(schedule, bool) or schedule is None:
        return {}
    elif isinstance(schedule, (str, int, float)):
        # "daily 08:00" / "once 2026-08-29T12:39:00" / a bare time or interval.
        head, _, tail = str(schedule).strip().partition(" ")
        if tail.strip() and head.lower() in _SCHEDULE_KIND_ALIASES:
            kind_hint, raw_value = head.lower(), tail.strip()
        else:
            raw_value = str(schedule).strip()
    else:
        return {}

    kind = _SCHEDULE_KIND_ALIASES.get(kind_hint, "")
    inferred, value = _coerce_schedule_value(raw_value)
    if value is None:
        return {}
    # A stated kind only wins when the value agrees with it; otherwise the value
    # is the more reliable signal (models mislabel "once" far more often than
    # they mistype an ISO timestamp).
    if kind and kind == inferred:
        return {"kind": kind, "value": value}
    return {"kind": inferred, "value": value}


# The agent writes these actions from a natural-language request, so it reaches
# for reasonable synonyms of the canonical keys. Live failure 2026-08-29
# 00:41:20: a perfectly good "Back Exercise" routine was sent five times as
# {"action": "guided_step", "data": "..."} — semantically exact, keyed wrong.
# Every action was dropped, the error said only "requires at least one valid
# action", and the agent reworded the prose it had right instead of the keys it
# had wrong until the turn budget died. Accept the synonyms; a rename is not a
# reason to lose a routine.
_ACTION_TYPE_KEYS = ("type", "action", "action_type", "kind")
_ACTION_TEXT_KEYS = ("instruction", "text", "question", "data", "content",
                     "message", "prompt", "say", "value", "description")


def _first_str(raw: Dict[str, Any], keys) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _no_valid_actions_error(problems: List[str]) -> str:
    """Build a rejection the agent can actually fix on its next turn."""
    detail = "; ".join(problems[:6]) if problems else "no actions were supplied"
    return (
        "a routine event requires at least one valid action — " + detail +
        '. Each action must be {"type": "speak|check_in|guided_step|'
        'memory_activity|play_music|observe|measure_vital|log|notify_caregiver"'
        ', "instruction": "<the words Kiki should say or do>", "needs_response":'
        " true|false}. Fix the KEY NAMES; do not reword the instruction text.")


def _normalize_routine_actions(
        actions: Any, problems: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Canonicalize routine actions, recording WHY any were rejected.

    `problems` (when given) collects one human-readable reason per dropped
    action. A validation error the model cannot act on is not validation — it
    is a retry loop, so callers surface these back to the agent verbatim.
    """
    if problems is None:
        problems = []
    if not isinstance(actions, list):
        problems.append(
            f"'actions' must be a JSON array, got {type(actions).__name__}")
        return []
    normalized = []
    for index, raw in enumerate(actions[:20]):
        if not isinstance(raw, dict):
            problems.append(f"action {index}: must be an object, got "
                            f"{type(raw).__name__}")
            continue
        action_type = (_first_str(raw, _ACTION_TYPE_KEYS) or "speak").lower()
        if action_type not in VALID_ROUTINE_ACTION_TYPES:
            problems.append(
                f"action {index}: type {action_type!r} is not one of "
                + "|".join(sorted(VALID_ROUTINE_ACTION_TYPES)))
            continue
        instruction = _first_str(raw, _ACTION_TEXT_KEYS)
        steps = [str(step).strip() for step in (raw.get("steps") or [])
                 if str(step).strip()]
        if action_type == "guided_step" and not instruction and steps:
            instruction = steps[0]
        if not instruction:
            problems.append(
                f"action {index} ({action_type}): no instruction text. Put the "
                f"words in \"instruction\". Keys received: "
                + (", ".join(sorted(raw)) or "none"))
            continue
        normalized.append({
            "type": action_type,
            "instruction": instruction,
            "needs_response": bool(raw.get(
                "needs_response",
                action_type in {"check_in", "guided_step", "measure_vital"})),
            "success_signal": str(raw.get("success_signal", "")).strip(),
            "on_concern": str(raw.get("on_concern", "")).strip(),
        })
        if action_type == "measure_vital":
            normalized[-1]["vital_type"] = str(
                raw.get("vital_type") or "heart_rate").strip().lower()
    return normalized


def _routine_turn_actions(actions: List[Dict[str, Any]], start: int) -> List[Dict[str, Any]]:
    """Return the ordered actions Kiki should conduct before waiting again."""
    batch: List[Dict[str, Any]] = []
    for action in actions[max(0, int(start)):]:
        batch.append(json.loads(json.dumps(action)))
        if action.get("needs_response"):
            break
    return batch


# ============================================================================
# Singleton
# ============================================================================

_care_plan_instance: Optional[CarePlan] = None
_singleton_lock = threading.Lock()


def get_care_plan_store() -> CarePlan:
    global _care_plan_instance
    with _singleton_lock:
        if _care_plan_instance is None:
            _care_plan_instance = CarePlan()
        return _care_plan_instance


def reload_care_plan() -> CarePlan:
    global _care_plan_instance
    with _singleton_lock:
        _care_plan_instance = CarePlan()
        return _care_plan_instance
