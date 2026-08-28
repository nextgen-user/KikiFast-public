"""Care Plan store for Senior Citizen Mode.

A small JSON-backed store (mirrors ``core/brain/knowledge_base.py``) holding the
caregiver-defined care plan: the senior's profile, family contacts, scheduled
reminders (medicine / hydration / meal / appointment / exercise), guided
exercise routines, family-approved music/topics, and a rolling ``care_log`` of
what actually happened during the day (used for the daily caregiver summary).

Voice-first: Kiki creates and edits this file from the caregiver's/senior's
spoken instructions through the ``update_care_plan`` tool, but it is also plain
JSON that a caregiver can hand-edit or a Web-UI can render.

Saves are atomic (tmp + ``os.replace``) so a crash mid-write never corrupts it.
"""

import json
import os
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
    "approved_music": [],       # ["song / query", ...]
    "approved_topics": [],      # ["cricket", "old bollywood", ...]
    "care_log": [],             # {ts, kind, text}
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


def _normalize_schedule(schedule: Any) -> Dict[str, Any]:
    """Best-effort coercion of a spoken/loose schedule into the canonical shape."""
    if not isinstance(schedule, dict):
        return {}
    kind = str(schedule.get("kind", "")).strip().lower()
    value = schedule.get("value")
    if kind == "recurring":
        if isinstance(value, bool):
            return {}
        try:
            value = int(value)
        except (TypeError, ValueError):
            return {}
    elif kind == "daily":
        value = str(value or "").strip()
        try:
            value = datetime.strptime(value, "%H:%M").strftime("%H:%M")
        except ValueError:
            return {}
    elif kind == "once":
        value = str(value or "").strip()
        try:
            datetime.fromisoformat(value)
        except ValueError:
            return {}
    else:
        return {}
    return {"kind": kind, "value": value}


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
