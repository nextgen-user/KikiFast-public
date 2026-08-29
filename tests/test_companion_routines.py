"""Phase H experiences are care-plan DATA, not code paths.

The architecture decision these tests defend (plan.md, "Non-negotiable"): a care
event stores a rich goal/context/session brief, not hardcoded dialogue or an
executable list of canned instructions. So the assertions below are as much
about what the briefs must NOT contain -- a script, a question list, a fixed
sequence -- as about the seeding being idempotent.
"""

import json
from pathlib import Path

import pytest

from core.senior import care_plan as care_plan_module
from core.health.companion_routines import (
    COMPANION_ROUTINES, COMPANION_SOURCE, ensure_companion_routines,
    existing_companion_keys,
)


@pytest.fixture
def plan(tmp_path, monkeypatch):
    path = tmp_path / "care_plan.json"
    path.write_text(json.dumps({
        "senior": {"name": "Test", "language": "en"},
        "family_contacts": [], "reminders": [], "exercises": [],
        "approved_music": [], "approved_topics": [], "care_log": [],
        "metadata": {}, "health_measurements": [],
        "routine_events": [], "active_session": None,
    }))
    monkeypatch.setattr(care_plan_module, "get_full_config",
                        lambda: {"senior_mode": {"care_agent": {}}}, raising=False)
    return care_plan_module.CarePlan(Path(str(path)))


# --- seeding ---

def test_seeding_installs_every_routine(plan):
    added = ensure_companion_routines(plan, {})
    assert len(added) == len(COMPANION_ROUTINES)
    assert len(plan.get_section("routine_events")) == len(COMPANION_ROUTINES)


def test_seeding_twice_adds_nothing_the_second_time(plan):
    ensure_companion_routines(plan, {})
    assert ensure_companion_routines(plan, {}) == []
    assert len(plan.get_section("routine_events")) == len(COMPANION_ROUTINES)


def test_every_routine_is_scheduled_daily_and_enabled(plan):
    ensure_companion_routines(plan, {})
    for event in plan.get_section("routine_events"):
        assert event["schedule"]["kind"] == "daily"
        assert event["enabled"] is True
        assert event["source"] == COMPANION_SOURCE


def test_times_can_be_configured(plan):
    ensure_companion_routines(plan, {"times": {"morning_briefing": "06:15"}})
    briefing = next(e for e in plan.get_section("routine_events")
                    if e["companion_key"] == "morning_briefing")
    assert briefing["schedule"]["value"] == "06:15"


def test_individual_routines_can_be_skipped(plan):
    ensure_companion_routines(plan, {"disabled": ["sleep_winddown"]})
    assert "sleep_winddown" not in existing_companion_keys(plan)
    assert "morning_briefing" in existing_companion_keys(plan)


def test_seeding_can_be_turned_off_entirely(plan):
    assert ensure_companion_routines(plan, {"seed_default_routines": False}) == []
    assert plan.get_section("routine_events") == []


# --- the person's edits win ---

def test_a_retimed_routine_is_not_duplicated(plan):
    """Identity is the companion_key, not the title or the time."""
    ensure_companion_routines(plan, {})
    briefing = next(e for e in plan.get_section("routine_events")
                    if e["companion_key"] == "morning_briefing")
    plan.edit_routine_event(briefing["id"], schedule={"kind": "daily", "value": "09:00"},
                            title="My morning chat")

    assert ensure_companion_routines(plan, {}) == []
    matches = [e for e in plan.get_section("routine_events")
               if e.get("companion_key") == "morning_briefing"]
    assert len(matches) == 1
    assert matches[0]["schedule"]["value"] == "09:00"


def test_a_disabled_routine_is_not_quietly_switched_back_on(plan):
    """An assistant that restores what you turned off is worse than one that
    never offered it."""
    ensure_companion_routines(plan, {})
    target = next(e for e in plan.get_section("routine_events")
                  if e["companion_key"] == "hydration_checkin")
    plan.edit_routine_event(target["id"], enabled=False)

    ensure_companion_routines(plan, {})
    still = next(e for e in plan.get_section("routine_events")
                 if e.get("companion_key") == "hydration_checkin")
    assert still["enabled"] is False


def test_a_deleted_routine_comes_back_only_on_a_fresh_seed(plan):
    """Deletion is not remembered -- documented, not accidental. Someone who
    wants it gone for good sets `disabled`, which IS remembered."""
    ensure_companion_routines(plan, {})
    target = next(e for e in plan.get_section("routine_events")
                  if e["companion_key"] == "sleep_winddown")
    plan.remove_routine_event(target["id"])
    assert "sleep_winddown" not in existing_companion_keys(plan)

    added = ensure_companion_routines(plan, {})
    assert [e["companion_key"] for e in added] == ["sleep_winddown"]
    assert ensure_companion_routines(plan, {"disabled": ["sleep_winddown"]}) == []


def test_unrelated_events_are_left_alone(plan):
    plan.add_routine_event(title="Blood pressure tablet", category="medicine",
                           schedule={"kind": "daily", "value": "08:00"},
                           session_brief="Their morning tablet.")
    ensure_companion_routines(plan, {})
    titles = [e["title"] for e in plan.get_section("routine_events")]
    assert "Blood pressure tablet" in titles
    assert existing_companion_keys(plan) == {r["key"] for r in COMPANION_ROUTINES}


# --- the briefs are hand-offs, not scripts ---

def test_the_plan_covers_what_phase_h_asks_for():
    keys = {r["key"] for r in COMPANION_ROUTINES}
    assert {"morning_briefing", "evening_reflection", "hydration_checkin",
            "movement_checkin", "sleep_winddown"} <= keys


def test_every_brief_is_substantial_enough_to_reason_from(plan):
    ensure_companion_routines(plan, {})
    for event in plan.get_section("routine_events"):
        assert len(event["session_brief"]) > 300, event["title"]
        assert event["objective"]


def test_no_brief_contains_a_dialogue_script():
    """The architecture decision, asserted.

    A quoted line for Kiki to read would make these canned dialogue and defeat
    the whole adaptive-session design -- and the live box is known to recite
    prompt examples verbatim.
    """
    for routine in COMPANION_ROUTINES:
        text = routine["brief"]
        assert '"' not in text, routine["key"]
        assert "Say exactly" not in text
        assert "Step 1" not in text


def test_the_briefs_that_need_live_conditions_point_at_them():
    """A briefing cannot honestly discuss heat or air quality unless it is told
    to read the block the care agent is actually given."""
    for key in ("morning_briefing", "hydration_checkin", "movement_checkin"):
        routine = next(r for r in COMPANION_ROUTINES if r["key"] == key)
        assert "CURRENT OUTSIDE CONDITIONS" in routine["brief"]


def test_the_morning_brief_forbids_inventing_a_reading():
    brief = next(r for r in COMPANION_ROUTINES
                 if r["key"] == "morning_briefing")["brief"]
    assert "never estimate" in brief.lower()


def test_the_evening_brief_insists_on_confirmed_outcomes():
    """Never infer that a scheduled medicine was actually taken."""
    brief = next(r for r in COMPANION_ROUTINES
                 if r["key"] == "evening_reflection")["brief"]
    assert "CONFIRMED" in brief
    assert "just because it was scheduled" in brief


def test_the_follow_ups_are_explicitly_non_shaming():
    for key in ("hydration_checkin", "movement_checkin", "evening_reflection"):
        brief = next(r for r in COMPANION_ROUTINES if r["key"] == key)["brief"]
        assert any(word in brief.lower()
                   for word in ("lecture", "moralise", "push", "accusation",
                                "scolded")), key


def test_no_routine_asks_for_continuous_vision(plan):
    """Filming someone through breakfast is not what these are for, and vision
    costs a frame fetch on every turn."""
    ensure_companion_routines(plan, {})
    assert all(e["continuous_vision"] is False
               for e in plan.get_section("routine_events"))


def test_the_shipped_config_agrees_with_the_routine_keys():
    from tools_and_config.config_loader import get_full_config

    times = get_full_config().get("companion", {}).get("times", {})
    assert set(times) == {r["key"] for r in COMPANION_ROUTINES}


# --- failure isolation ---

def test_one_bad_routine_does_not_stop_the_others(plan, monkeypatch):
    real = plan.add_routine_event
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("bad schedule")
        return real(*args, **kwargs)

    monkeypatch.setattr(plan, "add_routine_event", flaky)
    added = ensure_companion_routines(plan, {})
    assert len(added) == len(COMPANION_ROUTINES) - 1


def test_a_broken_plan_reads_as_no_keys_rather_than_raising():
    class Broken:
        def get_section(self, name):
            raise RuntimeError("plan down")

    assert existing_companion_keys(Broken()) == set()
