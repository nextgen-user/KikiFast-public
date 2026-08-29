"""The `CARE NOW` row: injected on change, silent otherwise.

Every character of this line lives in the warm KV prefix and is re-prefilled on
every later turn (Codestructure section 4), so the tests here are mostly about
what the snapshot REFUSES to say: on an ordinary turn, on an unchanged reading,
and when a source is broken.
"""

from datetime import datetime, timedelta, timezone

import pytest

from core.health.care_snapshot import (
    CareNowInjector, build_care_now, _next_due_part, _vitals_part,
)


class FakeProvider:
    def __init__(self, line=""):
        self._line = line

    def compact_line(self, now=None):
        return self._line


class FakeManager:
    def __init__(self, workers=None, scheduled=True):
        self._workers = workers or []
        self._scheduled = scheduled

    def schedule_receipt(self, item_id=""):
        return {"scheduled": self._scheduled and bool(self._workers),
                "workers": self._workers}


class FakePlan:
    def __init__(self, measurements=None, session=None):
        self._measurements = measurements or []
        self._session = session or {"status": "none"}

    def get_section(self, name):
        return self._measurements if name == "health_measurements" else None

    def care_session_state(self):
        return self._session


def _worker(name, minutes_ahead):
    return {"worker_name": name,
            "next_trigger_at": (datetime.now()
                                + timedelta(minutes=minutes_ahead)).isoformat()}


def _reading(bpm=72, hours_ago=1.0):
    return {"measurement": "heart_rate", "value": bpm, "unit": "bpm",
            "measured_at": (datetime.now(timezone.utc)
                            - timedelta(hours=hours_ago)).isoformat()}


# --- silence is the default ---

def test_an_ordinary_turn_says_nothing():
    assert build_care_now(FakeProvider(""), FakeManager(), FakePlan()) == ""


def test_no_sources_at_all_says_nothing():
    assert build_care_now() == ""


# --- assembly ---

def test_the_environment_line_is_carried_through():
    line = build_care_now(FakeProvider("OUTSIDE Delhi: AQI ~301 very poor"))
    assert line.startswith("CARE NOW: ")
    assert "AQI ~301 very poor" in line


def test_an_upcoming_item_is_announced_with_minutes():
    line = build_care_now(manager=FakeManager([_worker("senior:medicine:m1", 20)]))
    assert "NEXT medicine in 20min" in line


def test_the_soonest_item_wins():
    line = build_care_now(manager=FakeManager([
        _worker("senior:exercise:e1", 60),
        _worker("senior:medicine:m1", 10),
    ]))
    assert "medicine" in line
    assert "exercise" not in line


def test_a_distant_item_is_not_news():
    """Beyond the horizon it is not "now", and it would cost prefix every turn."""
    assert build_care_now(manager=FakeManager([_worker("senior:medicine:m1", 300)])) == ""


def test_an_item_already_past_is_ignored():
    assert build_care_now(manager=FakeManager([_worker("senior:medicine:m1", -30)])) == ""


def test_nothing_scheduled_says_nothing():
    assert _next_due_part(FakeManager(scheduled=False)) == ""
    assert _next_due_part(None) == ""


def test_a_recent_trusted_reading_is_included():
    line = build_care_now(plan=FakePlan(measurements=[_reading(72, hours_ago=2)]))
    assert "LAST heart_rate 72bpm" in line


def test_a_stale_reading_belongs_to_a_trend_not_to_now():
    assert _vitals_part(FakePlan(measurements=[_reading(72, hours_ago=30)])) == ""


def test_no_readings_at_all_is_silent():
    assert _vitals_part(FakePlan(measurements=[])) == ""


def test_an_active_session_is_flagged():
    line = build_care_now(plan=FakePlan(
        session={"status": "active", "event_title": "Neck Routine"}))
    assert "IN SESSION: Neck Routine" in line


def test_the_parts_are_joined_into_one_row():
    line = build_care_now(
        FakeProvider("OUTSIDE Delhi: AQI ~301 very poor"),
        FakeManager([_worker("senior:medicine:m1", 15)]),
        FakePlan(measurements=[_reading(80, hours_ago=1)]))
    assert line.count("CARE NOW:") == 1
    assert line.count("|") == 2


def test_the_line_stays_short():
    """The 80-token target from the plan, enforced as a hard character cap."""
    line = build_care_now(
        FakeProvider("OUTSIDE " + "x" * 500),
        FakeManager([_worker("senior:medicine:m1", 15)]),
        FakePlan(measurements=[_reading(80)]))
    assert len(line) <= 320


# --- a broken source must not cost the others ---

class Exploding:
    def compact_line(self, now=None):
        raise RuntimeError("provider down")

    def schedule_receipt(self, item_id=""):
        raise RuntimeError("manager down")

    def get_section(self, name):
        raise RuntimeError("plan down")

    def care_session_state(self):
        raise RuntimeError("plan down")


def test_a_broken_care_plan_does_not_cost_the_air_quality_warning():
    line = build_care_now(FakeProvider("OUTSIDE Delhi: AQI ~301 very poor"),
                          Exploding(), Exploding())
    assert "AQI ~301 very poor" in line


def test_a_broken_provider_does_not_cost_the_schedule():
    line = build_care_now(Exploding(), FakeManager([_worker("senior:medicine:m1", 10)]))
    assert "NEXT medicine" in line


def test_everything_broken_is_silent_rather_than_raising():
    assert build_care_now(Exploding(), Exploding(), Exploding()) == ""


# --- injection policy ---

def test_a_changed_line_is_injected():
    injector = CareNowInjector(cooldown_seconds=900)
    history = []
    assert injector.maybe_inject(
        history, provider=FakeProvider("OUTSIDE Delhi: AQI ~301 very poor"))
    assert len(history) == 1
    assert history[0]["role"] == "system"


def test_the_same_line_is_never_repeated():
    """The nagging-and-prompt-growth failure this exists to prevent."""
    injector = CareNowInjector(cooldown_seconds=900)
    history = []
    source = {"provider": FakeProvider("OUTSIDE Delhi: AQI ~301 very poor")}
    for _ in range(20):
        injector.maybe_inject(history, **source)
    assert len(history) == 1


def test_an_empty_snapshot_is_never_injected():
    injector = CareNowInjector(cooldown_seconds=0)
    history = []
    injector.maybe_inject(history, provider=FakeProvider(""))
    assert history == []


def test_a_changed_line_still_waits_out_the_cooldown():
    """A reading oscillating across a band edge must not inject every turn."""
    injector = CareNowInjector(cooldown_seconds=900)
    assert injector.should_inject("first", now=1000.0) is True
    injector.mark_injected("first", now=1000.0)
    assert injector.should_inject("second", now=1100.0) is False
    assert injector.should_inject("second", now=1000.0 + 901) is True


def test_the_very_first_line_does_not_wait():
    """Cooldown gates repeats, not the first thing Kiki ever learns."""
    injector = CareNowInjector(cooldown_seconds=900)
    assert injector.should_inject("first", now=0.0) is True


def test_injection_never_raises_on_a_broken_source():
    injector = CareNowInjector(cooldown_seconds=0)
    history = []
    assert injector.maybe_inject(history, provider=Exploding(),
                                 manager=Exploding(), plan=Exploding()) == ""
    assert history == []


def test_the_row_is_appended_and_never_rewritten():
    """Append-only keeps the warm prefix a valid byte-prefix (KV rule 2).

    Retracting the row would invalidate the cache and cost a full reprefill on
    the next voice turn -- far more expensive than the row itself.
    """
    injector = CareNowInjector(cooldown_seconds=0)
    history = [{"role": "user", "content": "hi"}]
    injector.maybe_inject(history, provider=FakeProvider("OUTSIDE a"))
    injector.maybe_inject(history, provider=FakeProvider("OUTSIDE b"))
    assert [row["content"] for row in history][:1] == ["hi"]
    assert len(history) == 3
