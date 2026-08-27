"""Cap exemption: the only way to make a category genuinely unlimited.

Zeroing a per_category row is NOT enough — cloud_budget checks the GLOBAL
windows first and they span every category, so the global cap silently
overrides it. `summary` is exempt because a denied summarization is permanent
memory loss: the conversation is compacted WITHOUT one.
"""

import pytest

from core import cloud_budget


@pytest.fixture
def budget(monkeypatch):
    cfg = {
        "cloud_limits": {
            "enabled": True,
            "kill_switch": False,
            "max_calls_per_hour": 5,
            "max_calls_per_day": 0,
            "exempt_categories": ["summary"],
            "per_category": {},
        }
    }
    monkeypatch.setattr(cloud_budget, "get_full_config", lambda: cfg)
    return cloud_budget.CloudBudget(), cfg


def test_exempt_category_is_never_denied(budget):
    b, _ = budget
    assert all(b.allow("summary")[0] for _ in range(50))


def test_exempt_calls_do_not_consume_the_global_budget(budget):
    """Otherwise 'unlimited summaries' would quietly starve the idle mind."""
    b, _ = budget
    for _ in range(50):
        b.allow("summary")

    assert b.allow("idle_mind")[0] is True


def test_non_exempt_categories_still_hit_the_global_cap(budget):
    b, _ = budget
    allowed = sum(b.allow("vision")[0] for _ in range(20))

    assert allowed == 5


def test_kill_switch_still_beats_exemption(budget):
    """The one-button OFF must remain absolute."""
    b, cfg = budget
    cfg["cloud_limits"]["kill_switch"] = True

    ok, reason = b.allow("summary")

    assert ok is False
    assert "kill switch" in reason


def test_snapshot_separates_counted_from_total(budget):
    b, _ = budget
    for _ in range(9):
        b.allow("summary")
    b.allow("vision")

    snap = b.snapshot()

    assert snap["global"]["last_hour"] == 10          # everything that happened
    assert snap["global"]["counted_last_hour"] == 1   # what the cap sees
    assert snap["categories"]["summary"]["exempt"] is True
    assert snap["categories"]["vision"]["exempt"] is False
    assert snap["exempt_categories"] == ["summary"]


def test_a_comma_separated_string_is_tolerated(budget):
    b, cfg = budget
    cfg["cloud_limits"]["exempt_categories"] = "summary, vision"

    assert b.allow("summary")[0] and b.allow("vision")[0]


def test_no_exemptions_configured_behaves_exactly_as_before(budget):
    b, cfg = budget
    cfg["cloud_limits"].pop("exempt_categories")

    assert sum(b.allow("summary")[0] for _ in range(20)) == 5
