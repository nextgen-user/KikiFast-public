"""Mode capability flags: the gate that lets two modes share one care stack.

Before this existed, five call sites compared ``get_active_mode() == "senior"``.
Nothing under ``core/senior/`` ever read the active mode, so those five gates
were the entire coupling between the care stack and one mode name — and any new
care-capable mode would have booted with the scheduler silently doing nothing.
These tests pin the capability contract itself, not the particular modes that
happen to declare it today.
"""

import pytest

from core import runtime_controls as rc


@pytest.fixture
def modes(monkeypatch):
    """Install a synthetic mode table so the tests do not track config.json."""
    table = {
        "plain": {"system_prompt": None},
        "cares": {"capabilities": ["care"]},
        "everything": {"capabilities": ["care", "environment", "companion"]},
        "bogus": {"capabilities": ["care", "not_a_real_capability"]},
        "malformed": {"capabilities": "care"},
        "not_a_dict": "just a string",
    }
    monkeypatch.setattr(rc, "_modes", lambda: table)
    return table


def _active(monkeypatch, name):
    monkeypatch.setattr(rc, "get_active_mode", lambda: name)


def test_a_mode_without_capabilities_has_none(modes, monkeypatch):
    _active(monkeypatch, "plain")
    assert rc.get_mode_capabilities() == frozenset()
    assert not rc.mode_has_capability("care")


def test_a_care_mode_reports_only_what_it_declares(modes, monkeypatch):
    _active(monkeypatch, "cares")
    assert rc.get_mode_capabilities() == {"care"}
    assert rc.mode_has_capability("care")
    assert not rc.mode_has_capability("environment")


def test_a_mode_can_declare_several(modes, monkeypatch):
    _active(monkeypatch, "everything")
    assert rc.get_mode_capabilities() == {"care", "environment", "companion"}
    for capability in ("care", "environment", "companion"):
        assert rc.mode_has_capability(capability)


def test_an_unknown_capability_name_is_dropped(modes, monkeypatch):
    """A typo must not become a live feature switch by being truthy."""
    _active(monkeypatch, "bogus")
    assert rc.get_mode_capabilities() == {"care"}
    assert not rc.mode_has_capability("not_a_real_capability")


def test_capabilities_can_be_read_for_a_mode_that_is_not_active(modes, monkeypatch):
    _active(monkeypatch, "plain")
    assert rc.mode_has_capability("care", mode="cares")
    assert not rc.mode_has_capability("care", mode="plain")


@pytest.mark.parametrize("mode", ["malformed", "not_a_dict", "missing_entirely"])
def test_a_broken_declaration_fails_closed(modes, monkeypatch, mode):
    """Never ON by accident.

    A capability starts subsystems that speak on a schedule and email families.
    Unlike context gating -- which fails open so a config problem can never be
    what stops Kiki from answering -- an unreadable capability must leave those
    off rather than switch them on in a mode that never asked for them. Note
    ``"care"`` as a bare string is deliberately rejected rather than treated as
    a one-item list: silently accepting it would make the same typo mean
    different things in different modes.
    """
    _active(monkeypatch, mode)
    assert rc.get_mode_capabilities() == frozenset()
    assert not rc.mode_has_capability("care")


def test_capability_lookup_is_case_and_space_insensitive(modes, monkeypatch):
    _active(monkeypatch, "cares")
    assert rc.mode_has_capability("  CARE ")


def test_the_shipped_config_still_gives_senior_the_care_capability():
    """The regression this whole mechanism exists to prevent.

    Deliberately reads the real config: if someone edits config.json and drops
    the flag, senior mode boots with the care scheduler doing nothing at all,
    and the only symptom is that no routine ever fires.
    """
    from tools_and_config.config_loader import get_full_config

    shipped = get_full_config()["assistant_modes"]["modes"]
    assert "care" in rc.get_mode_capabilities("senior")
    assert "care" in rc.get_mode_capabilities("health_sih")
    assert rc.get_mode_capabilities("default") == frozenset()

    # health_sih must not quietly lose the ordinary-Kiki tools by being given a
    # narrower list than senior: an explicit per-mode main_tools is a hard
    # permission boundary (core/llm.py::_mode_tool_override), not a hint.
    assert set(shipped["health_sih"]["main_tools"]) >= set(shipped["senior"]["main_tools"])


def test_health_sih_is_reachable_by_the_names_a_person_would_say():
    for spoken in ("health", "health mode", "health companion",
                   "switch to health sih mode"):
        assert rc.resolve_mode_name(spoken) == "health_sih", spoken
    # The new mode must not have stolen the older one's name.
    assert rc.resolve_mode_name("senior citizen mode") == "senior"
