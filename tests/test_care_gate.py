"""Gate 01-03 logic tests. No camera, no NPU, no Hailo imports.

The gate is the piece that decides whether a flicker becomes an event, so its
rejection paths matter more than its accept path: every test below is about
something that must NOT fire.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_GATE_PATH = Path(os.environ.get("KIKIFAST_CARE_GATE_PATH", "care_gate.py"))
pytestmark = pytest.mark.skipif(
    not _GATE_PATH.is_file(), reason="care_gate.py lives in the un-versioned Hailo tree")


def _load_module():
    spec = importlib.util.spec_from_file_location("care_gate", _GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["care_gate"] = module
    spec.loader.exec_module(module)
    return module


care_gate = _load_module() if _GATE_PATH.is_file() else None

DRINK = "person raising a cup or bottle to their mouth"
PHONE = "person holding a phone to their ear"
IDLE = "person standing still doing nothing"
NEGATIVES = {PHONE, IDLE}


def _config(tmp_path, **overrides):
    defaults = {
        "threshold": 0.28, "margin": 0.10, "need": 3, "window": 6,
        "min_dwell_seconds": 1.0, "max_gap_seconds": 3.0,
        "refractory_seconds": 60,
    }
    defaults.update(overrides)
    path = tmp_path / "care_events.json"
    path.write_text(json.dumps({
        "enabled": True,
        "defaults": defaults,
        "events": {"drinking": {"prompt": DRINK}},
    }))
    return path


def _feed(gate, scores, times, negatives=NEGATIVES, track=1):
    fired = []
    for t in times:
        fired.extend(gate.observe(track, scores, negatives=negatives, now=t))
    return fired


def test_a_clear_sustained_action_fires_once(tmp_path):
    gate = care_gate.CareEventGate(str(_config(tmp_path)))
    scores = {DRINK: 0.70, PHONE: 0.15, IDLE: 0.15}
    fired = _feed(gate, scores, [100.0, 100.5, 101.0, 101.5])

    assert len(fired) == 1
    assert fired[0]["event"] == "drinking"
    assert fired[0]["similarity"] == pytest.approx(0.70)
    assert fired[0]["dwell_seconds"] >= 1.0


def test_a_single_strong_frame_is_not_evidence(tmp_path):
    # The whole point of persistence: one spike must never become an event.
    gate = care_gate.CareEventGate(str(_config(tmp_path)))
    assert _feed(gate, {DRINK: 0.9, PHONE: 0.05, IDLE: 0.05}, [100.0]) == []


def test_high_score_with_no_margin_is_rejected(tmp_path):
    # Two prompts neck and neck means the frame is ambiguous, however high the
    # winner scores. Threshold alone would have let this through.
    gate = care_gate.CareEventGate(str(_config(tmp_path)))
    scores = {DRINK: 0.46, PHONE: 0.44, IDLE: 0.10}
    assert _feed(gate, scores, [100.0, 100.5, 101.0, 101.5, 102.0]) == []
    assert "low-margin" in gate.snapshot()["tracks"][0]["reason"]


def test_a_winning_distractor_blocks_the_event_and_is_named(tmp_path):
    # The phone-to-ear case. It must not fire, and the panel must say what won,
    # because that points at the prompt set rather than the threshold.
    gate = care_gate.CareEventGate(str(_config(tmp_path)))
    scores = {PHONE: 0.72, DRINK: 0.20, IDLE: 0.08}
    assert _feed(gate, scores, [100.0, 100.5, 101.0, 101.5, 102.0]) == []

    track = gate.snapshot()["tracks"][0]
    assert track["won_negative"] is True
    assert "distractor" in track["reason"]
    assert "phone" in track["reason"]


def test_flicker_never_accumulates_enough_hits(tmp_path):
    gate = care_gate.CareEventGate(str(_config(tmp_path)))
    good = {DRINK: 0.70, PHONE: 0.15, IDLE: 0.15}
    bad = {IDLE: 0.70, DRINK: 0.15, PHONE: 0.15}
    fired = []
    for i in range(12):
        scores = good if i % 3 == 0 else bad
        fired.extend(gate.observe(1, scores, negatives=NEGATIVES, now=100.0 + i * 0.4))
    assert fired == []


def test_dwell_floor_rejects_a_burst_that_is_too_brief(tmp_path):
    # Enough hits, but crammed into a fraction of a second.
    gate = care_gate.CareEventGate(str(_config(tmp_path, min_dwell_seconds=5.0)))
    scores = {DRINK: 0.70, PHONE: 0.15, IDLE: 0.15}
    fired = _feed(gate, scores, [100.0, 100.1, 100.2, 100.3])
    assert fired == []
    assert "dwell" in gate.snapshot()["tracks"][0]["reason"]


def test_refractory_stops_one_action_firing_twice(tmp_path):
    gate = care_gate.CareEventGate(str(_config(tmp_path)))
    scores = {DRINK: 0.70, PHONE: 0.15, IDLE: 0.15}
    assert len(_feed(gate, scores, [100.0, 100.5, 101.0, 101.5])) == 1
    # Same action continuing must not re-fire inside the cooldown.
    assert _feed(gate, scores, [102.0, 102.5, 103.0, 103.5, 104.0]) == []
    assert "refractory" in gate.snapshot()["tracks"][0]["reason"]
    # Well past it, a fresh run is allowed again.
    assert len(_feed(gate, scores, [200.0, 200.5, 201.0, 201.5])) == 1


def test_a_long_gap_restarts_the_run_rather_than_extending_it(tmp_path):
    # Two sips minutes apart are not one long drink; dwell must not span them.
    gate = care_gate.CareEventGate(str(_config(tmp_path, need=4)))
    scores = {DRINK: 0.70, PHONE: 0.15, IDLE: 0.15}
    assert _feed(gate, scores, [100.0, 100.5]) == []
    fired = _feed(gate, scores, [400.0, 400.5, 401.0, 401.5])
    assert len(fired) == 1
    assert fired[0]["dwell_seconds"] < 3.0


def test_separate_people_accumulate_separately(tmp_path):
    gate = care_gate.CareEventGate(str(_config(tmp_path)))
    scores = {DRINK: 0.70, PHONE: 0.15, IDLE: 0.15}
    per_track = {1: [], 2: []}
    for t in [100.0, 100.5, 101.0, 101.5]:
        for track in (1, 2):
            per_track[track].extend(
                gate.observe(track, scores, negatives=NEGATIVES, now=t))

    # Evidence is accumulated per track -- neither borrows the other's.
    assert {t["track_id"] for t in gate.snapshot()["tracks"]} == {1, 2}

    # But the refractory is global per event, so two tracks doing the same
    # thing at the same moment yield ONE event, not two. That is the deliberate
    # trade: a track id is not a person identity (the tracker churns ids
    # constantly), so a per-track cooldown would just be bypassed. The cost is
    # that two genuinely different people drinking together merge into one
    # event; for a single-occupant home that is the right side of the trade.
    assert len(per_track[1]) + len(per_track[2]) == 1


def test_config_is_hot_reloaded_when_the_file_changes(tmp_path):
    path = _config(tmp_path, threshold=0.90)
    gate = care_gate.CareEventGate(str(path))
    scores = {DRINK: 0.70, PHONE: 0.15, IDLE: 0.15}
    assert _feed(gate, scores, [100.0, 100.5, 101.0, 101.5]) == []

    # Tuning while the service runs is the whole workflow; no restart allowed.
    data = json.loads(path.read_text())
    data["defaults"]["threshold"] = 0.28
    path.write_text(json.dumps(data))
    import os
    os.utime(path, (200.0, 200.0))

    assert len(_feed(gate, scores, [300.0, 300.5, 301.0, 301.5])) == 1


def test_a_corrupt_config_keeps_the_previous_rules(tmp_path):
    # Saving a half-written file mid-edit must not take the pipeline down.
    path = _config(tmp_path)
    gate = care_gate.CareEventGate(str(path))
    path.write_text("{ not json")
    import os
    os.utime(path, (200.0, 200.0))

    scores = {DRINK: 0.70, PHONE: 0.15, IDLE: 0.15}
    assert len(_feed(gate, scores, [100.0, 100.5, 101.0, 101.5])) == 1


def test_softmax_scores_match_the_matchers_own_formula(tmp_path):
    import numpy as np
    text = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    image = np.array([1.0, 0.0])
    got = care_gate.softmax_scores(text, image)

    logits = 100.0 * (text @ image)
    logits -= logits.max()
    expected = np.exp(logits) / np.exp(logits).sum()
    assert got == pytest.approx(expected.tolist())
    assert sum(got) == pytest.approx(1.0)


def test_a_continuous_action_can_fire_again_after_the_refractory(tmp_path):
    """Regression: holding the cup kept the bar full but pinned dwell at 0.

    Firing cleared run_started but left last_pass set, so the next frame never
    opened a new run -- run_started stayed None and dwell() returned 0.0 for as
    long as the action continued. Once the refractory expired the reason stuck
    at "dwell 0.0/1.0s" and the event could never fire again.
    """
    gate = care_gate.CareEventGate(str(_config(tmp_path, refractory_seconds=10)))
    scores = {DRINK: 0.70, PHONE: 0.15, IDLE: 0.15}

    first = _feed(gate, scores, [100.0, 100.5, 101.0, 101.5])
    assert len(first) == 1

    # Keep drinking straight through the cooldown, no gap anywhere. The first
    # fire lands at 101.0, so stay safely inside the 10s window.
    during = _feed(gate, scores, [102.0 + 0.5 * i for i in range(12)])
    assert during == []                                   # refractory holds

    track = gate.snapshot()["tracks"][0]
    assert track["hits"] >= track["need"]                 # evidence is full
    # The heart of the regression: dwell used to be pinned at 0.0 here, so the
    # event could never satisfy min_dwell again however long you held the cup.
    assert track["dwell"] > 1.0, "dwell must advance while the action continues"

    # Past the cooldown, still drinking: it must be able to fire again.
    again = _feed(gate, scores, [113.0 + 0.5 * i for i in range(6)])
    assert len(again) == 1, "a continuing action can never fire again"
    assert again[0]["dwell_seconds"] >= 1.0


def test_refractory_holds_across_track_id_churn(tmp_path):
    """Regression: the tracker reassigns ids constantly (96 ids in ~10 min).

    A cooldown keyed only on (track, event) was bypassed every time the same
    person was re-acquired -- observed live as two `unsteady` fires one second
    apart and two `drinking` fires in the same second.
    """
    gate = care_gate.CareEventGate(str(_config(tmp_path)))
    scores = {DRINK: 0.70, PHONE: 0.15, IDLE: 0.15}

    assert len(_feed(gate, scores, [100.0, 100.5, 101.0, 101.5], track=51)) == 1
    # Same person, one second later, new track id.
    assert _feed(gate, scores, [102.0, 102.5, 103.0, 103.5], track=78) == []
