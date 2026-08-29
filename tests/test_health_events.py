"""Gates 04-05: adjudication and the decision to speak.

The point of gate 04 is to reject, so most of these assert that something does
NOT become an event. The single most important one is
``test_a_leading_description_does_not_confirm_by_itself`` -- a vision model that
merely mentions a cup must not be able to manufacture a drink.
"""

import time

import pytest

from core.senior import health_events as he


# ------------------------------------------------------------- entailment
def test_a_matching_description_confirms():
    assert he._looks_like("The person is raising a mug to their mouth and sipping.",
                          "drinking") == "YES"


def test_an_object_present_but_not_at_the_mouth_is_unclear():
    # A cup on the table is not a sip. Unclear, never yes.
    assert he._looks_like("A person sits at a table with a cup in front of them.",
                          "drinking") == "UNCLEAR"


def test_an_unrelated_description_is_rejected():
    assert he._looks_like("The person is typing on a laptop.", "drinking") == "NO"


def test_person_presence_is_no_longer_a_verdict(monkeypatch):
    """Gate 04 does not re-detect people; CLIP's own detector already did.

    A second person-test here was not verification -- it was a worse detector
    looking at the scene later in time, and it rejected three true events in one
    session (one at similarity 0.86). Absence of EVIDENCE still rejects; absence
    of a stated person does not.
    """
    assert he._looks_like("", "eating") == "NO"
    # No person mentioned anywhere, but the evidence for the activity is there.
    assert he._looks_like("A hand is raising a mug towards a mouth.",
                          "drinking") == "YES"
    assert "person" not in he._ADJUDICATION_PROMPT.casefold()


def test_an_unknown_activity_never_confirms():
    assert he._looks_like("doing something", "telekinesis") == "UNCLEAR"


# ------------------------------------------------------------- adjudication
class _Vision:
    """Stands in for the Groq VLM; records how many frames were asked for."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def describe(self, _frame, question=None, max_tokens=None):
        self.calls += 1
        return self.answers[min(self.calls - 1, len(self.answers) - 1)]


@pytest.fixture
def patched(monkeypatch):
    def install(answers):
        vision = _Vision(answers)
        module = type(sys)("core.vision.instant_vision")
        module.capture_best_frame_b64 = lambda cfg=None: "ZmFrZQ=="
        module.describe_image_b64 = vision.describe
        monkeypatch.setitem(sys.modules, "core.vision.instant_vision", module)
        monkeypatch.setattr(he.time, "sleep", lambda _s: None)
        return vision
    return install


import sys  # noqa: E402  (used by the fixture above)


def test_two_agreeing_frames_confirm(patched):
    vision = patched(["He is lifting a glass of water to his mouth.",
                      "He is drinking from a glass."])
    consumer = he.HealthEventConsumer(config={"frames_required": 2})
    verdict, _ = consumer.adjudicate("drinking")
    assert verdict == "YES"
    assert vision.calls == 2


def test_a_disagreeing_second_frame_rejects(patched):
    # Motion blur or an odd angle on frame one is exactly what this catches.
    vision = patched(["He is lifting a glass to his mouth.",
                      "He is scratching his chin."])
    consumer = he.HealthEventConsumer(config={"frames_required": 2})
    verdict, _ = consumer.adjudicate("drinking")
    assert verdict == "NO"
    assert vision.calls == 2


def test_vision_failure_fails_closed(monkeypatch):
    module = type(sys)("core.vision.instant_vision")
    module.capture_best_frame_b64 = lambda cfg=None: "ZmFrZQ=="

    def boom(*_a, **_k):
        raise RuntimeError("offline")
    module.describe_image_b64 = boom
    monkeypatch.setitem(sys.modules, "core.vision.instant_vision", module)

    consumer = he.HealthEventConsumer(config={"frames_required": 1})
    # Offline must never mean "assume yes".
    assert consumer.adjudicate("drinking")[0] == "UNCLEAR"


# ----------------------------------------------------------- gate 05 / speak
class _Plan:
    def __init__(self):
        self.entries = []

    def add_care_log(self, kind, text):
        self.entries.append((kind, text))


def test_a_confirmed_event_is_logged_and_stays_silent(patched):
    patched(["He is drinking from a mug at his mouth."] * 2)
    plan, spoken = _Plan(), []
    consumer = he.HealthEventConsumer(
        config={"frames_required": 1, "speak_for": ["medicine"]},
        care_plan=plan, speak=lambda a, r: spoken.append(a))

    record = consumer.handle({"event": "care_activity", "activity": "drinking"})

    assert record is not None
    assert plan.entries and plan.entries[0][0] == "observation"
    assert spoken == []                      # logging is the default outcome


def test_only_named_activities_are_allowed_to_speak(patched):
    patched(["He is holding a pill bottle."] * 2)
    spoken = []
    consumer = he.HealthEventConsumer(
        config={"frames_required": 1, "speak_for": ["medicine"]},
        care_plan=_Plan(), speak=lambda a, r: spoken.append(a))

    consumer.handle({"event": "care_activity", "activity": "medicine"})
    assert spoken == ["medicine"]


def test_a_rejected_event_is_never_logged_or_spoken(patched):
    patched(["He is watching television."] * 2)
    plan, spoken = _Plan(), []
    consumer = he.HealthEventConsumer(
        config={"frames_required": 1, "speak_for": ["drinking"]},
        care_plan=plan, speak=lambda a, r: spoken.append(a))

    assert consumer.handle({"event": "care_activity", "activity": "drinking"}) is None
    assert plan.entries == []
    assert spoken == []
    assert consumer.stats["rejected"] == 1


def test_the_speaking_budget_is_enforced(patched):
    patched(["He is holding a pill bottle."] * 2)
    spoken = []
    consumer = he.HealthEventConsumer(
        config={"frames_required": 1, "speak_for": ["medicine"],
                "max_spoken_per_window": 1, "speak_window_seconds": 3600},
        care_plan=_Plan(), speak=lambda a, r: spoken.append(a))

    consumer.handle({"event": "care_activity", "activity": "medicine"})
    consumer.handle({"event": "care_activity", "activity": "medicine"})
    # Still logged twice, but Kiki only says it once.
    assert spoken == ["medicine"]
    assert consumer.stats["confirmed"] == 2


def test_the_budget_frees_up_once_the_window_passes(patched):
    patched(["He is holding a pill bottle."] * 2)
    spoken = []
    consumer = he.HealthEventConsumer(
        config={"frames_required": 1, "speak_for": ["medicine"],
                "max_spoken_per_window": 1, "speak_window_seconds": 60},
        care_plan=_Plan(), speak=lambda a, r: spoken.append(a))

    consumer.handle({"event": "care_activity", "activity": "medicine"})
    consumer._spoken_at = [time.time() - 120]     # the window has rolled over
    consumer.handle({"event": "care_activity", "activity": "medicine"})
    assert spoken == ["medicine", "medicine"]


def test_a_silent_event_still_reaches_kikis_context(patched):
    patched(["He is drinking from a mug at his mouth."] * 2)
    injected = []
    consumer = he.HealthEventConsumer(
        config={"frames_required": 1, "speak_for": []},
        care_plan=_Plan(), inject=injected.append)

    consumer.handle({"event": "care_activity", "activity": "drinking"})
    assert len(injected) == 1
    assert "drinking" in injected[0]
    # Phrased like a face event, not as a terse tag.
    assert injected[0].startswith("[System:")
    assert "mug" in injected[0]                    # carries what was actually seen


def test_injection_is_rate_limited(patched):
    # A repeating event fired nine times in ten minutes during live testing; an
    # uncapped injection would push the real conversation out of the window.
    patched(["He is holding a walking stick in his right hand."] * 2)
    injected = []
    consumer = he.HealthEventConsumer(
        config={"frames_required": 1, "speak_for": [],
                "max_injections_per_5min": 2, "inject_window_seconds": 300},
        care_plan=_Plan(), inject=injected.append)

    for _ in range(5):
        consumer.handle({"event": "care_activity", "activity": "walking_aid"})

    assert len(injected) == 2
    # Still logged every time -- only the prompt is capped, not the record.
    assert consumer.stats["confirmed"] == 5
    assert consumer.stats["injected"] == 2


def test_injection_frees_up_once_the_window_rolls_over(patched):
    patched(["He is holding a walking stick in his right hand."] * 2)
    injected = []
    consumer = he.HealthEventConsumer(
        config={"frames_required": 1, "speak_for": [],
                "max_injections_per_5min": 1, "inject_window_seconds": 300},
        care_plan=_Plan(), inject=injected.append)

    consumer.handle({"event": "care_activity", "activity": "walking_aid"})
    consumer._injected_at = [time.time() - 600]
    consumer.handle({"event": "care_activity", "activity": "walking_aid"})
    assert len(injected) == 2


def test_a_mode_can_suppress_care_injection(patched, monkeypatch):
    patched(["He is drinking from a mug at his mouth."] * 2)
    module = type(sys)("core.runtime_controls")
    module.context_enabled = lambda source: source != "care_events"
    monkeypatch.setitem(sys.modules, "core.runtime_controls", module)

    injected = []
    consumer = he.HealthEventConsumer(
        config={"frames_required": 1, "speak_for": []},
        care_plan=_Plan(), inject=injected.append)
    consumer.handle({"event": "care_activity", "activity": "drinking"})

    assert injected == []                       # roleplay mode stays in character
    assert consumer.stats["confirmed"] == 1     # but the care record is unaffected


def test_a_spoken_event_is_not_also_injected(patched):
    # The autonomous_vision prompt already carries the description into the
    # history; injecting again would duplicate it.
    patched(["He is holding a pill bottle."] * 2)
    injected, spoken = [], []
    consumer = he.HealthEventConsumer(
        config={"frames_required": 1, "speak_for": ["medicine"]},
        care_plan=_Plan(), speak=lambda a, r: spoken.append(a),
        inject=injected.append)

    consumer.handle({"event": "care_activity", "activity": "medicine"})
    assert spoken == ["medicine"]
    assert injected == []


# --------------------------------------------------- the new activity set
@pytest.mark.parametrize("description,activity", [
    ("The person is standing with both arms raised above their head.", "exercising"),
    ("The person is wearing a face mask over their nose and mouth.", "wearing_mask"),
    ("An elderly person is holding a walking stick in their right hand.", "walking_aid"),
    ("The person is pressing their palm against their forehead.", "head_in_hands"),
    ("The person is wrapped in a blanket on the sofa.", "wrapped_in_blanket"),
    ("The person is sitting slumped forward with their head hanging down.",
     "slumped_forward"),
])
def test_the_new_activities_confirm_on_a_matching_description(description, activity):
    assert he._looks_like(description, activity) == "YES"


@pytest.mark.parametrize("description,activity", [
    # The one the whole slumped_forward design is about: working hunched over a
    # laptop is visually near-identical, and a wrong "collapsed" reading is the
    # most alarming false positive in the set.
    ("The person is leaning forward, hunched over a laptop, typing.", "slumped_forward"),
    ("The person is bent forward reading a book on the desk.", "slumped_forward"),
    # Reaching for something is not exercise, whatever the arms look like.
    ("The person is reaching up towards a shelf with one arm raised above their head.",
     "exercising"),
    # A broom is not a walking stick.
    ("The person is holding a broom and sweeping the floor.", "walking_aid"),
    # A coat is not a blanket.
    ("The person is wearing a thick jacket indoors.", "wrapped_in_blanket"),
])
def test_a_named_disqualifier_rejects_even_when_the_shape_matches(description, activity):
    assert he._looks_like(description, activity) == "NO"


def test_unsteady_is_gone():
    # Demoted to a CLIP negative after firing 9 times in 10 minutes on a
    # healthy person. Nothing downstream should still confirm it.
    assert "unsteady" not in he._ENTAILMENT
    assert he._looks_like("The person is leaning on the wall.", "unsteady") == "UNCLEAR"


# ------------------------------------------------- free scene context
def test_a_rejected_candidate_still_donates_its_scene_description(patched):
    # Gate 04 already paid for this VLM look at the room. Throwing the
    # description away just because CLIP guessed wrong wastes it.
    patched(["The person is sitting at a desk reading a newspaper."] * 2)
    scenes = []
    consumer = he.HealthEventConsumer(
        config={"frames_required": 1, "speak_for": []},
        care_plan=_Plan(), see=scenes.append)

    assert consumer.handle({"event": "care_activity", "activity": "drinking"}) is None
    assert scenes == ["The person is sitting at a desk reading a newspaper."]
    assert consumer.stats["rejected"] == 1
    assert consumer.stats["scenes"] == 1


def test_a_confirmed_event_does_not_also_donate_a_scene(patched):
    # Its own care line already carries the description; a scene note as well
    # would put the same sentence into the prompt twice.
    patched(["He is lifting a glass of water to his mouth."] * 2)
    scenes, injected = [], []
    consumer = he.HealthEventConsumer(
        config={"frames_required": 1, "speak_for": []},
        care_plan=_Plan(), see=scenes.append, inject=injected.append)

    consumer.handle({"event": "care_activity", "activity": "drinking"})
    assert len(injected) == 1
    assert scenes == []


def test_an_empty_description_is_not_worth_injecting(patched):
    patched([""] * 2)
    scenes = []
    consumer = he.HealthEventConsumer(
        config={"frames_required": 1, "speak_for": []},
        care_plan=_Plan(), see=scenes.append)

    consumer.handle({"event": "care_activity", "activity": "drinking"})
    assert scenes == []


def test_scene_injection_has_its_own_budget(patched):
    # A burst of rejected candidates must not crowd out real care observations,
    # which are rate-limited separately.
    patched(["The person is watching television."] * 2)
    scenes = []
    consumer = he.HealthEventConsumer(
        config={"frames_required": 1, "speak_for": [],
                "max_scene_injections_per_5min": 2},
        care_plan=_Plan(), see=scenes.append)

    for _ in range(6):
        consumer.handle({"event": "care_activity", "activity": "drinking"})
    assert len(scenes) == 2
    assert consumer.stats["rejected"] == 6      # all still recorded


def test_a_failing_scene_injector_never_breaks_the_care_path(patched):
    patched(["The person is watching television."] * 2)

    def boom(_text):
        raise RuntimeError("history locked")

    consumer = he.HealthEventConsumer(
        config={"frames_required": 1, "speak_for": []},
        care_plan=_Plan(), see=boom)
    consumer.handle({"event": "care_activity", "activity": "drinking"})
    assert consumer.stats["rejected"] == 1
    assert consumer.stats["scenes"] == 0


# ------------------------------------------- the frame the event fired on
def test_the_attached_detection_frame_is_what_gets_judged(patched, monkeypatch):
    """The whole fix: judge the moment that fired, not a later one.

    Capturing fresh looked at the room 1-3s after the event (ZMQ hop, HTTP
    fetch, inter-frame sleep) -- long enough for the cup to go back on the
    table. Three true detections died that way.
    """
    seen = []
    vision = _Vision(["A hand is lifting a glass towards a mouth."])
    module = type(sys)("core.vision.instant_vision")
    module.capture_best_frame_b64 = lambda cfg=None: "LIVE-FRAME"
    def describe(frame, question=None, max_tokens=None):
        seen.append(frame)
        return vision.describe(frame, question, max_tokens)
    module.describe_image_b64 = describe
    monkeypatch.setitem(sys.modules, "core.vision.instant_vision", module)
    monkeypatch.setattr(he.time, "sleep", lambda _s: None)

    consumer = he.HealthEventConsumer(
        config={"frames_required": 2, "speak_for": []}, care_plan=_Plan())
    record = consumer.handle({
        "event": "care_activity", "activity": "drinking",
        "image_b64": "FIRE-MOMENT-FRAME",
    })

    assert record is not None
    assert seen == ["FIRE-MOMENT-FRAME"]
    # One look, not two: a later capture is a different moment, not a second
    # opinion, so two-frame agreement is meaningless once we hold the real one.
    assert len(seen) == 1


def test_an_event_with_no_attached_frame_still_falls_back_to_live_capture(
        patched, monkeypatch):
    seen = []
    module = type(sys)("core.vision.instant_vision")
    module.capture_best_frame_b64 = lambda cfg=None: "LIVE-FRAME"
    def describe(frame, question=None, max_tokens=None):
        seen.append(frame)
        return "A hand is lifting a glass towards a mouth."
    module.describe_image_b64 = describe
    monkeypatch.setitem(sys.modules, "core.vision.instant_vision", module)
    monkeypatch.setattr(he.time, "sleep", lambda _s: None)
    monkeypatch.setattr(he.HealthEventConsumer, "_clean_frame_b64",
                        lambda self: "LIVE-FRAME")

    consumer = he.HealthEventConsumer(
        config={"frames_required": 2, "speak_for": []}, care_plan=_Plan())
    consumer.handle({"event": "care_activity", "activity": "drinking"})

    # No attached frame, so the two-frame blur guard still applies.
    assert seen == ["LIVE-FRAME", "LIVE-FRAME"]
