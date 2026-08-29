"""Gates 04-05 of the care cascade: adjudicate, then decide whether to speak.

The Hailo process (``care_gate.py``) has already applied threshold, margin,
persistence and dwell, so what arrives here is a handful of defensible
candidates an hour rather than thirty frames a second. Two things still have to
happen before Kiki opens its mouth:

  gate 04  a vision model looks at the actual frame and either confirms the
           activity or throws it out. CLIP matched a sentence to a crop; that
           is not the same as knowing what happened.

  gate 05  a confirmed activity is checked for whether it is worth *saying*.
           Almost always it is not: the default outcome of this whole pipeline
           is a line in the care log, not speech.

Gate 04 is easy to build so badly that it rubber-stamps everything, so the
adjudication deliberately never asks a leading question. It asks what is
happening, open-ended, and then tests whether that free answer entails the
hypothesis. A model asked "is she drinking?" says yes far too often.

What it does NOT do is re-check that a person is present. CLIP only produces a
crop when its own detector found one, so a second person-test here is not
verification -- it is a worse detector looking at the scene from further away
in time. It rejected three true events in one session.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Callable, Dict, List, Optional

try:
    import zmq
except Exception:                                    # pragma: no cover
    zmq = None

DEFAULT_EVENT_PORT = 5556

# Verdict vocabulary. UNCLEAR is first-class and counts as a rejection: a
# vision model that cannot tell must not be able to create an event.
_YES, _NO, _UNCLEAR = "YES", "NO", "UNCLEAR"

# Deliberately says nothing about a person being present. CLIP only produces a
# crop when its detector already found one, so re-testing for a person here was
# never verification -- it was a second detector with a worse view, and it threw
# out three true events in a single session. The question is what is HAPPENING;
# whether someone is in shot is not this gate's business.
_ADJUDICATION_PROMPT = (
    "Look at this photo. In one short sentence, describe what is happening: "
    "what is being held, worn, or done with hands, body and mouth. Describe "
    "only what you can actually see."
)

# What the open-ended description has to contain for each activity to stand.
# Deliberately concrete: these are checked against the model's own words, so
# they describe appearances, not intentions.
_ENTAILMENT: Dict[str, Dict[str, tuple]] = {
    "drinking": {
        "need": ("drink", "sip", "cup", "mug", "glass", "bottle", "straw"),
        "near": ("mouth", "lips", "drinking"),
        "not": (),
    },
    "eating": {
        "need": ("eat", "food", "spoon", "fork", "chewing", "meal", "plate", "bite"),
        "near": ("mouth", "eating", "chewing"),
        "not": (),
    },
    "medicine": {
        "need": ("pill", "medicine", "tablet", "capsule", "blister", "medication"),
        "near": (),
        "not": (),
    },
    "sleeping": {
        "need": ("sleep", "asleep", "lying", "lie", "resting", "eyes closed", "reclin"),
        "near": (),
        "not": (),
    },
    "heat_distress": {
        "need": ("fan", "fanning", "sweat", "wiping", "perspir", "hot"),
        "near": (),
        "not": (),
    },
    "exercising": {
        "need": ("arms raised", "raising their arms", "arms above", "hands above",
                 "overhead", "above their head", "stretch", "exercis"),
        "near": (),
        # Reaching for something is not exercise, whatever the arms look like.
        "not": ("shelf", "cupboard", "reaching for", "cabinet"),
    },
    "wearing_mask": {
        "need": ("mask", "covering their nose", "face covering"),
        "near": (),
        "not": (),
    },
    "walking_aid": {
        "need": ("walking stick", "cane", "walker", "walking aid", "crutch"),
        "near": (),
        "not": ("broom", "mop", "umbrella"),
    },
    "head_in_hands": {
        "need": ("forehead", "temple", "head in their hands", "holding their head",
                 "hand on their head", "rubbing their head"),
        "near": (),
        "not": (),
    },
    "wrapped_in_blanket": {
        "need": ("blanket", "shawl", "quilt", "wrapped in"),
        "near": (),
        "not": ("jacket", "coat"),
    },
    "slumped_forward": {
        "need": ("slump", "hunch", "head down", "head hanging", "bent forward",
                 "leaning forward", "drooping"),
        "near": (),
        # The whole difficulty of this event: working hunched over a laptop is
        # visually near-identical to slumping. CLIP has a negative for it; this
        # is the second line of defence, because a wrong "collapsed" reading is
        # the most alarming false positive in the set.
        "not": ("laptop", "computer", "desk", "keyboard", "screen", "monitor",
                "typing", "writing", "reading", "phone", "book"),
    },
}


def _looks_like(description: str, activity: str) -> str:
    """Does the model's own description entail the activity? Never leading."""
    rule = _ENTAILMENT.get(activity)
    if rule is None:
        return _UNCLEAR
    text = (description or "").casefold()
    if not text:
        return _NO
    if any(token in text for token in rule.get("not", ())):
        # An explicit disqualifier the model named itself -- "hunched over a
        # laptop" is not slumping, however much it looks like it.
        return _NO
    if not any(token in text for token in rule["need"]):
        return _NO
    if rule["near"] and not any(token in text for token in rule["near"]):
        # The object is there but not where it would have to be for the
        # activity to be happening -- a cup on the table is not a sip.
        return _UNCLEAR
    return _YES


class HealthEventConsumer:
    """Subscribes to care_activity, adjudicates, logs, and rarely speaks."""

    def __init__(self, config: Optional[dict] = None,
                 care_plan=None,
                 speak: Optional[Callable[[str, dict], None]] = None,
                 inject: Optional[Callable[[str], None]] = None,
                 see: Optional[Callable[[str], None]] = None,
                 host: str = "127.0.0.1",
                 port: int = DEFAULT_EVENT_PORT,
                 recorder=None):
        self.config = config or {}
        self.care_plan = care_plan
        self._speak = speak
        self._inject = inject
        self._see = see
        self.host = host
        self.port = port
        self._recorder = recorder          # injectable so tests stay off the real feed
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._spoken_at: List[float] = []
        self._injected_at: List[float] = []
        self._scene_at: List[float] = []
        self.stats = {"received": 0, "confirmed": 0, "rejected": 0,
                      "spoken": 0, "injected": 0, "scenes": 0}

    # ------------------------------------------------------------- lifecycle
    def start(self) -> bool:
        if zmq is None:
            print("[Care] pyzmq unavailable; health events disabled")
            return False
        if not self.config.get("enabled", True):
            print("[Care] health events disabled in config")
            return False
        self._thread = threading.Thread(
            target=self._run, name="care-health-events", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        context = zmq.Context.instance()
        socket = context.socket(zmq.SUB)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.setsockopt(zmq.RCVTIMEO, 1000)
        socket.connect(f"tcp://{self.host}:{self.port}")
        print(f"[Care] health events listening on {self.host}:{self.port}")
        while not self._stop.is_set():
            try:
                raw = socket.recv_string()
            except Exception:
                continue                                  # RCVTIMEO tick
            try:
                event = json.loads(raw)
            except Exception:
                continue
            if event.get("event") != "care_activity":
                continue
            try:
                self.handle(event)
            except Exception as exc:
                print(f"[Care] handler error: {exc}")
        try:
            socket.close(0)
        except Exception:
            pass

    def _record(self, activity: str, outcome: str, **meta) -> None:
        """Publish the decision to the WebUI feed (§5.23).

        Every candidate is recorded, including the ones the vision model threw
        out -- a rejection is the interesting row when you are working out why
        Kiki stayed quiet. Never allowed to raise: observability is a
        convenience, not a dependency of the care path.
        """
        try:
            recorder = self._recorder
            if recorder is None:
                from core.observability import get_recorder
                recorder = self._recorder = get_recorder()
            recorder.record("care_event", name=activity,
                            outcome=outcome, **meta)
        except Exception:
            pass

    # ------------------------------------------------------------ gates 04-05
    def handle(self, event: dict) -> Optional[dict]:
        activity = str(event.get("activity", ""))
        if not activity:
            return None
        self.stats["received"] += 1

        verdict, description = self.adjudicate(
            activity, image_b64=event.get("image_b64") or None)
        if verdict != _YES:
            self.stats["rejected"] += 1
            print(f"[Care] {activity} rejected by vision ({verdict}): "
                  f"{description[:90]!r}")
            self._record(activity, "rejected", verdict=verdict,
                         saw=description[:160],
                         similarity=event.get("similarity"),
                         track_id=event.get("track_id"))
            # The candidate was not what CLIP guessed, but the vision model
            # still just looked at the room and said what it saw -- and that
            # call is already paid for. Keeping it is free situational
            # awareness that was previously thrown away. Confirmed events do
            # NOT come through here: their own care line already carries the
            # description, so this would only duplicate it.
            self._offer_scene(description)
            return None

        self.stats["confirmed"] += 1
        record = {
            "activity": activity,
            "at": time.time(),
            "similarity": event.get("similarity"),
            "description": description,
        }
        self._log(activity, description)

        if self._should_speak(activity):
            self.stats["spoken"] += 1
            self._spoken_at.append(time.time())
            self._record(activity, "spoken", saw=description[:160],
                         similarity=event.get("similarity"),
                         track_id=event.get("track_id"))
            if self._speak:
                self._speak(activity, record)
        else:
            self._record(activity, "logged", saw=description[:160],
                         similarity=event.get("similarity"),
                         track_id=event.get("track_id"))
            self._inject_observation(activity, description)
        return record

    def _inject_observation(self, activity: str, description: str) -> bool:
        """Put a confirmed observation into the prompt, as face events do.

        Same shape as robot/face_handler.py: a natural-language system line via
        hot_inject, behind a mode gate and a rolling rate limit. The rate limit
        is not optional here -- `unsteady` alone fired nine times in ten minutes
        during testing, and without a cap a twitchy prompt would push the real
        conversation out of the window.

        Spoken events do not come through here: their autonomous_vision prompt
        already carries the description into the history, so injecting again
        would duplicate it.
        """
        if not self._inject:
            return False
        try:
            from core.runtime_controls import context_enabled
            if not context_enabled("care_events"):
                print("[Care] injection suppressed for this mode")
                return False
        except Exception:
            pass                     # gating must never block the care path

        now = time.time()
        window = float(self.config.get("inject_window_seconds", 300))
        budget = int(self.config.get("max_injections_per_5min", 3))
        self._injected_at = [t for t in self._injected_at if now - t < window]
        if len(self._injected_at) >= budget:
            print(f"[Care] injection rate limit ({budget}/{window:.0f}s) hit; "
                  f"{activity} logged but not injected")
            return False
        self._injected_at.append(now)

        readable = activity.replace("_", " ")
        # State the observation and stop. Face events do the same -- whether it
        # is worth mentioning is the model's call, not something to preempt in
        # the injection.
        message = (f"[System: Kiki just saw this happen — {readable}. "
                   f"Observed: {description[:160]}]")
        self._inject(message)
        self.stats["injected"] += 1
        print(f"[Care] injected: {readable}")
        return True

    def adjudicate(self, activity: str, image_b64: Optional[str] = None) -> tuple:
        """Gate 04. Returns (verdict, description).

        ``image_b64`` is THE frame the detection fired on, shipped with the
        event. Judging it is the whole point: capturing a fresh frame here
        looked at the room 1-3 seconds later (ZMQ hop, HTTP fetch, inter-frame
        sleep), by which time the cup was down and the person had moved out of
        shot. Three true detections were thrown out that way, one at similarity
        0.86.

        With the real frame in hand there is nothing to corroborate against --
        a second capture is a different moment, not a second opinion -- so the
        two-frame agreement rule now applies only to the fallback path, where
        we are capturing live anyway and blur is the risk it was written for.
        """
        try:
            from core.vision.instant_vision import describe_image_b64
        except Exception as exc:
            print(f"[Care] vision unavailable, failing closed: {exc}")
            return (_UNCLEAR, "")

        gap = float(self.config.get("second_frame_delay_seconds", 2.0))
        needed = 1 if image_b64 else int(self.config.get("frames_required", 2))
        descriptions: List[str] = []

        for index in range(max(1, needed)):
            if index:
                time.sleep(gap)
            try:
                frame = image_b64 if index == 0 and image_b64 else self._clean_frame_b64()
                if not frame:
                    return (_UNCLEAR, "")
                description = describe_image_b64(
                    frame, question=_ADJUDICATION_PROMPT, max_tokens=90)
            except Exception as exc:
                # Offline or rate-limited. Fail closed rather than guess: an
                # unverifiable event is not an event.
                print(f"[Care] adjudication failed, failing closed: {exc}")
                return (_UNCLEAR, "")
            descriptions.append(description)
            if _looks_like(description, activity) != _YES:
                return (_NO, description)

        return (_YES, descriptions[-1] if descriptions else "")

    def _clean_frame_b64(self) -> Optional[str]:
        """An UN-ANNOTATED frame, which is the only kind worth adjudicating.

        The MJPEG stream carries hailooverlay's baked-in labels -- a translucent
        caption drawn straight across the subject's face -- plus the CARE debug
        panel over the top third. A vision model shown that reads it as a UI
        screenshot: every candidate came back "NO PERSON" for a person filling
        half the frame. ``/clean`` serves the raw camera frame instead.
        """
        import base64
        import urllib.request

        url = self.config.get(
            "clean_frame_url",
            f"http://{self.host}:{int(self.config.get('frame_port', 5001))}/clean")
        try:
            with urllib.request.urlopen(url, timeout=4) as response:
                data = response.read()
            if not data:
                raise RuntimeError("empty frame")
            return base64.b64encode(data).decode("utf-8")
        except Exception as exc:
            # Fall back to the annotated stream rather than going blind, but say
            # so -- the verdicts that follow are much less trustworthy.
            print(f"[Care] clean frame unavailable ({exc}); "
                  f"falling back to the annotated stream")
            try:
                from core.vision.instant_vision import capture_best_frame_b64
                return capture_best_frame_b64()
            except Exception:
                return None

    def _offer_scene(self, description: str) -> bool:
        """Hand a plain scene description to the normal vision-context path.

        Goes through the same injector ``core/vision/vision_handler.py`` uses,
        so it inherits the turn-active guard, the prompt-prefix rewarm and the
        ``vision`` mode gate rather than opening a second route into history.
        Rate-limited on its own budget so a burst of rejected candidates cannot
        crowd out the confirmed care observations.
        """
        if not self._see or not description:
            return False
        text = description.strip()
        if not text:
            return False

        now = time.time()
        window = float(self.config.get("scene_window_seconds", 300))
        budget = int(self.config.get("max_scene_injections_per_5min", 4))
        self._scene_at = [t for t in self._scene_at if now - t < window]
        if len(self._scene_at) >= budget:
            return False
        self._scene_at.append(now)
        try:
            self._see(text)
        except Exception as exc:
            print(f"[Care] scene inject failed: {exc}")
            return False
        self.stats["scenes"] += 1
        print(f"[Care] scene context: {text[:80]}")
        return True

    def _should_speak(self, activity: str) -> bool:
        """Gate 05. Speaking is the exception, not the outcome."""
        speak_for = set(self.config.get("speak_for", []))
        if activity not in speak_for:
            return False
        window = float(self.config.get("speak_window_seconds", 1800))
        budget = int(self.config.get("max_spoken_per_window", 2))
        now = time.time()
        self._spoken_at = [t for t in self._spoken_at if now - t < window]
        if len(self._spoken_at) >= budget:
            print(f"[Care] {activity} confirmed but the speaking budget is spent")
            return False
        return True

    def _log(self, activity: str, description: str) -> None:
        if self.care_plan is None:
            return
        try:
            self.care_plan.add_care_log(
                "observation", f"{activity}: {description[:200]}")
        except Exception as exc:
            print(f"[Care] could not write the care log: {exc}")
