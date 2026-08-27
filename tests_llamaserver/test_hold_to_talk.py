"""Unit tests for the IR push-to-talk hold gesture (no hardware needed).

Covers:
  1. _Endpointer: hold_event blocks the silence-window commit indefinitely.
  2. _Endpointer.force_commit(): immediate commit on hand release; bare False
     (no commit) when nothing committable was captured.
  3. Speculative ASR still fires during held silence (reply pipeline warms).
  4. Normal endpoint behaviour unchanged when no hold is active.
  5. max_utterance safety valve still commits even while held.
  6. IRControls normal-mode tick: immediate hold-start on either sensor,
     debounced hold-end, release cooldown, both-hold settings entry cancels
     the hold.

Run on the Pi:
  source /srv/kikifast/.venv/bin/activate
  cd /srv/kikifast && python -m tests_llamaserver.test_hold_to_talk
"""

import sys
import threading
from concurrent.futures import Future

import numpy as np

sys.path.insert(0, "/srv/kikifast")

from core.stt import _Endpointer, FRAME, FRAME_MS

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✔ {name}")
    else:
        FAIL += 1
        print(f"  ✘ {name} {detail}")


class FakeProb:
    def __init__(self, p):
        self.p = p

    def item(self):
        return self.p


class FakeModel:
    """Scripted VAD: pops probabilities from a list (last value repeats)."""

    def __init__(self):
        self.script = []

    def __call__(self, tensor, sr):
        p = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return FakeProb(p)

    def reset_states(self):
        pass


class FakeAsr:
    def __init__(self, text="hello world"):
        self.text = text
        self.calls = 0

    def submit(self, audio):
        self.calls += 1
        f = Future()
        f.set_result((self.text, 0.05))
        return f


def make_ep(hold_event, events, asr_text="hello world", max_utt=20.0):
    cfg = {
        "vad_threshold": 0.5,
        "endpoint_ms": 600,
        "spec_silence_ms": 240,
        "min_speech_ms": 150,
        "preroll_ms": 250,
        "max_utterance_seconds": max_utt,
        "partial_asr_interval_ms": 0,
    }
    model = FakeModel()
    asr = FakeAsr(asr_text)
    ep = _Endpointer(
        model, cfg, asr,
        on_commit=lambda t: events.append(("final", t)),
        on_interim=lambda t: None,
        on_speculative=lambda t: events.append(("speculative", t)),
        on_spec_invalid=lambda: events.append(("spec_invalid", None)),
        hold_event=hold_event,
    )
    return ep, model, asr


def feed(ep, model, n_frames, prob, t0=0.0):
    frame = np.zeros(FRAME, dtype=np.float32)
    t = t0
    for _ in range(n_frames):
        model.script = [prob]
        ep.process(frame, t)
        t += FRAME_MS / 1000.0
    return t


def test_hold_blocks_endpoint():
    print("[1] hold blocks the silence-window commit")
    hold = threading.Event()
    events = []
    ep, model, asr = make_ep(hold, events)
    t = feed(ep, model, 16, 0.9)          # ~512ms speech
    hold.set()
    t = feed(ep, model, 80, 0.0, t)       # ~2.5s silence, way past endpoint_ms
    check("no commit while held", not any(e[0] == "final" for e in events),
          f"events={events}")
    check("still in_speech while held", ep.in_speech)
    hold.clear()
    committed = ep.force_commit()
    check("force_commit commits", committed and ("final", "hello world") in events,
          f"events={events}")
    check("endpointer reset after commit", not ep.in_speech and ep.utt == [])


def test_force_commit_no_speech():
    print("[2] force_commit with nothing captured")
    hold = threading.Event()
    events = []
    ep, model, asr = make_ep(hold, events)
    hold.set()
    feed(ep, model, 10, 0.0)              # silence only, never in_speech
    hold.clear()
    committed = ep.force_commit()
    check("returns False", committed is False)
    check("no events", events == [], f"events={events}")


def test_spec_fires_during_hold():
    print("[3] speculative ASR fires during held silence")
    hold = threading.Event()
    events = []
    ep, model, asr = make_ep(hold, events)
    t = feed(ep, model, 16, 0.9)
    hold.set()
    feed(ep, model, 12, 0.0, t)           # ~384ms silence > spec_silence_ms
    check("speculative announced", any(e[0] == "speculative" for e in events),
          f"events={events}")
    check("but no final yet", not any(e[0] == "final" for e in events))
    hold.clear()
    ep.force_commit()
    check("release commits using the spec result",
          ("final", "hello world") in events, f"events={events}")
    check("single ASR call (spec reused)", asr.calls == 1, f"calls={asr.calls}")


def test_normal_endpoint_unchanged():
    print("[4] normal endpoint (no hold) unchanged")
    hold = threading.Event()
    events = []
    ep, model, asr = make_ep(hold, events)
    t = feed(ep, model, 16, 0.9)
    feed(ep, model, 25, 0.0, t)           # ~800ms silence > endpoint_ms 600
    check("committed on silence", ("final", "hello world") in events,
          f"events={events}")


def test_max_utterance_valve_while_held():
    print("[5] max_utterance safety valve still fires while held")
    hold = threading.Event()
    events = []
    ep, model, asr = make_ep(hold, events, max_utt=2.0)
    hold.set()
    t = feed(ep, model, 16, 0.9)          # ~512ms speech
    feed(ep, model, 60, 0.0, t)           # push speech+silence past 2.0s
    check("committed at the cap", ("final", "hello world") in events,
          f"events={events}")


def test_ir_tick_state_machine():
    print("[6] IRControls normal-mode tick")
    import core.ir_controls as irc
    irc_gpiod = irc.GPIOD_AVAILABLE
    irc.GPIOD_AVAILABLE = False           # never touch real GPIO lines
    try:
        fired = []
        c = irc.IRControls(
            on_talk_hold_start=lambda: fired.append("start"),
            on_talk_hold_end=lambda: fired.append("end"),
            on_talk_hold_cancel=lambda: fired.append("cancel"),
            on_enter_settings=lambda: fired.append("settings"),
        )
        t = 100.0
        # immediate start on the RIGHT sensor alone
        c._tick_normal(False, True, t)
        check("start fires on first poll of either sensor", fired == ["start"],
              f"fired={fired}")
        check("hold_active exposed", c.hold_active)
        # blip: clear for only 60ms then back — no end
        c._tick_normal(False, False, t + 0.30)
        c._tick_normal(False, True, t + 0.36)
        check("60ms clear blip does not end the hold", fired == ["start"],
              f"fired={fired}")
        # real release: clear past the debounce
        c._tick_normal(False, False, t + 0.60)
        c._tick_normal(False, False, t + 0.75)
        check("debounced release fires end", fired == ["start", "end"],
              f"fired={fired}")
        check("hold_active cleared", not c.hold_active)
        # cooldown: re-press 100ms later is ignored, 500ms later fires
        c._tick_normal(True, False, t + 0.85)
        check("cooldown blocks instant re-trigger", fired == ["start", "end"],
              f"fired={fired}")
        c._tick_normal(False, False, t + 1.0)
        c._tick_normal(True, False, t + 1.30)
        check("re-press after cooldown starts again",
              fired == ["start", "end", "start"], f"fired={fired}")
        # both-hold from inside the hold -> cancel + settings
        c._tick_normal(True, True, t + 1.40)
        c._tick_normal(True, True, t + 2.70)   # >= BOTH_HOLD_S later
        check("both-hold cancels the hold and enters settings",
              fired == ["start", "end", "start", "cancel", "settings"]
              and c.mode == "settings", f"fired={fired} mode={c.mode}")
    finally:
        irc.GPIOD_AVAILABLE = irc_gpiod


if __name__ == "__main__":
    test_hold_blocks_endpoint()
    test_force_commit_no_speech()
    test_spec_fires_during_hold()
    test_normal_endpoint_unchanged()
    test_max_utterance_valve_while_held()
    test_ir_tick_state_machine()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
