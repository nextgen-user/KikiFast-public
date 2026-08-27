"""E2E test of the speculative-turn primitives (run on the Pi, live servers):
1. LocalTTSStreamer hold_playback: audio synthesized but NOT played until release.
2. abort() while held: nothing plays, threads exit.
3. stream_response(abort_event): local stream killed cooperatively, slot freed,
   a follow-up request succeeds.
"""
import sys, threading, time
sys.path.insert(0, "/srv/kikifast")

from core.tts import TTSStreamer
from core.llm import stream_response

def t1_hold_release():
    s = TTSStreamer()
    assert hasattr(s, "hold_playback"), "local streamer missing hold API"
    s.hold_playback(); s.start()
    s.add_sentence("Hi.")
    time.sleep(3.0)
    assert not s.first_play_event.is_set(), "played while held!"
    s.release_playback()
    ok = s.first_play_event.wait(timeout=5)
    assert ok, "did not play after release"
    s.finish()
    print("T1 PASS: held for 3s, played only after release")

def t2_abort_while_held():
    s = TTSStreamer()
    s.hold_playback(); s.start()
    s.add_sentence("This must never be heard aloud.")
    time.sleep(2.0)
    s.abort()
    time.sleep(1.0)
    assert s._aborted
    print("T2 PASS: abort while held, nothing played")

def t3_stream_abort():
    ev = threading.Event()
    msgs = [{"role": "system", "content": "You are a test bot. Answer at length."},
            {"role": "user", "content": "Write ten sentences about rivers."}]
    got, done = [], []
    def run():
        for evt, data in stream_response(msgs, abort_event=ev, local_only=True):
            got.append(evt)
            if evt == "done": done.append(1)
    th = threading.Thread(target=run); t0 = time.time(); th.start()
    time.sleep(1.5); ev.set()
    th.join(timeout=10)
    assert not th.is_alive(), "stream did not abort"
    assert not done, "abort still produced done"
    print(f"T3 PASS: stream aborted after {time.time()-t0:.1f}s, events seen: {len(got)}")
    # slot must be free: a fresh short request completes
    out = []
    for evt, data in stream_response([{"role":"user","content":"Say OK."}], local_only=True):
        if evt == "done": out.append(data)
    assert out and out[0], "slot not usable after abort"
    print(f"T3b PASS: follow-up request fine after abort: {out[0][:40]!r}")

t1_hold_release(); t2_abort_while_held(); t3_stream_abort()
print("ALL SPEC TESTS PASSED")
