"""Synthetic end-to-end TTFW simulation on the Pi — replays the production
turn sequence without the mic: warm prefix -> partial prefill (user 'speaking')
-> speculative stream+TTS (held) -> commit -> release -> first audio.
Measures the exact TTFW clock (commit -> first_play_event)."""
import sys, time, threading
sys.path.insert(0, "/srv/kikifast")
from core.llm import stream_response, register_history
from core import local_llm
from core.tts import TTSStreamer

SYS = ("You are Kiki, a witty companion robot who lives with Alex. "
       "You keep replies short and spoken-style. " * 20)
history = [{"role": "system", "content": SYS}]

TURNS = [
    "So what do you think about the weather tonight?",
    "Not bad. Tell me something interesting about space.",
    "Nice. And what is your favourite planet then?",
]

register_history(history)
local_llm.rewarm()   # synchronous warm of the prefix
time.sleep(1)

for n, full_user in enumerate(TURNS, 1):
    # 1) user is 'speaking': partial ASR ~1s before the end fires a prefill
    words = full_user.split()
    partial = " ".join(words[:max(2, len(words) - 3)])
    local_llm.prefill_partial(partial)
    time.sleep(1.2)  # partial prefill completes while user finishes the sentence

    # 2) speculative ASR resolves ~100ms before the endpoint commits
    spec_msgs = list(history) + [{"role": "user", "content": full_user}]
    tts = TTSStreamer(); tts.hold_playback(); tts.start()
    result = {"raw": "", "sentences": 0}
    def run_stream():
        for evt, data in stream_response(spec_msgs, local_only=True):
            if evt == "sentence":
                result["sentences"] += 1
                tts.add_sentence(data)
            elif evt == "done":
                result["raw"] = data
    th = threading.Thread(target=run_stream, daemon=True); th.start()
    time.sleep(0.10)

    # 3) endpoint commit: TTFW clock starts, audio released
    t_commit = time.time()
    tts.release_playback()
    ok = tts.first_play_event.wait(timeout=15)
    ttfw = time.time() - t_commit
    th.join(timeout=30)
    tts.finish()
    print(f"=== TURN {n}: TTFW {'%.3f' % ttfw}s {'' if ok else '(TIMEOUT!)'} "
          f"sentences={result['sentences']} reply={result['raw'][:60]!r}")
    history = spec_msgs + [{"role": "assistant", "content": result["raw"]}]
    register_history(history, after_speaking=True)
    time.sleep(6)  # let --prefill-after-response warm the next turn
