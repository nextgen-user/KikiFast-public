"""Verify the PCM cache: prewarm, then a streamer synth of the same tag must
hit the cache instantly and produce audio; a fresh synth of the same text
must byte-match the cached copy (determinism proof on the live server)."""
import sys, time
sys.path.insert(0, "/srv/kikifast")
from core import tts as T

# 1) prewarm two fragments
t0 = time.time()
T.prewarm_tts_cache(["[laughter]", "Let me check that."])
print(f"prewarm took {time.time()-t0:.2f}s, cache size={len(T._PCM_CACHE)}")

# 2) determinism: fresh POST of the same text == cached bytes
import requests
r = requests.post(f"{T._LOCAL_TTS_URL}/v1/audio/speech",
                  json={"input": "[laughter]", "response_format": "pcm"}, timeout=30)
fresh = T._trim_edge_silence(r.content, T._LOCAL_SILENCE_THR, T._LOCAL_EDGE_KEEP_MS)
cached = T._pcm_cache_get("[laughter]")
assert cached == fresh, f"determinism broken: {len(cached)} vs {len(fresh)}"
print(f"determinism OK: cached==fresh ({len(fresh)} bytes)")

# 3) streamer cache-hit path: time to first play must be ~instant
s = T.TTSStreamer(); s.start()
t0 = time.time()
s.add_sentence("[laughter]")
ok = s.first_play_event.wait(timeout=5)
dt = time.time() - t0
s.finish()
assert ok, "no playback"
print(f"cache-hit first-play in {dt*1000:.0f}ms {'PASS' if dt < 0.15 else 'TOO SLOW'}")
