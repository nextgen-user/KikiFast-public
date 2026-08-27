"""
Phase-2 verification: does closing the client connection DURING PREFILL actually
free the llama.cpp box, or does the orphaned prefill keep burning GPU?

Method:
  1. Baseline: TTFT of a small speaking-style request on an otherwise idle box.
  2. Fire a large-prefill background request (unique ~3k-token prompt so nothing
     is cached), hard-close it after ABORT_AFTER_S (mid-prefill).
  3. Immediately fire the same small speaking request again and measure TTFT.
If abort works server-side, (3) ~= (1). If the orphaned prefill keeps running,
(3) is inflated by the leftover prefill work.
"""
import random
import threading
import time

import requests

URL = "http://192.0.2.10:8080/v1/chat/completions"
ABORT_AFTER_S = 0.3

WORDS = ("apple river mountain quantum violet thunder crystal nomad ember harbor "
         "lattice meadow cipher orbit prism saffron tundra velvet wharf zenith").split()


def big_prompt(n_words=2400):
    rng = random.Random()  # unseeded → unique each run → no cache hits
    return "Summarize the following list of words: " + " ".join(rng.choice(WORDS) + str(rng.randint(0, 9999)) for _ in range(n_words))


def ttft(prompt, max_tokens=16):
    t0 = time.time()
    with requests.post(URL, json={
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "cache_prompt": True,
    }, stream=True, timeout=(3, 300)) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line and line.startswith(b"data: ") and b'"content"' in line:
                return time.time() - t0
    return None


def fire_and_abort(prompt):
    resp = requests.post(URL, json={
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64, "temperature": 0, "stream": True,
        "cache_prompt": True,
    }, stream=True, timeout=(3, 300))
    time.sleep(ABORT_AFTER_S)
    resp.close()  # what preempt_background() does
    print(f"  aborted background request after {ABORT_AFTER_S}s (mid-prefill)")


def main():
    speak_prompt = "Say hello in five words."

    print("1) warm the tiny speaking prompt (so later runs measure only interference)...")
    ttft(speak_prompt)
    t_base = ttft(speak_prompt)
    print(f"   baseline speaking TTFT (idle box): {t_base*1000:.0f} ms")

    print("2) fire big-prefill background request, abort mid-prefill...")
    bg = big_prompt()
    th = threading.Thread(target=fire_and_abort, args=(bg,))
    th.start()
    time.sleep(ABORT_AFTER_S + 0.05)  # let the abort happen first

    print("3) speaking request immediately after abort...")
    t_after = ttft(speak_prompt)
    th.join()
    print(f"   speaking TTFT after mid-prefill abort: {t_after*1000:.0f} ms")

    print()
    ratio = t_after / t_base if t_base else float("inf")
    print(f"RESULT: baseline={t_base*1000:.0f}ms after-abort={t_after*1000:.0f}ms ratio={ratio:.1f}x")
    if ratio < 2.0:
        print("→ PASS: disconnect during prefill frees the box. No llama-server change needed (no R2).")
    else:
        print("→ FAIL: orphaned prefill kept the box busy. R2 (disconnect detection during prefill) is required.")


if __name__ == "__main__":
    main()
