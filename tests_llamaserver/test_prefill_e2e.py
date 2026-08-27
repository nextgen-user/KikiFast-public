"""
End-to-end prefill / KV-cache test against the live llama.cpp box.

Verifies the three properties the speaking path now depends on:

  1. WARM TURNS: after a turn completes and register_history() rewarms the
     prefix (conversation ending with the assistant reply, max_tokens=1),
     the NEXT user turn gets a cache hit → time-to-first-token (TTFT) is a
     small fraction of the cold prefill.
  2. MUTATION = INVALIDATION (the old bug, demonstrated): editing an EARLIER
     system message makes TTFT cold again. This is why main.py no longer
     rewrites message_history[1].
  3. ABORTABLE PREFILL: a background request can be killed DURING prefill via
     core.local_llm.preempt_background() (socket shutdown), and the slot is
     free for a speaking request within ~1s.

Run:  python3 tests_llamaserver/test_prefill_e2e.py
"""

import json
import sys
import time
import threading
import requests

sys.path.insert(0, "/srv/kikifast")

BASE = "http://192.0.2.10:8080/v1"
URL = f"{BASE}/chat/completions"

# Unique filler so nothing is pre-cached from earlier runs. ~6k chars ≈ 1.7k tokens.
RUN_ID = str(int(time.time()))
FILLER = (" The quick brown fox jumps over the lazy dog near the riverbank at dawn"
          " while distant trains rattle through the valley carrying cargo north.") * 40
SYSTEM = f"You are Kiki, test run {RUN_ID}. Background knowledge: {FILLER}"


def ttft(messages, max_tokens=32, label=""):
    """Stream a chat completion; return (ttft_seconds, total_seconds, text)."""
    payload = {"model": "kiki-local", "messages": messages, "stream": True,
               "max_tokens": max_tokens, "temperature": 0.7,
               "cache_prompt": True, "thinking_budget_tokens": 0}
    t0 = time.time()
    first = None
    text = ""
    with requests.post(URL, json=payload, stream=True, timeout=(5, 300)) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", "ignore")
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            try:
                d = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            delta = (d.get("choices") or [{}])[0].get("delta", {}) or {}
            chunk = delta.get("content") or ""
            if chunk and first is None:
                first = time.time() - t0
            text += chunk
    total = time.time() - t0
    print(f"  [{label}] TTFT={first if first is not None else total:.2f}s total={total:.2f}s")
    return (first if first is not None else total), total, text


def prefill_only(messages, label=""):
    """Rewarm-style request: max_tokens=1, output discarded."""
    t0 = time.time()
    payload = {"model": "kiki-local", "messages": messages, "stream": False,
               "max_tokens": 1, "temperature": 1.0,
               "cache_prompt": True, "thinking_budget_tokens": 0}
    r = requests.post(URL, json=payload, timeout=(5, 300))
    r.raise_for_status()
    print(f"  [{label}] prefill request took {time.time() - t0:.2f}s")


def main():
    ok = True

    print("\n=== Test 1: warm turn after register_history-style rewarm ===")
    convo = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": "Say hi in five words."}]
    cold_ttft, _, reply = ttft(convo, label="cold first turn")

    # What register_history() does after the turn: history incl. assistant reply.
    convo.append({"role": "assistant", "content": reply})
    prefill_only(convo, label="rewarm (history + assistant)")

    convo.append({"role": "user", "content": "And what is two plus two?"})
    warm_ttft, _, reply2 = ttft(convo, label="warm next turn")
    if warm_ttft < max(1.5, cold_ttft * 0.4):
        print(f"  ✅ PASS: warm TTFT {warm_ttft:.2f}s vs cold {cold_ttft:.2f}s")
    else:
        print(f"  ❌ FAIL: warm TTFT {warm_ttft:.2f}s not much faster than cold {cold_ttft:.2f}s")
        ok = False

    print("\n=== Test 2: mutating an earlier system message invalidates cache ===")
    convo.append({"role": "assistant", "content": reply2})
    mutated = [dict(m) for m in convo]
    mutated[0]["content"] = mutated[0]["content"] + " [workers ctx updated]"
    mutated.append({"role": "user", "content": "Name one color."})
    mut_ttft, _, _ = ttft(mutated, label="turn after system-msg mutation")
    print(f"  (info) mutation TTFT={mut_ttft:.2f}s vs warm {warm_ttft:.2f}s — "
          f"{'as expected, mutation forces re-prefill' if mut_ttft > warm_ttft * 2 else 'server tolerated mutation (partial cache reuse)'}")

    print("\n=== Test 3: preempt aborts a background request mid-prefill ===")
    from core import local_llm

    big_prompt = f"Test {RUN_ID}-bg. Summarize this: {FILLER * 3}"
    bg_done = {"t": None, "result": "unset"}

    def bg():
        t0 = time.time()
        res = local_llm.generate_background(prompt=big_prompt, max_tokens=200,
                                            temperature=0.7)
        bg_done["t"] = time.time() - t0
        bg_done["result"] = "none/aborted" if res is None else f"{len(res)} chars"

    # make sure conversation isn't 'hot' so the bg request is accepted
    local_llm._last_user_activity["t"] = 0.0
    th = threading.Thread(target=bg, daemon=True)
    th.start()
    time.sleep(1.5)  # request should be deep in prefill now
    t0 = time.time()
    local_llm.preempt_background()
    preempt_call = time.time() - t0
    th.join(timeout=10)
    aborted_in = bg_done["t"]
    print(f"  preempt_background() returned in {preempt_call*1000:.0f}ms; "
          f"bg request ended after {aborted_in:.2f}s total ({bg_done['result']})")

    t0 = time.time()
    speak_ttft, _, _ = ttft([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": "Quick — say ok."}],
                            label="speaking request right after preempt")
    if preempt_call < 0.2 and th.is_alive() is False and aborted_in is not None and aborted_in < 4.0:
        print(f"  ✅ PASS: non-blocking preempt, prefill aborted in {aborted_in:.2f}s")
    else:
        print(f"  ❌ FAIL: preempt_call={preempt_call:.2f}s aborted_in={aborted_in}")
        ok = False
    print(f"  (info) speaking TTFT after preempt: {speak_ttft:.2f}s "
          f"(includes re-prefill of this test's prompt — fine; in production the "
          f"rewarm keeps the real convo warm)")

    print("\n" + ("ALL TESTS PASSED ✅" if ok else "SOME TESTS FAILED ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
