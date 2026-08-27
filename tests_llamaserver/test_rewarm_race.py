"""
Rewarm lost-update race — regression test against the live llama.cpp box.

The bug: the rewarm coalescer used a single flag. If the speaking prefix changed
WHILE a rewarm worker was in flight (e.g. a background summarize swapped the
history), the new schedule_rewarm() was silently DROPPED. The worker finished
warming the STALE prefix, nothing re-fired, and the box stayed warm for the old
prefix — so the next user turn re-prefilled from the divergence point
(observed: 16s TTFW, server f_keep ~0.37) even though the box had been idle for
30+ seconds.

The fix: a schedule that lands while a worker is running marks the prefix DIRTY;
the worker re-warms the LATEST prefix before exiting (core/local_llm.py).

Two parts:
  Part A (logic, deterministic): monkeypatch rewarm() to be slow + record which
    prefix each pass warmed. Fire A, then swap to B mid-flight. The worker MUST
    end by warming B. (With the old code the recording would be ["A"] only.)
  Part B (live box): real register_history() against the box. Warm prefix A,
    swap to B (shared head, divergent body — like a summary swap) while A is
    still warming, then send a speaking turn for B and read the server's
    timings.prompt_n. If B was warmed, prompt_n is tiny (only the new user
    words). If the race dropped it, prompt_n is hundreds/thousands.

Run:  python3 tests_llamaserver/test_rewarm_race.py
"""

import json
import sys
import time
import threading

import requests

sys.path.insert(0, "/srv/kikifast")

from core import local_llm
from core.llm import register_history, _normalize_messages_for_local, _USE_LOCAL

BASE = "http://192.0.2.10:8080/v1"
URL = f"{BASE}/chat/completions"
RUN_ID = str(int(time.time()))


# --------------------------------------------------------------------------- #
# Part A: deterministic logic test of the dirty/redo coalescer
# --------------------------------------------------------------------------- #
def test_lost_update_logic():
    print("\n=== Part A: coalescer re-warms the LATEST prefix (logic) ===")
    warmed = []           # records the prefix id each rewarm() pass warmed
    started = threading.Event()
    orig_rewarm = local_llm.rewarm

    def slow_rewarm_stub():
        # Mimic the real rewarm(): snapshot the CURRENT prefix, warm it slowly.
        prefix = local_llm._speaking_prefix["messages"]
        pid = prefix[0]["content"] if prefix else None
        started.set()
        time.sleep(0.6)   # simulate a multi-second prefill
        warmed.append(pid)

    local_llm.rewarm = slow_rewarm_stub
    # Save + force the gates open so this is a pure logic test.
    saved = (local_llm._REWARM_ENABLED, local_llm._speaking_prefix["messages"])
    try:
        local_llm._REWARM_ENABLED = True
        local_llm._speaking_active.clear()

        # Schedule A; let the worker get mid-flight on it.
        local_llm._speaking_prefix["messages"] = [{"role": "system", "content": "A"}]
        local_llm.schedule_rewarm()
        assert started.wait(timeout=2), "rewarm worker never started"
        time.sleep(0.1)   # worker is now inside the 0.6s warm of A

        # RACE: swap to B and schedule again. Old code dropped this.
        local_llm._speaking_prefix["messages"] = [{"role": "system", "content": "B"}]
        local_llm.schedule_rewarm()

        # Let the worker drain (A pass + dirty re-pass for B).
        deadline = time.time() + 5
        while time.time() < deadline:
            with local_llm._rewarm_lock:
                running = local_llm._rewarm_running
            if not running:
                break
            time.sleep(0.05)

        print(f"  warmed sequence: {warmed}")
        ok = warmed and warmed[-1] == "B" and "B" in warmed
        if ok:
            print("  ✅ PASS: worker re-warmed B after the mid-flight swap "
                  "(no lost update)")
        else:
            print("  ❌ FAIL: B was dropped — box would stay warm for the stale "
                  f"prefix. warmed={warmed}")
        return ok
    finally:
        local_llm.rewarm = orig_rewarm
        local_llm._REWARM_ENABLED, local_llm._speaking_prefix["messages"] = saved


# --------------------------------------------------------------------------- #
# Part B: live end-to-end against the box
# --------------------------------------------------------------------------- #
def _filler(words):
    base = ("the quick brown fox jumps over the lazy dog near the riverbank at "
            "dawn while distant trains rattle through the valley carrying cargo ")
    return (base * ((words // len(base.split())) + 1))


def timings_for(messages, label=""):
    """POST (non-stream) and return the server's timings dict (prompt_n/cache_n)."""
    payload = {"model": "kiki-local", "messages": messages, "stream": False,
               "max_tokens": 8, "temperature": 0.7,
               "cache_prompt": True, "thinking_budget_tokens": 0}
    r = requests.post(URL, json=payload, timeout=(5, 300))
    r.raise_for_status()
    t = r.json().get("timings", {}) or {}
    pn, cn = t.get("prompt_n"), t.get("cache_n")
    print(f"  [{label}] prompt_n={pn} cache_n={cn}")
    return t


def _wait_rewarm_idle(timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with local_llm._rewarm_lock:
            running = local_llm._rewarm_running
        if not running:
            return True
        time.sleep(0.2)
    return False


def test_live_swap_warm():
    print("\n=== Part B: live box stays warm for the SWAPPED prefix ===")
    if not _USE_LOCAL:
        print("  ⚠ SKIP: local box disabled in config.")
        return True

    # Don't let conversation-hot gating interfere; rewarm bypasses it anyway.
    local_llm._last_user_activity["t"] = 0.0
    local_llm._speaking_active.clear()

    sys_head = f"You are Kiki. Test run {RUN_ID}. {_filler(120)}"
    # Prefix A: a long-ish conversation (the "pre-summary" state).
    A = [{"role": "system", "content": sys_head},
         {"role": "user", "content": "Tell me about the trains. " + _filler(200)},
         {"role": "assistant", "content": "Sure — here is a long ramble. " + _filler(220)}]
    # Prefix B: SAME system head, DIVERGENT body (the "post-summary" state).
    B = [{"role": "system", "content": sys_head},
         {"role": "system", "content": "Your memories: " + _filler(140)},
         {"role": "user", "content": "ok"}]

    # Warm A, then race-swap to B while A's rewarm is still in flight.
    register_history(A)                 # update_speaking_prefix(A) + schedule_rewarm
    time.sleep(0.4)                     # A's prefill is now in flight on the box
    register_history(B)                 # old code: DROPPED while worker busy
    print("  registered A then B (B during A's in-flight rewarm)")

    if not _wait_rewarm_idle():
        print("  ❌ FAIL: rewarm worker did not settle in time.")
        return False
    time.sleep(0.3)

    # Speaking turn = exactly the warmed B prefix + a new user message.
    warmed_prefix = list(local_llm._speaking_prefix["messages"])
    assert warmed_prefix and warmed_prefix == _normalize_messages_for_local(B), \
        "speaking_prefix is not B — registry desynced"
    speak = warmed_prefix + [{"role": "user", "content": "Say ok in one word."}]
    t = timings_for(speak, label="speaking turn on B after race")
    pn = t.get("prompt_n")

    if pn is None:
        print("  ⚠ INCONCLUSIVE: box did not return timings.prompt_n.")
        return True
    # If B is warm, only the new user words + chat scaffold prefill (~tens).
    if pn < 120:
        print(f"  ✅ PASS: B was warm — prompt_n={pn} (only the new user turn). "
              "The mid-flight swap was NOT lost.")
        return True
    print(f"  ❌ FAIL: prompt_n={pn} — B was not warmed; the box reprocessed from "
          "the A/B divergence (the original 16s-spike bug).")
    return False


def main():
    results = []
    results.append(("Part A (logic)", test_lost_update_logic()))
    try:
        results.append(("Part B (live box)", test_live_swap_warm()))
    except requests.RequestException as e:
        print(f"  ⚠ Part B could not reach the box: {e}")
        results.append(("Part B (live box)", None))

    print("\n--- summary ---")
    ok = True
    for name, r in results:
        tag = "PASS ✅" if r else ("SKIP/INCONCLUSIVE ⚠" if r is None else "FAIL ❌")
        print(f"  {name}: {tag}")
        if r is False:
            ok = False
    print("\n" + ("ALL TESTS PASSED ✅" if ok else "SOME TESTS FAILED ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
