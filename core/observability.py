"""
Lightweight, non-blocking observability recorder for KikiFast.

Gives the Web UI (webui/server.py) an honest window into Kiki's whole workflow:
every speaking turn, Unified Idle Mind session, worker run,
tool call, vision update and summarization — with durations (incl. time-to-first-word),
the exact context fed to the local LLM, and the model's response/appends.

DESIGN — must never slow the speaking path:
  - record() does only an in-memory append under a short lock, then hands the event
    to a background thread for disk I/O. No blocking, no network.
  - Every public method is wrapped so a failure here can NEVER break a real turn.
  - Coarse-grained: call at boundaries (turn/cycle/tool/worker), not in tight loops.

One process-wide singleton: `get_recorder()`.
"""

import os
import json
import time
import threading
import itertools
import queue
from collections import deque, defaultdict
from contextlib import contextmanager

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
_EVENTS_PATH = os.path.join(_LOG_DIR, "events.jsonl")
_MAX_EVENTS = 4000          # in-memory ring buffer
_MAX_DUR_SAMPLES = 300      # per-type duration samples kept for averages
_MAX_JSONL_BYTES = 20 * 1024 * 1024   # rotate events.jsonl at 20 MB
_MAX_SESSIONS = 200         # how many recent grouped sessions to keep in memory
_MAX_SESSION_STEPS = 250    # cap steps per session
_STEP_STR_CAP = 8000        # cap any single string field on a step/meta


def _cap_str(v, n=_STEP_STR_CAP):
    """Truncate a string value for storage so one giant prompt/result can't
    bloat memory or the JSON payload to the Web UI."""
    if isinstance(v, str) and len(v) > n:
        return v[:n] + f"\n…[+{len(v) - n} chars]"
    return v


class Recorder:
    def __init__(self):
        self._buf = deque(maxlen=_MAX_EVENTS)
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._counters = defaultdict(int)
        self._durations = defaultdict(list)      # type -> [ms, ...]
        self._latest_context = None              # last full context fed to the LLM
        self._latest_by_type = {}                # type -> most recent event
        self._sessions = {}                      # sid -> session dict (grouped view)
        self._session_order = deque(maxlen=_MAX_SESSIONS)   # sids, oldest→newest
        self._sids = itertools.count(1)
        self._session_start = time.time()
        self._wq = queue.Queue(maxsize=10000)
        self._writer = threading.Thread(target=self._writer_loop, daemon=True,
                                        name="obs_writer")
        self._writer.start()

    # ---- background disk writer -------------------------------------------
    def _writer_loop(self):
        try:
            os.makedirs(_LOG_DIR, exist_ok=True)
        except Exception:
            pass
        while True:
            ev = self._wq.get()
            try:
                if (os.path.exists(_EVENTS_PATH)
                        and os.path.getsize(_EVENTS_PATH) > _MAX_JSONL_BYTES):
                    try:
                        os.replace(_EVENTS_PATH, _EVENTS_PATH + ".1")
                    except Exception:
                        pass
                with open(_EVENTS_PATH, "a") as f:
                    f.write(json.dumps(ev, default=str) + "\n")
            except Exception:
                pass

    # ---- core API ---------------------------------------------------------
    def record(self, etype, name=None, phase=None, duration_ms=None, **meta):
        """Record one event. Returns its id (or None on failure — never raises)."""
        try:
            ev = {
                "id": next(self._ids),
                "ts": round(time.time(), 3),
                "type": etype,
                "name": name,
                "phase": phase,
                "duration_ms": duration_ms,
                "meta": meta,
            }
            with self._lock:
                self._buf.append(ev)
                self._latest_by_type[etype] = ev
                if phase != "start":
                    self._counters[etype] += 1
                if duration_ms is not None:
                    d = self._durations[etype]
                    d.append(duration_ms)
                    if len(d) > _MAX_DUR_SAMPLES:
                        del d[:len(d) - _MAX_DUR_SAMPLES]
            try:
                self._wq.put_nowait(ev)
            except queue.Full:
                pass
            return ev["id"]
        except Exception:
            return None

    @contextmanager
    def span(self, etype, name=None, **meta):
        """Time a block: emits a start then an end event with duration_ms.

        Yield value is a dict you can mutate to attach result metadata to the end
        event, e.g.:  with rec.span("tool", name="search_web") as s: s["result"]=...
        """
        t0 = time.time()
        self.record(etype, name=name, phase="start", **meta)
        extra = {}
        try:
            yield extra
        finally:
            merged = dict(meta)
            merged.update(extra)
            self.record(etype, name=name, phase="end",
                        duration_ms=int((time.time() - t0) * 1000), **merged)

    def set_context(self, messages, **info):
        """Snapshot the EXACT context fed to the local LLM (capped per message)."""
        try:
            snap = []
            for m in messages:
                if isinstance(m, dict):
                    role, content = m.get("role", "?"), m.get("content", "")
                else:
                    role, content = getattr(m, "role", "?"), getattr(m, "content", "")
                if isinstance(content, list):
                    content = " ".join(
                        str(p.get("text", "")) for p in content if isinstance(p, dict))
                content = str(content)
                if len(content) > 8000:
                    content = content[:8000] + f"\n…[+{len(content) - 8000} chars]"
                snap.append({"role": role, "content": content})
            with self._lock:
                self._latest_context = {"ts": round(time.time(), 3),
                                        "messages": snap, "info": info}
        except Exception:
            pass

    # ---- read side (for the Web UI) ---------------------------------------
    def get_events(self, limit=300, etype=None, since_id=0):
        with self._lock:
            evs = list(self._buf)
        if etype and etype != "all":
            evs = [e for e in evs if e["type"] == etype]
        if since_id:
            evs = [e for e in evs if e["id"] > since_id]
        return evs[-limit:]

    def get_stats(self):
        with self._lock:
            counts = dict(self._counters)
            durs = {
                k: {
                    "count": len(v),
                    "avg_ms": int(sum(v) / len(v)) if v else 0,
                    "max_ms": max(v) if v else 0,
                    "last_ms": v[-1] if v else 0,
                }
                for k, v in self._durations.items()
            }
            uptime = time.time() - self._session_start
            latest = dict(self._latest_by_type)
        return {"counters": counts, "durations": durs,
                "uptime_s": int(uptime), "latest": latest}

    def get_latest_context(self):
        with self._lock:
            return self._latest_context

    # ---- sessions: grouped, end-to-end view of one unit of work -----------
    # A "session" is one coherent process — a speaking turn, Unified Idle Mind
    # session, worker run, or summarization. Its
    # `steps` are the ordered sub-events (context fed, each LLM prompt+response,
    # each tool call+result, the final appends), so the Web UI can show every
    # process segregated and readable end to end instead of one flat firehose.

    def start_session(self, kind, name=None, **meta):
        """Open a session; returns its id (sid). Never raises."""
        try:
            with self._lock:
                sid = f"{kind}-{next(self._sids)}"
                # If the ring is full, the oldest sid is about to be evicted from
                # _session_order — drop its dict too so memory stays bounded.
                if len(self._session_order) == self._session_order.maxlen:
                    old = self._session_order[0]
                    self._sessions.pop(old, None)
                self._sessions[sid] = {
                    "sid": sid, "kind": kind, "name": name,
                    "ts": round(time.time(), 3), "end_ts": None,
                    "duration_ms": None, "status": "running",
                    "meta": {k: _cap_str(v) for k, v in meta.items()},
                    "steps": [],
                }
                self._session_order.append(sid)
            return sid
        except Exception:
            return None

    def log_step(self, sid, kind, **data):
        """Append one ordered sub-event to an open session. Never raises."""
        if not sid:
            return
        try:
            step = {"ts": round(time.time(), 3), "kind": kind}
            for k, v in data.items():
                step[k] = _cap_str(v)
            with self._lock:
                sess = self._sessions.get(sid)
                if sess is None:
                    return
                sess["steps"].append(step)
                if len(sess["steps"]) > _MAX_SESSION_STEPS:
                    del sess["steps"][:len(sess["steps"]) - _MAX_SESSION_STEPS]
        except Exception:
            pass

    def end_session(self, sid, status="done", **meta):
        """Close a session with a final status and optional summary meta."""
        if not sid:
            return
        try:
            with self._lock:
                sess = self._sessions.get(sid)
                if sess is None:
                    return
                sess["end_ts"] = round(time.time(), 3)
                sess["duration_ms"] = int((sess["end_ts"] - sess["ts"]) * 1000)
                sess["status"] = status
                for k, v in (meta or {}).items():
                    sess["meta"][k] = _cap_str(v)
        except Exception:
            pass

    def get_sessions(self, limit=80, kind=None):
        """Newest-first list of session summaries (no steps — just a count)."""
        with self._lock:
            sids = list(self._session_order)
            out = []
            for sid in reversed(sids):
                s = self._sessions.get(sid)
                if not s:
                    continue
                if kind and kind != "all" and s["kind"] != kind:
                    continue
                out.append({
                    "sid": s["sid"], "kind": s["kind"], "name": s["name"],
                    "ts": s["ts"], "end_ts": s["end_ts"],
                    "duration_ms": s["duration_ms"], "status": s["status"],
                    "step_count": len(s["steps"]), "meta": s["meta"],
                })
                if len(out) >= limit:
                    break
            kinds = sorted({self._sessions[k]["kind"] for k in sids if k in self._sessions})
        return {"sessions": out, "kinds": kinds}

    def get_session(self, sid):
        """Full detail (including every step) for one session."""
        with self._lock:
            s = self._sessions.get(sid)
            if not s:
                return None
            # shallow copy + copy steps list so the caller can't mutate state
            out = dict(s)
            out["steps"] = list(s["steps"])
            return out


def observe(etype, name=None):
    """Decorator that times an async function as an observability span.
    Records a start + end (with duration_ms) around the call. Never interferes
    with the wrapped function's result or exceptions."""
    import functools

    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            rec = get_recorder()
            nm = name or fn.__name__.lstrip("_")
            t0 = time.time()
            rec.record(etype, name=nm, phase="start")
            ok = True
            try:
                return await fn(*args, **kwargs)
            except Exception:
                ok = False
                raise
            finally:
                rec.record(etype, name=nm, phase="end",
                           duration_ms=int((time.time() - t0) * 1000), ok=ok)
        return wrapper
    return deco


_rec = None
_rec_lock = threading.Lock()


def get_recorder():
    global _rec
    if _rec is None:
        with _rec_lock:
            if _rec is None:
                _rec = Recorder()
    return _rec
