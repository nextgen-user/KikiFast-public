"""
Cloud cost guard — rate limits + per-category caps for every CLOUD LLM call.

The single-slot local box is "free"; the cloud (Vertex / Gemini / Groq) costs
money per call. Unified Idle Mind, vision, summaries, face enrollment, and
shared reasoning can all fire cloud requests in the background.

This module is the ONE choke point. Every cloud request funnels through
`core/brain/generate_llm_resp.generate()`, which calls `allow(category)` here
before touching a provider. If the budget is spent the call is denied and the
caller skips (returns None) — exactly the same "task simply skips" behaviour as
when the cloud is unreachable, so nothing downstream breaks.

Two kinds of ceiling, both live-editable from the Web UI (`cloud_limits` block):
  - GLOBAL  : max cloud calls per hour / per day across ALL categories.
  - PER-CATEGORY: the same two windows, scoped to one active category.

A cap of 0 (or missing) means "no limit for this window". Setting
`cloud_limits.enabled = false` disables the guard entirely.

`cloud_limits.kill_switch = true` is the master OFF button: it denies every
background cloud call outright (every category), no matter what the caps say.
Use it to silence ALL paid cloud activity in one toggle. (The speaking-path
emergency fallback in core/llm.py is intentionally exempt so Kiki can still talk
if the local box is down.)

Counters are in-memory sliding windows (timestamps, pruned to 24h). They reset on
restart — this is a cost guard, not accounting.
"""

import threading
import time

from tools_and_config.config_loader import get_full_config

_HOUR = 3600
_DAY = 86400

# Categories that get a dedicated per-category cap row in the UI / config.
KNOWN_CATEGORIES = [
    "idle_mind", "vision", "summary", "reasoning", "face_enroll",
]


class CloudBudget:
    def __init__(self):
        self._lock = threading.Lock()
        self._events: dict[str, list[float]] = {}   # category -> [timestamps]
        self._denied: dict[str, int] = {}           # category -> denied count

    # ----------------------------------------------------------------- config
    def _cfg(self) -> dict:
        return get_full_config().get("cloud_limits", {}) or {}

    @staticmethod
    def _exempt(cfg: dict) -> set:
        """Categories that bypass every cap (but not the kill switch)."""
        raw = cfg.get("exempt_categories") or []
        if isinstance(raw, str):          # tolerate a comma-separated string
            raw = raw.split(",")
        return {str(c).strip() for c in raw if str(c).strip()}

    def _prune(self, now: float):
        cutoff = now - _DAY
        for cat in list(self._events.keys()):
            self._events[cat] = [t for t in self._events[cat] if t >= cutoff]

    @staticmethod
    def _count_since(ts_list, since: float) -> int:
        return sum(1 for t in ts_list if t >= since)

    # ------------------------------------------------------------------ check
    def allow(self, category: str) -> tuple[bool, str]:
        """Check (and, if allowed, RECORD) one cloud call for `category`.
        Returns (allowed, reason)."""
        category = category or "reasoning"
        cfg = self._cfg()
        # Master kill switch: when on, EVERY background cloud call is denied
        # regardless of caps/enabled. The one-button "silence all paid cloud
        # activity" lever (the speaking-path emergency fallback in core/llm.py is
        # deliberately NOT gated by this — Kiki must still be able to talk if the
        # local box goes down).
        if cfg.get("kill_switch", False):
            return self._deny(category, "all cloud calls disabled (kill switch)")
        if not cfg.get("enabled", True):
            return True, "limits disabled"

        now = time.time()
        with self._lock:
            self._prune(now)
            exempt = self._exempt(cfg)
            cat_ts = self._events.get(category, [])

            # An exempt category is genuinely uncapped: zeroing its per-category
            # row is NOT enough, because the GLOBAL window applies to every
            # category. Exempt calls are still recorded (so the usage panel
            # shows them) and still obey the kill switch, but they neither hit
            # a cap nor consume the global budget other features draw on —
            # otherwise "unlimited summaries" would quietly starve the idle mind.
            if category in exempt:
                self._events.setdefault(category, []).append(now)
                return True, "exempt from caps"

            all_ts = [t for cat, lst in self._events.items() if cat not in exempt
                      for t in lst]

            # --- global windows ---
            gph = int(cfg.get("max_calls_per_hour", 0) or 0)
            gpd = int(cfg.get("max_calls_per_day", 0) or 0)
            if gph and self._count_since(all_ts, now - _HOUR) >= gph:
                return self._deny(category, f"global hourly cap reached ({gph}/hr)")
            if gpd and len(all_ts) >= gpd:
                return self._deny(category, f"global daily cap reached ({gpd}/day)")

            # --- per-category windows ---
            pc = (cfg.get("per_category", {}) or {}).get(category, {}) or {}
            cph = int(pc.get("per_hour", 0) or 0)
            cpd = int(pc.get("per_day", 0) or 0)
            if cph and self._count_since(cat_ts, now - _HOUR) >= cph:
                return self._deny(category, f"{category} hourly cap reached ({cph}/hr)")
            if cpd and len(cat_ts) >= cpd:
                return self._deny(category, f"{category} daily cap reached ({cpd}/day)")

            # allowed → record
            self._events.setdefault(category, []).append(now)
            return True, "ok"

    def _deny(self, category: str, reason: str) -> tuple[bool, str]:
        self._denied[category] = self._denied.get(category, 0) + 1
        return False, reason

    # --------------------------------------------------------------- snapshot
    def snapshot(self) -> dict:
        """Current usage per category (last hour / last day / total denials)."""
        now = time.time()
        with self._lock:
            self._prune(now)
            exempt = self._exempt(self._cfg())
            cats = {}
            for cat, lst in self._events.items():
                cats[cat] = {
                    "last_hour": self._count_since(lst, now - _HOUR),
                    "last_day": len(lst),
                    "denied": self._denied.get(cat, 0),
                    "exempt": cat in exempt,
                }
            # categories that were only ever denied (never allowed)
            for cat, n in self._denied.items():
                cats.setdefault(cat, {"last_hour": 0, "last_day": 0, "denied": n,
                                      "exempt": cat in exempt})
            all_ts = [t for lst in self._events.values() for t in lst]
            # `counted_*` is what the global cap is actually compared against;
            # `last_*` is every call that happened. Reporting only the total
            # would read as "171 calls against a cap of 60" and look broken.
            counted_ts = [t for cat, lst in self._events.items() if cat not in exempt
                          for t in lst]
            return {
                "enabled": bool(self._cfg().get("enabled", True)),
                "kill_switch": bool(self._cfg().get("kill_switch", False)),
                "exempt_categories": sorted(exempt),
                "global": {
                    "last_hour": self._count_since(all_ts, now - _HOUR),
                    "last_day": len(all_ts),
                    "counted_last_hour": self._count_since(counted_ts, now - _HOUR),
                    "counted_last_day": len(counted_ts),
                },
                "categories": cats,
            }


_budget: CloudBudget | None = None


def get_cloud_budget() -> CloudBudget:
    global _budget
    if _budget is None:
        _budget = CloudBudget()
    return _budget
