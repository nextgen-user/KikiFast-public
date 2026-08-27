"""
core/oled_log_feed.py — turn Kiki's interesting terminal logs into OLED life.

Every `print()` in KikiFast already flows through stdout. This module taps that
stream (wrapping whatever `sys.stdout`/`stderr` currently is, so it works with or
without the file-logging tee) and, for the *interesting* lines only, pushes a
short, human-friendly item into the OLED's background-activity feed
(`oled_manager.log_activity`). That's what makes the secondary display show real
insight into what Kiki is doing right now — "Alex just appeared", "neck
tracking off", "taking in the scene", "reviewing the last 5 turns" — instead of
any hard-coded filler.

Design:
* A curated allowlist + prettifiers decide what's worth showing; a denylist
  drops noise (per-turn agent chatter, errors, debug dumps, token stats, our own
  [OLED]/[LCD] prints, ...). Anything unmatched is silently ignored.
* The tap is best-effort and completely passive: it never raises into the write
  path and never prints (which would recurse), so it can't affect the program.
* The richer feed items (full idle-thought summaries, reflection talking points,
  worker results, the compacted summary text) are pushed explicitly from their
  own modules; the patterns those produce are intentionally NOT scraped here to
  avoid double entries.
"""

import re
import sys

# Lines we never surface (substring match on the lowercased line). Covers
# errors/warnings, the agent-loop's per-turn chatter (already shown live via the
# OLED state), token/latency stats, prompt/response dumps, and our own plumbing.
_DENY = (
    "error", "warning", "traceback", "exception", "failed", "could not",
    "unable", "timed out", "timeout", "debug", "tokens/s", "ttfw", "http://",
    "https://", "[oled]", "[lcd]", "[logger]", "[main]", "llm turn",
    "calling tool", "tool result", "tool '", "task completed",
    "task failed", "non-json", "json decode", "exhausted", "continue with",
    "llm response:",
    "rate limit", "monitor error", "retry", "injected current time", "prompt",
    "skipping", "aborted", "preempt", "duplicate", "exit code", "stderr",
    "generated summary", "journaled", "worker completed", "warning:",
)

# Specific, hand-tuned transforms (tried first). (regex, KIND, formatter).
_PRETTY = [
    (re.compile(r"'([^']+)' has just appeared"), "SAW",
     lambda m: f"{m.group(1)} just appeared"),
    (re.compile(r"'([^']+)' has left Kiki's view"), "SAW",
     lambda m: f"{m.group(1)} left the view"),
    (re.compile(r"Neck tracking set to (ON|OFF)"), "NECK",
     lambda m: f"neck tracking {m.group(1).lower()}"),
    (re.compile(r"Playing music: (.+)"), "MUSIC",
     lambda m: f"playing {m.group(1)}"),
    (re.compile(r"Now playing (.+)"), "MUSIC",
     lambda m: f"playing {m.group(1)}"),
    (re.compile(r"Reflecting on (\d+) turn"), "REFLECT",
     lambda m: f"reviewing the last {m.group(1)} turns of chat"),
    (re.compile(r"Triggering vision capture"), "VISION",
     lambda m: "having a look around"),
    (re.compile(r"Scene context appended|Silently injected scene"), "VISION",
     lambda m: "taking in the scene"),
    (re.compile(r"Injected autonomous vision|Injected pending vision"), "VISION",
     lambda m: "noting what it sees"),
    (re.compile(r"Injected (\d+) skill"), "MEMORY",
     lambda m: f"loaded {m.group(1)} skill(s) into context"),
    (re.compile(r"Re-injected talking points"), "MEMORY",
     lambda m: "refreshed its talking points"),
    (re.compile(r"Starting peep|listen for \d+s"), "LISTEN",
     lambda m: "quietly listening to the room"),
    (re.compile(r"Music started . beginning choreography"), "DANCE",
     lambda m: "starting to dance"),
    (re.compile(r"Choreography complete|\bDance\b.*Complete"), "DANCE",
     lambda m: "finished dancing"),
    (re.compile(r"Sequence complete"), "MOVE",
     lambda m: "finished moving"),
]

# Generic fallback: tag → KIND. Lines "[Tag] message" with one of these tags get
# their message shown verbatim (cleaned). Tuned to tags whose messages read well.
_TAG_KIND = {
    "Neck": "NECK", "Vision": "VISION", "Face": "VISION", "Camera": "VISION",
    "Peeping": "LISTEN", "Dance": "DANCE", "Move": "MOVE",
    "SelfExtend": "SELF-EXTEND", "KnowledgeBase": "MEMORY", "Timer": "TIMER",
    "Search": "SEARCH", "Context": "MEMORY",
}

_TS_RE = re.compile(r"^\d\d:\d\d:\d\d\.\d+\s+")
_TAG_RE = re.compile(r"^\[([A-Za-z][A-Za-z0-9 _]*)\]\s*(.*)$")


def _clean(text: str) -> str:
    """Drop non-ASCII (emoji won't render on the 1-bit panel) and tidy."""
    text = re.sub(r"[^\x20-\x7e]", "", text)
    return " ".join(text.split()).strip(" -—:·")


def classify(line: str):
    """Return (KIND, text) for an interesting log line, else None."""
    line = _TS_RE.sub("", line).rstrip()
    if not line:
        return None
    low = line.lower()
    if any(bad in low for bad in _DENY):
        return None
    # Specific prettifiers run on ANY line (some interesting prints — e.g.
    # "Playing music: ..." — aren't bracket-tagged).
    for rx, kind, fmt in _PRETTY:
        m = rx.search(line)
        if m:
            text = _clean(fmt(m))
            return (kind, text) if len(text) >= 3 else None
    # Generic tag fallback is limited to bracket-tagged lines.
    if not line.startswith("["):
        return None
    m = _TAG_RE.match(line)
    if m and m.group(1) in _TAG_KIND:
        text = _clean(m.group(2))
        if len(text) >= 6:
            return (_TAG_KIND[m.group(1)], text)
    return None


class _OLEDLogTap:
    """Wraps a text stream; mirrors writes through and scrapes complete lines."""

    def __init__(self, stream):
        self._stream = stream
        self._buf = ""

    def write(self, data):
        # Pass through untouched FIRST — the console/file must never be affected.
        try:
            self._stream.write(data)
        except Exception:
            pass
        try:
            self._buf += data
            if "\n" in self._buf:
                *lines, self._buf = self._buf.split("\n")
                for ln in lines:
                    self._feed(ln)
                # Guard against a pathological no-newline flood.
                if len(self._buf) > 8192:
                    self._buf = self._buf[-1024:]
        except Exception:
            pass

    @staticmethod
    def _feed(line):
        try:
            hit = classify(line)
            if hit:
                from core.oled_display import oled_manager
                oled_manager.log_activity(hit[0], hit[1])
        except Exception:
            pass  # never raise / never print from the write path

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


_installed = False


def install_oled_log_tap():
    """Wrap stdout+stderr so interesting log lines flow to the OLED feed.
    Idempotent and safe to call once at startup after the OLED is imported."""
    global _installed
    if _installed:
        return
    _installed = True
    try:
        sys.stdout = _OLEDLogTap(sys.stdout)
        sys.stderr = _OLEDLogTap(sys.stderr)
    except Exception:
        pass
