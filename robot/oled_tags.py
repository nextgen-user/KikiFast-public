"""
OLED expression tags for Kiki's face.

Kiki's secondary display is a 128x64 SSD1306 showing a procedurally drawn pixel crab
whose pose encodes what Kiki is doing (`core/oled_display.py`). Two layers drive it:

  1. AUTOMATIC system state — listening / thinking / tool / music / summarizing, pushed by
     the runtime (and by `lcd_display.update_status`'s keyword mapper).
  2. EXPLICIT expressions — the speaking model can emit an inline tag to deliberately set
     its face for what it is saying, e.g.  "that is wild <oled:surprised> I did not expect that".

This module handles layer 2, and is deliberately the same shape as `robot/neck.py`:
tags are parsed out of each streamed sentence, stripped before TTS *and* before the
logging/observability copy, and dispatched asynchronously. The KV-cache contract is
identical to the neck tags — `message_history` keeps the model's reply VERBATIM with the
tags in it (see docs/ARCHITECTURE.md §4 rule 2); only the spoken/clean copy is stripped.

Unlike neck gestures — which are collected for the whole turn and fired once it ends —
an expression is fired by the TTS player on the first audible PCM chunk of the sentence
that carried it, so the face changes exactly when Kiki says those words. It then holds
until the next tag or the end of the turn.

LATENCY RULE: a tag must never open a reply or a sentence. Tokens spent on a tag before
the first real word directly delay time-to-first-word. The prompt note in
`core/oled_display.get_oled_tag_prompt_note()` states this, and `core/llm.py` refuses to
let a leading tag disarm the eager first-sentence flush.
"""

import re

# <oled:love>  <oled:shy>  <oled:surprised>  ...  (names validated against the display)
_OLED_TAG_RE = re.compile(r'<oled:([A-Za-z_]{2,20})>', re.IGNORECASE)

_VALID = None


def _valid_names() -> frozenset:
    """The expressions the display can actually draw. Imported lazily so this
    module stays cheap for callers that never touch the OLED (and so an
    emulated/off-robot import failure degrades to 'no tags' instead of raising)."""
    global _VALID
    if _VALID is None:
        try:
            from core.oled_display import EXPRESSION_STATES
            _VALID = frozenset(EXPRESSION_STATES)
        except Exception:
            _VALID = frozenset()
    return _VALID


def extract_oled_tags(text: str) -> list:
    """Extract valid expression names from LLM response text → ['shy', 'giggle'].

    Unknown names are dropped rather than passed through: a hallucinated
    `<oled:banana>` should leave the face alone, not reset it to idle.
    """
    valid = _valid_names()
    out = []
    for m in _OLED_TAG_RE.finditer(text or ""):
        name = m.group(1).lower()
        if name in valid:
            out.append(name)
    return out


def last_oled_tag(text: str):
    """The expression a sentence should end up showing (the last valid tag in
    it), or None. One face per sentence — mid-sentence flicker reads as a
    glitch, not a mood."""
    tags = extract_oled_tags(text)
    return tags[-1] if tags else None


def strip_oled_tags(text: str) -> str:
    """Remove expression tags before sending to TTS / logging.

    Strips *any* syntactically valid `<oled:name>` — including unknown names —
    so a hallucinated expression is never spoken aloud.
    """
    return _OLED_TAG_RE.sub('', text or '').strip()


def apply_oled(name: str) -> bool:
    """Set the face, honouring the display's own priority rules.

    Returns True if the expression was applied. Never raises: the display is
    best-effort decoration and must not be able to break a speaking turn.
    """
    if not name:
        return False
    try:
        from core.oled_display import oled_manager
        return bool(oled_manager.set_expression(name))
    except Exception as e:
        print(f"[OLED] Warning: could not apply expression '{name}': {e}")
        return False
