"""
Instant-vision path — live-image questions answered by Groq's Qwen VLM.
=======================================================================

When the user asks something that is genuinely ABOUT what Kiki is looking at
*right now* ("look at this smartphone, should I buy it?", "does my shirt look
good?", "what am I holding?"), the local llama.cpp speaking box is the wrong
tool: vision injection to it is disabled on the speaking path
(``core/llm._normalize_messages_for_local`` drops image parts) and its single
slot must stay warm for text latency. Instead we grab a fresh camera frame and
STREAM the answer from Groq's multimodal ``qwen/qwen3.6-27b`` — real-time look,
real-time spoken reply, box untouched.

Rate-limit contract
-------------------
``qwen/qwen3.6-27b`` on Groq is capped at **8K TPM** (tokens per minute). So the
whole request — system prompt + trimmed history/summary + the image + the
completion — is built to sit under that ceiling: the personality system prompt
and the current question are always kept, older turns are dropped oldest-first
to fit ``max_context_tokens`` (see :func:`build_capped_messages`), and the
completion is bounded by ``max_completion_tokens``.

Key source
----------
The Groq key is taken from ``GROQ_API_KEY_LIST`` in ``.env`` (a JSON array — the
same rotated key pool ``core/brain/generate_llm_resp.py`` uses), falling back to
the single ``GROQ_API_KEY`` if the list is empty. Keys are tried in order so a
throttled key rolls over to the next.
"""

import os
import re
import json
import time
import threading
import itertools
import base64

import requests

from tools_and_config.config_loader import get_llm_config
from core.brain.token_counter import count_tokens

try:
    from groq import Groq
except Exception:  # groq SDK missing → feature disables itself gracefully
    Groq = None

# --- Key pool (mirrors generate_llm_resp): rotated list, single-key fallback ---
_GROQ_KEY_LIST = []
try:
    _GROQ_KEY_LIST = json.loads(os.getenv("GROQ_API_KEY_LIST", "[]")) or []
except Exception:
    _GROQ_KEY_LIST = []
_single = os.getenv("GROQ_API_KEY")
if _single and _single not in _GROQ_KEY_LIST:
    _GROQ_KEY_LIST.append(_single)

# Keys that returned 401 this session — skipped so a dead key in the pool doesn't
# cost a wasted round trip on every single request.
_DEAD_KEYS = set()

# Per-key 429 cooldown: {api_key: unix_ts_when_budget_returns}. Each key is a
# SEPARATE Groq org with its own 8K tokens/minute budget, so the pool's real
# capacity is len(keys) * 8K. Two things make that capacity usable at low latency:
#   * round-robin (_KEY_CURSOR) spreads load instead of always draining key[0];
#   * cooldowns skip a key we KNOW is exhausted, so we don't pay a wasted 429
#     round trip (~0.3-0.5s each) probing it on every request.
_KEY_COOLDOWN = {}
_KEY_CURSOR = itertools.count()
_KEY_LOCK = threading.Lock()

# Hard per-request ceiling. Without it a stalled connection can hang the speaking
# turn; the local box (or an apology) is far better than dead air.
_REQUEST_TIMEOUT_S = 20.0

# Upper bound on a 429 bench. Groq's token window is per-MINUTE, but the API has
# occasionally returned retry-after values of ~30 minutes; honouring those
# literally sidelines a key that actually recovers in ~60s, which then forces
# every later turn to probe already-benched keys (measured: 3-8s of wasted 429
# round trips per turn). Capping means the worst case is one cheap re-probe a
# minute and a half — self-correcting rather than sticky.
_MAX_BENCH_S = 90.0


def _retry_after_seconds(exc, default=20.0):
    """Seconds until a 429'd key regains budget (retry-after header, else the
    'try again in 8.5s' text in Groq's message, else a conservative default)."""
    try:
        header = exc.response.headers.get("retry-after")
        if header:
            return max(1.0, float(header))
    except Exception:
        pass
    m = re.search(r"try again in ([0-9.]+)(ms|s|m)\b", str(exc))
    if m:
        value = float(m.group(1))
        unit = m.group(2)
        return max(1.0, value / 1000 if unit == "ms" else value * 60 if unit == "m" else value)
    return default


def _ordered_keys():
    """Keys to try, best-first: round-robin start, dead keys dropped, keys in
    429 cooldown pushed to the back (soonest-to-recover first) rather than
    removed — so a fully-throttled pool still makes an attempt instead of
    failing the turn outright."""
    now = time.time()
    with _KEY_LOCK:
        start = next(_KEY_CURSOR)
    live = [k for k in _GROQ_KEY_LIST if k not in _DEAD_KEYS]
    if not live:
        return []
    rotated = [live[(start + i) % len(live)] for i in range(len(live))]
    ready = [k for k in rotated if _KEY_COOLDOWN.get(k, 0) <= now]
    cooling = sorted((k for k in rotated if _KEY_COOLDOWN.get(k, 0) > now),
                     key=lambda k: _KEY_COOLDOWN[k])
    return ready + cooling


# The speaking model emits this tool call to signal "answer from live vision".
# core/llm.py intercepts it and routes the turn here (the smart-model fallback
# for anything the regex fast-path below doesn't catch).
VISION_TOOL_NAME = "look_at_scene"


def _cfg():
    """Live ``llm.instant_vision`` config block (empty dict if absent)."""
    return get_llm_config().get("instant_vision", {}) or {}


def enabled():
    """Instant-vision routing is on AND the SDK + at least one key are present."""
    return bool(_cfg().get("enabled", False)) and Groq is not None and bool(_GROQ_KEY_LIST)


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------
# High-precision: only fire when the utterance is clearly about the LIVE scene
# (visual verb / appearance judgement / "should I buy THIS") so ordinary
# questions keep going to the fast local box. Recall is deliberately traded for
# precision — a miss just answers on the (blind) local path as before.
_INSTANT_IMAGE_RE = re.compile(
    # --- explicit "use your eyes / look" ---
    r"\blook(?:ing)?\s+at\s+(?:this|these|that|my|the|me|him|her|them|here|it)\b"
    r"|\b(?:take|have)\s+a\s+look\b"
    r"|\blook\s+(?:around|over\s+(?:here|there))\b"
    r"|\b(?:check|look)\s+(?:this|these|it)\s+out\b"
    r"|\bcheck\s+out\s+(?:this|these|my|the)\b"
    # --- "what/who do you see", "describe the scene/surroundings" ---
    r"|\b(?:can|could)\s+you\s+see\b"
    r"|\b(?:what|who|how\s+many)\s+(?:do|can)\s+you\s+see\b"
    r"|\b(?:what|who)('?s| is| are)\s+(?:in\s+front\s+of\s+you|around\s+you|"
    r"(?:in|on)\s+the\s+(?:room|table|desk|screen|picture|image|frame))\b"
    r"|\bdo\s+you\s+see\s+(?:this|these|that|my|the|what|it|anyone|anything|me)\b"
    r"|\b(?:describe|tell\s+me\s+(?:about|what))\b[^.?!]*"
    r"\b(?:see|seeing|scene|surroundings?|room|around|view|looking\s+at|front\s+of\s+you)\b"
    r"|\bwhat\s+are\s+you\s+(?:seeing|looking\s+at)\b"
    r"|\bwhat(?:'?s| is| are)\s+(?:this|these|that|going\s+on\s+(?:here|around))\b"
    # --- appearance / holding / showing ---
    r"|\bwhat\s+am\s+i\s+(?:wearing|holding|showing|pointing|doing)\b"
    r"|\b(?:who|what)\s+is\s+(?:this|that)\s+(?:person|guy|man|woman|behind|next)\b"
    r"|\bwhat\s+do\s+you\s+think\s+(?:of|about)\s+(?:this|these|my|it|that)\b"
    r"|\bhow\s+(?:do|does)\s+(?:i|this|these|it|my)\b[^.?!]*\blooks?\b"
    r"|\bdo(?:es)?\s+(?:i|this|these|it|my|the)\b[^.?!]*\blooks?\s+"
    r"(?:good|nice|ok|okay|bad|weird|cool|right|fine|great|ugly|silly|off|better)\b"
    r"|\bmy\s+\w+(?:\s+\w+){0,3}\s+looks?\s+"
    r"(?:good|nice|ok|okay|bad|weird|cool|right|fine|great|ugly|silly|off|better)\b"
    r"|\b(?:rate|rank)\s+(?:my|this|these|it)\b"
    r"|\bhow\s+(?:do|does)\s+i\s+look\b"
    # --- buy/pick something in view, read text, colour ---
    r"|\bshould\s+i\s+(?:buy|get|keep|wear|pick|choose|take)\b[^.?!]*"
    r"\b(?:this|these|it|one|ones)\b"
    r"|\bread\s+(?:this|the|it|that|out)\b"
    r"|\bwhat\s+colou?r\s+(?:is|are)\s+(?:this|these|my|it|that)\b",
    re.IGNORECASE,
)


def is_instant_image_query(text):
    """True when ``text`` is unmistakably a question about the live camera view."""
    text = str(text or "").strip()
    if not text:
        return False
    return bool(_INSTANT_IMAGE_RE.search(text))


# ---------------------------------------------------------------------------
# Message building (8K TPM budget)
# ---------------------------------------------------------------------------
def _flatten_text(msg):
    """Flatten a KikiFast message's content (str OR list-of-parts) to plain text."""
    content = msg.get("content", "")
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return str(content or "").strip()


def _msg_tokens(role, text):
    return count_tokens([{"role": role, "content": text}])


def cap_normalized_messages(messages, max_ctx):
    """Cap an already-normalized [{role, content:str}] list under ``max_ctx``
    tokens for Groq's TPM budget (text-only — no image). Always keeps the first
    system message (personality) and the last user message; fills the remaining
    budget newest-first, dropping the oldest middle turns. Used by the Groq
    SPEAKING path (llm._stream_groq_speaking)."""
    flat = [(m.get("role"), m.get("content", "")) for m in messages
            if m.get("role") and m.get("content")]
    if not flat:
        return list(messages)
    last_user_idx = next(
        (i for i in range(len(flat) - 1, -1, -1) if flat[i][0] == "user"), len(flat) - 1)
    sys_idx = next((i for i, (r, _) in enumerate(flat) if r == "system"), None)
    keep, used = set(), 0
    if sys_idx is not None:
        keep.add(sys_idx)
        used += _msg_tokens(*flat[sys_idx])
    keep.add(last_user_idx)
    used += _msg_tokens(*flat[last_user_idx])
    for i in range(len(flat) - 1, -1, -1):
        if i in keep:
            continue
        cost = _msg_tokens(*flat[i])
        if used + cost > max_ctx:
            continue
        keep.add(i)
        used += cost
    return [{"role": flat[i][0], "content": flat[i][1]}
            for i in range(len(flat)) if i in keep]


def capture_best_frame_b64(cfg=None):
    """Grab the SHARPEST of a few FRESH camera snapshots as base64 JPEG.

    The Hailo server's ``/clean`` endpoint returns the latest unannotated frame
    over a short HTTP request.  Pulling snapshots is important here: OpenCV's
    MJPEG/FFmpeg reader can block for 30 seconds per read while a pipeline is
    restarting, which is catastrophic inside a spoken care turn.  We sample a
    handful of bounded requests and retain the frame with the highest Laplacian
    variance.  The regular Flask snapshot is a bounded fallback when the clean
    frame server is unavailable; neither path can build an MJPEG backlog.
    """
    cfg = cfg if cfg is not None else _cfg()
    n = max(1, int(cfg.get("capture_frames", 4)))
    quality = int(cfg.get("jpeg_quality", 92))
    timeout = max(0.2, float(cfg.get("snapshot_timeout_seconds", 1.5)))
    primary = str(cfg.get(
        "clean_snapshot_url", "http://127.0.0.1:5001/clean"))
    fallback = str(cfg.get(
        "snapshot_fallback_url", "http://127.0.0.1:5000/snapshot"))

    import cv2
    import numpy as np

    errors = []
    for url, count in ((primary, n), (fallback, 1)):
        best, best_sharp, shape = None, -1.0, None
        for _ in range(count):
            try:
                response = requests.get(url, timeout=(timeout, timeout))
                response.raise_for_status()
                frame = cv2.imdecode(
                    np.frombuffer(response.content, dtype=np.uint8),
                    cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError("response was not a decodable JPEG")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
                if sharp > best_sharp:
                    best_sharp, best, shape = sharp, frame, frame.shape
            except Exception as exc:
                errors.append(f"{url}: {exc}")
        if best is None:
            continue
        ok, buf = cv2.imencode(
            ".jpg", best, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            errors.append(f"{url}: JPEG encode failed")
            continue
        print(f"[InstantVision] captured sharpest of {count} snapshots "
              f"(focus={best_sharp:.0f}, {shape[1]}x{shape[0]})")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    print("[InstantVision] snapshot capture failed ("
          + "; ".join(errors[-2:])[:500] + ")")
    return None


def build_capped_messages(messages, image_b64, cfg):
    """Build the Groq chat payload for a live-image turn, capped under the TPM budget.

    Guarantees the personality system prompt and the current user question (with
    the image attached) are always present; fills the remaining
    ``max_context_tokens`` budget with the most recent history/summary,
    oldest-first dropped. ``tool`` messages are skipped. Returns a Groq-format
    ``messages`` list (the last user turn is multimodal text+image_url).
    """
    max_ctx = int(cfg.get("max_context_tokens", 5000))

    # (role, text) pairs, tool role dropped, empties dropped.
    flat = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            continue
        text = _flatten_text(m)
        if not text:
            continue
        flat.append((role, text))

    if not flat:
        flat = [("user", "What do you see?")]

    # Index of the last user turn — that one carries the image and is mandatory.
    last_user_idx = next(
        (i for i in range(len(flat) - 1, -1, -1) if flat[i][0] == "user"), len(flat) - 1
    )

    # The first system message (personality) is mandatory too.
    sys_idx = next((i for i, (r, _) in enumerate(flat) if r == "system"), None)

    keep = set()
    used = 0
    if sys_idx is not None:
        keep.add(sys_idx)
        used += _msg_tokens(*flat[sys_idx])
    keep.add(last_user_idx)
    used += _msg_tokens(*flat[last_user_idx])

    # Greedily add the remaining messages newest→oldest until the budget is gone.
    for i in range(len(flat) - 1, -1, -1):
        if i in keep:
            continue
        cost = _msg_tokens(*flat[i])
        if used + cost > max_ctx:
            continue
        keep.add(i)
        used += cost

    # Emit in chronological order; attach the image to the last user turn.
    data_url = f"data:image/jpeg;base64,{image_b64}"
    instruction = cfg.get("vision_instruction", "")
    out = []
    for i, (role, text) in enumerate(flat):
        if i not in keep:
            continue
        if i == last_user_idx:
            user_text = f"{text}\n\n{instruction}" if instruction else text
            out.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    # detail:high nudges the VLM to use the full-resolution image
                    # (more tiles → can actually read small labels/branding).
                    {"type": "image_url",
                     "image_url": {"url": data_url, "detail": "high"}},
                ],
            })
        else:
            out.append({"role": role, "content": text})
    return out


# ---------------------------------------------------------------------------
# Streaming from Groq (thinking stripped inline)
# ---------------------------------------------------------------------------
def _partial_tag_suffix(text, tag):
    """Length of the longest suffix of ``text`` that is a prefix of ``tag``.

    Lets the streaming think-filter hold back a partial ``<thi`` that might
    become ``<think>`` in the next delta, instead of speaking it."""
    max_len = min(len(text), len(tag) - 1)
    for n in range(max_len, 0, -1):
        if tag.startswith(text[-n:]):
            return n
    return 0


def _think_filter(piece, st):
    """Incrementally drop ``<think>…</think>`` spans from streamed content.

    ``st`` is a mutable dict {"in": bool, "carry": str} carried across calls so
    tags split across SSE deltas are still caught."""
    text = st["carry"] + piece
    st["carry"] = ""
    out = []
    while text:
        if not st["in"]:
            idx = text.find("<think>")
            if idx == -1:
                keep = _partial_tag_suffix(text, "<think>")
                if keep:
                    out.append(text[:-keep])
                    st["carry"] = text[-keep:]
                else:
                    out.append(text)
                text = ""
            else:
                out.append(text[:idx])
                text = text[idx + len("<think>"):]
                st["in"] = True
        else:
            idx = text.find("</think>")
            if idx == -1:
                # Inside a think block: emit nothing, but hold a possible
                # partial closing tag so we can detect it next delta.
                keep = _partial_tag_suffix(text, "</think>")
                st["carry"] = text[-keep:] if keep else ""
                text = ""
            else:
                text = text[idx + len("</think>"):]
                st["in"] = False
    return "".join(out)


def _create_stream(client, model, groq_messages, cfg):
    """Open a streaming completion; retry with a minimal kwarg set if Groq
    rejects an optional param (keeps the feature working across SDK/model
    versions instead of failing the whole call)."""
    base = dict(
        model=model,
        messages=groq_messages,
        stream=True,
        temperature=cfg.get("temperature", 0.6),
        max_completion_tokens=int(cfg.get("max_completion_tokens", 900)),
        top_p=cfg.get("top_p", 0.95),
    )
    # NOTE: qwen/qwen3.6-27b on Groq only accepts "none" or "default" here
    # ("low"/"high" → HTTP 400). "none" disables thinking → lowest latency.
    effort = cfg.get("reasoning_effort", "none")
    fmt = cfg.get("reasoning_format", "")
    rich = dict(base)
    if effort:
        rich["reasoning_effort"] = effort
    if fmt:
        rich["reasoning_format"] = fmt
    if rich == base:
        return client.chat.completions.create(**base)
    try:
        return client.chat.completions.create(**rich)
    except Exception as e:
        # Only a genuine bad-parameter error (400) is worth retrying without the
        # optional kwargs. Auth/rate errors (401/429/…) must propagate so
        # iter_deltas rotates to the next key instead of wasting a second call.
        status = getattr(e, "status_code", None)
        msg = str(e).lower()
        is_param_error = (status == 400) or ("reasoning" in msg) or (
            "must be one of" in msg) or ("unsupported" in msg)
        if not is_param_error:
            raise
        print(f"[InstantVision] optional params rejected ({e}); retrying minimal")
        return client.chat.completions.create(**base)


def iter_deltas(groq_messages, cfg, abort_event=None):
    """Yield thinking-stripped content deltas from Groq for the built messages.

    Tries each key in the rotated pool until one opens a stream. Raises the last
    error if none work (the caller treats a pre-first-token failure as
    "unavailable" and falls back to the local path)."""
    model = cfg.get("model", "qwen/qwen3.6-27b")
    last_err = None
    for api_key in _ordered_keys():
        try:
            # max_retries=0 is CRITICAL. The SDK defaults to 2 retries and, on a
            # 429, SLEEPS for the server's retry-after (measured 11-40s) before
            # re-trying the SAME exhausted key — that alone produced a 50s
            # time-to-first-word. Failing fast lets us roll to the next key
            # instantly; keys are separate orgs with independent TPM budgets, so
            # rotation is what actually recovers, not waiting.
            client = Groq(api_key=api_key, max_retries=0, timeout=_REQUEST_TIMEOUT_S)
            stream = _create_stream(client, model, groq_messages, cfg)
        except Exception as e:
            last_err = e
            status = getattr(e, "status_code", None)
            if status == 401:
                _DEAD_KEYS.add(api_key)
                print("[InstantVision] key invalid (401) — skipping it this session")
            elif status == 429:
                cool = min(_retry_after_seconds(e), _MAX_BENCH_S)
                _KEY_COOLDOWN[api_key] = time.time() + cool
                print(f"[InstantVision] key rate-limited (429) — rotating now, "
                      f"benching it {cool:.0f}s")
            else:
                print(f"[InstantVision] key failed to open stream: {e}")
            continue

        st = {"in": False, "carry": ""}
        for chunk in stream:
            if abort_event is not None and abort_event.is_set():
                try:
                    stream.close()
                except Exception:
                    pass
                return
            try:
                delta = chunk.choices[0].delta
            except (IndexError, AttributeError):
                continue
            piece = getattr(delta, "content", None) or ""
            if not piece:
                continue
            cleaned = _think_filter(piece, st)
            if cleaned:
                yield cleaned
        return  # stream consumed successfully

    raise RuntimeError(f"all Groq keys failed: {last_err}")


# ---------------------------------------------------------------------------
# Standalone image description (used by read_whatsapp_image)
# ---------------------------------------------------------------------------
_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
}


def describe_image_b64(image_b64, question=None, mime="image/jpeg",
                       max_tokens=None):
    """Describe an arbitrary image with the same free Groq VLM ``look_at_scene``
    uses. Unlike ``_stream_instant_vision`` this is NOT a conversation turn: it
    takes a bare image and returns one finished string, so the WhatsApp image
    tool and background agents can call it without touching the speaking path.

    Raises RuntimeError when no key can serve the request (callers surface that
    as a tool error rather than inventing a description).
    """
    if not image_b64:
        raise RuntimeError("no image data")
    cfg = dict(_cfg())
    # This is a one-shot lookup, not a spoken turn — a tight completion budget
    # keeps it inside the shared 8K/min pool that instant vision also draws on.
    cfg["max_completion_tokens"] = int(
        max_tokens or cfg.get("describe_max_tokens", 500))

    ask = str(question or "").strip() or (
        "Describe this image. If it contains text, dates, times, amounts or "
        "any actionable detail, read them out exactly.")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": ask},
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{image_b64}",
                           "detail": "high"}},
        ],
    }]
    parts = [piece for piece in iter_deltas(messages, cfg)]
    text = "".join(parts).strip()
    if not text:
        raise RuntimeError("vision model returned nothing")
    return text


def describe_image_file(path, question=None, max_tokens=None):
    """``describe_image_b64`` for a local file path (what the MCP hands back)."""
    import base64 as _b64
    from pathlib import Path as _Path

    p = _Path(str(path))
    if not p.is_file():
        raise RuntimeError(f"image file not found: {p}")
    mime = _MIME_BY_SUFFIX.get(p.suffix.lower(), "image/jpeg")
    return describe_image_b64(
        _b64.b64encode(p.read_bytes()).decode("ascii"),
        question=question, mime=mime, max_tokens=max_tokens)
