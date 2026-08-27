"""Stranger auto-enrollment brain (Kiki side).

The face-recognition server auto-enrolls a lone, clear unknown face (gives it a
temporary name, captures images, trains) and publishes `unknown_face_enrolled`
with a face crop. This module reacts to that:

  1. (rate-limited) ask Gemini to describe the face + clothing AND check whether
     this is actually one of the people we already know (a duplicate temp, or a
     known person who was mis-read as Unknown);
  2. depending on that verdict: forget the duplicate, merge into the known
     person (retrain with more images), or persist a brand-new temp person in the
     knowledge base with their description + a saved face thumbnail;
  3. show the face + name on the OLED, and queue Kiki to ask their real name when
     it next has a calm moment.

Everything here runs off the face-event listener as background work.
"""
import asyncio
import base64
import json
import os
import re
import time
from collections import deque
from datetime import datetime

from tools_and_config.config_loader import get_full_config

# --- Gemini rate limiter: at most N enroll-time Gemini calls per 2 hours ---
_GEMINI_WINDOW_SECONDS = 2 * 60 * 60
_gemini_call_times: deque = deque()

# Remember which temp people we've already prompted Kiki to ask about, so a
# person standing in view doesn't re-trigger the ask every few seconds.
_last_asked: dict = {}
_ASK_COOLDOWN_SECONDS = 600.0


def _auto_cfg() -> dict:
    return (get_full_config().get("face_events", {}) or {}).get("auto_enroll", {}) or {}


def _temp_prefix() -> str:
    return _auto_cfg().get("temp_name_prefix", "guest")


def is_temp_identity(name: str) -> bool:
    """True if `name` looks like an auto-assigned temporary tag (guest_...)."""
    return bool(name) and name.startswith(_temp_prefix() + "_")


def _faces_dir() -> str:
    cfg = _auto_cfg()
    d = cfg.get("faces_dir", "faces")
    if not os.path.isabs(d):
        # Relative to the KikiFast repo root (two levels up from robot/).
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), d)
    os.makedirs(d, exist_ok=True)
    return d


def face_thumb_path(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return os.path.join(_faces_dir(), f"{safe}.png")


def _rate_ok() -> bool:
    limit = int(_auto_cfg().get("max_gemini_calls_per_2h", 15))
    now = time.time()
    while _gemini_call_times and now - _gemini_call_times[0] > _GEMINI_WINDOW_SECONDS:
        _gemini_call_times.popleft()
    return len(_gemini_call_times) < limit


def _record_gemini_call():
    _gemini_call_times.append(time.time())


def _save_thumb(name: str, image_b64: str) -> str:
    """Persist the face crop as a PNG for the OLED. Returns the path or ""."""
    if not image_b64:
        return ""
    try:
        raw = base64.b64decode(image_b64)
        path = face_thumb_path(name)
        with open(path, "wb") as f:
            f.write(raw)
        return path
    except Exception as e:
        print(f"[Stranger] thumbnail save failed: {e}")
        return ""


async def ensure_face_thumb(person_name: str, controller) -> str:
    """Return a usable thumbnail path for `person_name`, fetching a live face crop
    from the server and saving it if we don't already have one. Returns "" if no
    image could be obtained. This is why the OLED shows a real face instead of a
    "?" even for people enrolled while a crop wasn't captured."""
    from core.brain.knowledge_base import get_knowledge_base, save_knowledge_base
    try:
        kb = get_knowledge_base()
        p = kb.get_person(person_name) or {}
        existing = p.get("face_thumb") or face_thumb_path(person_name)
        if existing and os.path.exists(existing):
            return existing
    except Exception:
        pass
    # No stored thumbnail → grab whatever face is in view right now.
    try:
        b64 = await controller.get_face_crop()
        if b64:
            path = _save_thumb(person_name, b64)
            if path:
                try:
                    kb = get_knowledge_base()
                    person = kb.get_person(person_name)
                    if person is not None:
                        person["face_thumb"] = path
                        save_knowledge_base()
                except Exception:
                    pass
                return path
    except Exception as e:
        print(f"[Stranger] ensure_face_thumb failed: {e}")
    return ""


def _known_people_for_dedup() -> list:
    """People to compare a new stranger against: recent temp strangers + known
    people who have an appearance on file. Returns [{name, kind, appearance}]."""
    from core.brain.knowledge_base import get_knowledge_base
    kb = get_knowledge_base()
    out = []
    for name, info in (kb.data.get("people", {}) or {}).items():
        if name == "Kiki":
            continue
        appearance = info.get("appearance") or ""
        clothing = info.get("clothing") or ""
        desc = (appearance + " " + clothing).strip()
        if info.get("is_temp_name"):
            out.append({"name": name, "kind": "temp", "appearance": desc})
        elif desc:
            out.append({"name": name, "kind": "known", "appearance": desc})
    return out


def _gemini_describe_and_match(image_b64: str) -> dict:
    """Blocking Gemini call. Returns a dict:
       {appearance, clothing, match_name, match_kind ('none'|'temp'|'known')}.
    Best-effort: any failure returns an empty/none result."""
    from core.brain.generate_llm_resp import generate

    people = _known_people_for_dedup()
    people_block = "\n".join(f"- {p['name']} ({p['kind']}): {p['appearance']}" for p in people) or "(none)"
    prompt = (
        "You are helping a home robot remember people it meets. Look at the face in the image.\n"
        "1) Briefly describe the person's FACE/appearance (age range, gender presentation, hair, "
        "glasses, facial hair, distinguishing features).\n"
        "2) Briefly describe their CLOTHING (colors, type).\n"
        "3) Decide whether this is one of the people already on file below, or a NEW person.\n\n"
        f"People already on file:\n{people_block}\n\n"
        "Respond with ONLY a JSON object, no prose:\n"
        '{"appearance": "...", "clothing": "...", "match_name": "<name or empty>", '
        '"match_kind": "none|temp|known"}\n'
        "Use match_kind 'temp' if it matches someone tagged (temp), 'known' if it matches a "
        "(known) person, else 'none'. Only claim a match if you are confident it is the SAME person."
    )
    try:
        _record_gemini_call()
        text = generate(prompt, b64_image=image_b64, thinking_level="LOW",
                        purpose="vision", cloud_category="face_enroll")
        if not text:
            return {}
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        return {
            "appearance": (data.get("appearance") or "").strip(),
            "clothing": (data.get("clothing") or "").strip(),
            "match_name": (data.get("match_name") or "").strip(),
            "match_kind": (data.get("match_kind") or "none").strip().lower(),
        }
    except Exception as e:
        print(f"[Stranger] Gemini describe failed: {e}")
        return {}


async def process_enrolled_event(event: dict, controller, stt_queue, idle_mgr):
    """Handle an `unknown_face_enrolled` event end-to-end (background)."""
    temp_name = event.get("temp_name")
    image_b64 = event.get("image_b64") or ""
    if not temp_name:
        return
    print(f"[Stranger] Enrolled temp person '{temp_name}' — analyzing…")

    loop = asyncio.get_event_loop()

    # 1) Gemini describe + de-dup (rate-limited; skip gracefully if over budget).
    info = {}
    if image_b64 and _rate_ok():
        info = await loop.run_in_executor(None, _gemini_describe_and_match, image_b64)
    else:
        if image_b64:
            print("[Stranger] Gemini enroll budget exhausted — enrolling without description.")

    match_name = info.get("match_name") or ""
    match_kind = info.get("match_kind") or "none"

    # 2a) Duplicate of an existing TEMP stranger → undo this enrollment.
    if match_kind == "temp" and match_name and match_name != temp_name:
        print(f"[Stranger] '{temp_name}' is a duplicate of temp '{match_name}' — forgetting it.")
        try:
            await controller.forget_person(temp_name)
        except Exception as e:
            print(f"[Stranger] forget failed: {e}")
        return

    # 2b) Actually a KNOWN person mis-read as Unknown → merge + retrain (more data).
    if match_kind == "known" and match_name:
        print(f"[Stranger] '{temp_name}' is really known '{match_name}' — merging + retraining.")
        try:
            await controller.merge_into(match_name, temp_name)
            from core.brain.knowledge_base import get_knowledge_base, save_knowledge_base
            kb = get_knowledge_base()
            kb.add_note_to_person(
                match_name,
                f"Re-trained with extra photos on {datetime.now():%Y-%m-%d %H:%M} "
                f"(was briefly mis-recognized as a stranger).")
            save_knowledge_base()
        except Exception as e:
            print(f"[Stranger] merge failed: {e}")
        return

    # 2c) Genuinely NEW person → persist a temp identity in the knowledge base.
    appearance = info.get("appearance") or "a person I haven't been introduced to yet"
    clothing = info.get("clothing") or ""
    now = datetime.now()
    thumb = _save_thumb(temp_name, image_b64)
    try:
        from core.brain.knowledge_base import get_knowledge_base, save_knowledge_base
        kb = get_knowledge_base()
        kb.add_person(
            temp_name,
            relationship="stranger",
            appearance=appearance,
            is_temp_name=True,
            pending_real_name=True,
            clothing=clothing,
            first_met=now.isoformat(),
            face_thumb=thumb,
        )
        kb.add_note_to_person(
            temp_name,
            f"Met on {now:%A %d %B %Y at %H:%M}; wearing {clothing or 'unknown clothes'}. "
            f"Temporary name until I learn their real one.")
        save_knowledge_base()
    except Exception as e:
        print(f"[Stranger] KB save failed: {e}")

    # 3) OLED: show the freshly met face + its temporary name. If the enroll
    # event carried no crop, grab a live one from the server so it's never a "?".
    try:
        if not thumb and not image_b64:
            thumb = await ensure_face_thumb(temp_name, controller)
        from core.oled_display import oled_manager
        oled_manager.show_face(temp_name, image=thumb or image_b64 or None, subtitle="meeting…")
    except Exception as e:
        print(f"[Stranger] OLED show_face failed: {e}")

    # 4) Preserve one next-turn note in case the immediate question cannot fire.
    try:
        from core.brain.unified_idle_mind import set_next_turn_note
        set_next_turn_note(
            text=(
                f"Ask {temp_name} their real name — we just met "
                f"({clothing or 'unknown outfit'}, {now:%d %b %H:%M})."
            ),
            reason="A newly enrolled person still needs a real name.",
            expires_hours=6,
        )
    except Exception as e:
        print(f"[Stranger] next-turn note failed: {e}")

    # 5) If Kiki is free right now, ask immediately (they're still in view).
    maybe_ask_real_name(temp_name, stt_queue, idle_mgr)


def maybe_ask_real_name(person_name: str, stt_queue, idle_mgr) -> bool:
    """Queue a proactive 'ask your real name' turn if `person_name` is a temp
    person still awaiting their real name and Kiki is calm. Rate-limited per
    person. Returns True if queued.

    Robust to a missed `unknown_face_enrolled` event: if the face DB recognises
    a 'guest_...' tag that Kiki has no (or an incomplete) record for — e.g. the
    enroll PUB message was dropped because Kiki wasn't subscribed yet — we stub a
    pending record here so Kiki still asks for the real name."""
    print(f"[Stranger] maybe_ask_real_name('{person_name}') — evaluating…")
    try:
        from core.brain.knowledge_base import get_knowledge_base, save_knowledge_base
        kb = get_knowledge_base()
        person = kb.get_person(person_name)
        pending = bool(person and person.get("pending_real_name"))
        if not pending and is_temp_identity(person_name):
            # Recognised a temp tag with no/incomplete memory → treat as pending.
            print(f"[Stranger] '{person_name}' is a temp tag with no pending record — stubbing one.")
            kb.add_person(person_name, relationship="stranger",
                          is_temp_name=True, pending_real_name=True)
            kb.add_note_to_person(
                person_name,
                "Recognised by their temporary face tag but I don't have their "
                "real name yet.")
            save_knowledge_base()
            pending = True
        if not pending:
            print(f"[Stranger] '{person_name}' not pending a real name → not asking.")
            return False
    except Exception as e:
        import traceback
        print(f"[Stranger] maybe_ask_real_name KB step failed: {e}")
        traceback.print_exc()
        return False

    # Don't interrupt active thinking or spam the same person.
    if idle_mgr is not None and getattr(idle_mgr, "is_thinking", False):
        print(
            f"[Stranger] Unified Idle Mind is active → "
            f"deferring the ask for {person_name}.")
        return False
    now = time.time()
    since = now - _last_asked.get(person_name, 0.0)
    if since < _ASK_COOLDOWN_SECONDS:
        print(f"[Stranger] Asked {person_name} {int(since)}s ago (< {int(_ASK_COOLDOWN_SECONDS)}s cooldown) → skipping.")
        return False
    _last_asked[person_name] = now

    try:
        stt_queue.put_nowait(("meet_stranger", person_name))
        print(f"[Stranger] ✅ Queued real-name ask for {person_name}")
        return True
    except Exception as e:
        print(f"[Stranger] could not queue ask: {e}")
        return False


def show_crowd_notice():
    """Note that a crowd was detected.

    This used to take the OLED over with a 'too many people, come one by one'
    card. `crowd_detected` fires constantly in a normal room, so the notice was
    up far more often than Kiki's actual face — the state has been removed and
    this is now just a log line.
    """
    print("[Stranger] Crowd detected (enrollment paused; display unchanged).")
