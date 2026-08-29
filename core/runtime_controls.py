"""Thread-safe runtime controls for Kiki's spoken settings.

The active personality and follow-up preference intentionally live in memory:
``config.json`` supplies startup defaults and the available modes, while spoken
commands can change them without rewriting the user's configuration file.
"""

import re
import threading
from difflib import SequenceMatcher

from tools_and_config.config_loader import get_full_config


_lock = threading.RLock()
_current_mode = None
_followups_enabled = None
_mode_revision = 0
_HINDI_PROMPT_SUFFIX = "Always output hindi devnagri."


def _settings():
    return get_full_config().get("assistant_modes", {})


def _modes():
    modes = _settings().get("modes", {})
    return modes if isinstance(modes, dict) else {}


def _normalize_mode_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower()).strip("_")


_MODE_NOISE_WORDS = {
    "activate", "back", "be", "change", "go", "in", "into", "me", "mode",
    "personality", "please", "set", "switch", "the", "to", "use", "voice",
}


def _mode_search_forms(value):
    """Return useful normalized forms for fuzzy/contained mode matching."""
    normalized = _normalize_mode_name(value)
    tokens = [token for token in normalized.split("_") if token]
    meaningful = [token for token in tokens if token not in _MODE_NOISE_WORDS]
    cleaned = "_".join(meaningful)
    forms = {normalized, normalized.replace("_", "")}
    if cleaned:
        forms.update({cleaned, cleaned.replace("_", ""), *meaningful})
    return {form for form in forms if form}


def _primary_mode_forms(value):
    """Whole-phrase forms, excluding individual tokens used by fuzzy search."""
    normalized = _normalize_mode_name(value)
    meaningful = [
        token for token in normalized.split("_")
        if token and token not in _MODE_NOISE_WORDS
    ]
    cleaned = "_".join(meaningful)
    forms = {normalized, normalized.replace("_", "")}
    if cleaned:
        forms.update({cleaned, cleaned.replace("_", "")})
    return {form for form in forms if form}


def resolve_mode_name(requested, modes=None):
    """Resolve loose model/ASR text to the exact configured mode key.

    Priority is exact normalized match, then in-string containment, then fuzzy
    similarity. Fuzzy results require both a minimum score and a clear margin
    over the runner-up so two similarly named modes are never chosen at random.
    """
    available = modes if isinstance(modes, dict) else _modes()
    if not available:
        return None
    requested_forms = _mode_search_forms(requested)
    requested_primary = _primary_mode_forms(requested)
    if not requested_forms:
        return None

    candidates = []
    for name in available:
        candidate_forms = _mode_search_forms(name)
        candidate_primary = _primary_mode_forms(name)

        if requested_primary & candidate_primary:
            score = 1.0
        else:
            containment = [
                min(len(left), len(right)) / max(len(left), len(right))
                for left in requested_forms
                for right in candidate_forms
                if min(len(left), len(right)) >= 3 and (left in right or right in left)
            ]
            if containment:
                score = 0.90 + 0.09 * max(containment)
            else:
                score = max(
                    SequenceMatcher(None, left, right).ratio()
                    for left in requested_forms
                    for right in candidate_forms
                )
        candidates.append((score, str(name)))

    candidates.sort(key=lambda item: (-item[0], item[1].lower()))
    best_score, best_name = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else 0.0
    if best_score < 0.72:
        return None
    if best_score < 1.0 and best_score - second_score < 0.08:
        return None
    return best_name


def _startup_mode():
    modes = _modes()
    match = resolve_mode_name(_settings().get("active_on_startup", "default"), modes)
    if match is not None:
        return match
    if "default" in modes:
        return "default"
    return next(iter(modes), "default")


def get_active_mode():
    global _current_mode
    with _lock:
        if _current_mode is None:
            _current_mode = _startup_mode()
        return _current_mode


def get_mode_revision():
    with _lock:
        return _mode_revision


def get_active_system_prompt():
    """Return the selected mode prompt, with the legacy prompt as default.

    A null/empty prompt on the ``default`` mode deliberately inherits
    ``llm.system_prompt``. This classifies the existing (large) prompt as the
    default without duplicating it in config.json.
    """
    mode = get_active_mode()
    entry = _modes().get(mode, {})
    prompt = entry.get("system_prompt") if isinstance(entry, dict) else None
    if prompt:
        active_prompt = str(prompt)
    else:
        active_prompt = get_full_config().get(
            "llm", {}
        ).get("system_prompt", "You are Kiki.")

    # English is deliberately a no-op. Hindi adds one final, mode-independent
    # instruction, so it survives default/custom prompts and runtime mode
    # switches without rewriting every configured mode.
    language = str(
        _settings().get("language_on_startup", "english")
    ).strip().casefold()
    if (
        language == "hindi"
        and _HINDI_PROMPT_SUFFIX.casefold() not in active_prompt.casefold()
    ):
        active_prompt = f"{active_prompt.rstrip()}\n\n{_HINDI_PROMPT_SUFFIX}"
    return active_prompt


# Non-conversational context that main.py injects into the prompt. A character
# mode is a different PERSON, so Kiki's memories, sensors and background
# thoughts are not theirs to have: injected verbatim they turn "you are Rohan"
# into a briefing about Alex's IMS portal and Goibibo registrations, which
# no persona instruction can outweigh (the memory block alone is ~10k chars).
_CONTEXT_SOURCES = (
    "memory",       # knowledge base, past-conversation summaries, last session
    "skills",       # installed SKILL.md summaries
    "vision",       # scene descriptions and autonomous vision prompts
    "face_events",  # greetings, "introduce yourself as Kiki", stranger names
    "idle_notes",   # Unified Idle Mind next-turn notes and proactive openers
    "peeping",      # passively overheard room audio
    "care_events",  # confirmed CLIP care observations (drinking, eating, ...)
)


def get_context_policy():
    """Which context sources the ACTIVE mode is allowed to receive.

    Resolution order, most specific last:
      1. every source enabled (today's behaviour, so `default` is untouched)
      2. ``assistant_modes.context_defaults``
      3. ``roleplay: true`` on the mode — the one-word switch that turns the
         whole lot off, because a character should only know what they are
         told in conversation
      4. an explicit per-mode ``context`` dict, for fine-grained control
    """
    policy = {source: True for source in _CONTEXT_SOURCES}
    policy.update({
        key: bool(value)
        for key, value in (_settings().get("context_defaults") or {}).items()
        if key in policy
    })

    entry = _modes().get(get_active_mode(), {})
    if not isinstance(entry, dict):
        return policy

    if entry.get("roleplay"):
        policy = {source: False for source in _CONTEXT_SOURCES}

    policy.update({
        key: bool(value)
        for key, value in (entry.get("context") or {}).items()
        if key in policy
    })
    return policy


def context_enabled(source):
    """True when the active mode may receive this context source."""
    try:
        return get_context_policy().get(source, True)
    except Exception:
        # Context gating must never be what stops Kiki from answering.
        return True


def mode_has_own_character():
    """True when the ACTIVE mode declares its own ``system_prompt``.

    The dividing line between a roleplay character (``rohan``, ``jarvis``) and
    Kiki herself: ``default`` leaves ``system_prompt`` null and inherits the
    ~7.9k-char ``llm.system_prompt``, which already carries the voice, the tag
    rules and the tool discipline. Persona-preserving softeners are worth their
    cost for a 253-char character prompt that would otherwise be drowned out;
    applied to ``default`` they only dilute instructions that are already there.

    Same predicate ``get_persona_reinforcement`` gates on, named separately so
    callers that are not about the prompt block do not read as one.
    """
    entry = _modes().get(get_active_mode(), {})
    if not isinstance(entry, dict):
        return False
    return bool(str(entry.get("system_prompt") or "").strip())


def get_persona_reinforcement():
    """The character, repeated AFTER the protocol and memory context.

    A mode prompt replaces ``llm.system_prompt`` entirely, so for a short
    character like ``rohan`` the tools instruction is ~90% of the first system
    message and is also the last thing in it. Measured on the box, that alone
    flattens the persona into a generic assistant even though the identical
    model plays the character perfectly from the same prompt in a bare chat.

    Repeating the character at a FIXED position at the end of the leading
    system block puts identity closest to the conversation without moving
    per turn, so the KV-cache contract in Codestructure section 4 holds.

    Returns "" when disabled, or for modes that inherit the (already
    persona-heavy) default prompt.
    """
    cfg = _settings().get("persona_reinforce", {})
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        return ""

    entry = _modes().get(get_active_mode(), {})
    character = entry.get("system_prompt") if isinstance(entry, dict) else None
    if not character:
        # `default` inherits llm.system_prompt, which already spends ~7.9k chars
        # making these exact points. Repeating it would only cost prefix tokens.
        if cfg.get("custom_prompt_modes_only", True):
            return ""
        character = get_full_config().get("llm", {}).get("system_prompt", "")
    character = str(character or "").strip()
    if not character:
        return ""

    limit = int(cfg.get("max_character_chars", 700) or 700)
    if len(character) > limit:
        clipped = character[:limit]
        cut = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "))
        character = clipped[:cut + 1] if cut > limit * 0.5 else clipped
    return character + str(cfg.get("text", ""))


def get_active_voice():
    entry = _modes().get(get_active_mode(), {})
    return str(entry.get("voice", "") or "").strip().lower() if isinstance(entry, dict) else ""


def apply_active_voice():
    """Apply the configured voice for the current mode to local TTS."""
    from core import tts

    voice = get_active_voice()
    tts.set_local_voice(voice)
    return voice


def switch_mode(mode):
    """Select a configured mode and apply its voice. Returns spoken tool text."""
    global _current_mode, _mode_revision
    modes = _modes()

    # Accept exact names, names embedded in a phrase, and small ASR/model typos.
    match = resolve_mode_name(mode, modes)
    if match is None:
        available = ", ".join(sorted(modes)) or "default"
        return f"Unknown mode '{mode}'. Available modes: {available}."

    with _lock:
        old_mode = get_active_mode()
        _current_mode = match
        if match != old_mode:
            _mode_revision += 1

    voice = apply_active_voice()
    if match == old_mode:
        return f"Already in {match} mode."
    voice_note = f" using the {voice} voice" if voice else " using the default voice"
    return f"Switched to {match} mode{voice_note}."


def followups_enabled():
    global _followups_enabled
    with _lock:
        if _followups_enabled is None:
            _followups_enabled = bool(_settings().get("followups_enabled_on_startup", True))
        return _followups_enabled


def set_followups(enabled):
    """Enable/disable the post-response open-microphone window."""
    global _followups_enabled
    if isinstance(enabled, str):
        normalized = enabled.strip().lower()
        if normalized in {"true", "on", "enable", "enabled", "yes", "1"}:
            enabled = True
        elif normalized in {"false", "off", "disable", "disabled", "no", "0"}:
            enabled = False
        else:
            return "Say whether follow-ups should be enabled or disabled."
    with _lock:
        _followups_enabled = bool(enabled)
    if _followups_enabled:
        return "Follow-ups enabled. I'll keep listening briefly after I answer."
    return "Follow-ups disabled. I'll listen only after the wake word."


def is_shut_up_command(text):
    """Return True only for a direct request telling Kiki to stop talking.

    Keep this deliberately narrower than a substring check so sentences such as
    "why did they tell you to shut up?" still reach the conversation model.
    """
    utterance = re.sub(r"[^a-z0-9']+", " ", str(text or "").lower()).strip()
    return bool(re.fullmatch(
        r"(?:(?:oh|hey|please)\s+)*(?:kiki\s+)?"
        r"shut(?:\s+the\s+(?:hell|fuck))?\s+up(?:\s+please)?",
        utterance,
    ))


def parse_spoken_control(text):
    """Recognize unambiguous spoken control commands.

    This conservative router makes common commands reliable even when the fast
    speaking model misses a tool call. Less explicit phrasing still goes to the
    model, which has the same tools available.
    """
    utterance = " ".join(str(text or "").strip().lower().split())
    if not utterance:
        return None
    # Do not turn questions about a setting, or explicitly negated phrases,
    # into device actions.
    if re.search(r"^(?:what|why|how|when|where|who)\b", utterance) \
            or re.search(r"\b(?:tell|show|explain)\b.{0,30}\bhow\b", utterance) \
            or re.search(r"\b(?:do not|don't|dont|never)\b", utterance):
        return None

    # Stateful music commands are routed in code because "this song" needs the
    # exact video retained by the player, not a fresh YouTube search guessed by
    # the model.
    if re.fullmatch(
        r"(?:(?:please|kiki)\s+)*(?:like|save|add)\s+(?:this|the current)\s+song"
        r"(?:\s+(?:to|in)\s+(?:my\s+)?liked songs?)?(?:\s+please)?",
        utterance,
    ) or re.fullmatch(
        r"(?:(?:please|kiki)\s+)*add\s+(?:this|the current)\s+song\s+to\s+"
        r"(?:my\s+)?liked songs?(?:\s+please)?",
        utterance,
    ):
        return "like_current_song", {}
    if re.fullmatch(
        r"(?:(?:please|kiki)\s+)*play\s+(?:my\s+|the\s+)?liked songs?"
        r"(?:\s+playlist)?(?:\s+please)?",
        utterance,
    ):
        return "play_liked_songs", {}
    if re.fullmatch(
        r"(?:(?:please|kiki)\s+)*play\s+(?:the\s+|my\s+)?last song"
        r"(?:\s+again)?(?:\s+please)?",
        utterance,
    ):
        return "play_last_song", {}
    if re.fullmatch(
        r"(?:(?:please|kiki)\s+)*(?:play\s+)?(?:the\s+)?next song(?:\s+please)?",
        utterance,
    ):
        return "control_music", {"action": "next"}
    if re.fullmatch(
        r"(?:(?:please|kiki)\s+)*(?:play\s+|go\s+to\s+)?(?:the\s+)?"
        r"previous song(?:\s+please)?",
        utterance,
    ):
        return "control_music", {"action": "previous"}
    if re.fullmatch(
        r"(?:(?:please|kiki)\s+)*pause(?:\s+(?:the\s+)?(?:song|music|playback))?"
        r"(?:\s+please)?",
        utterance,
    ):
        return "control_music", {"action": "pause"}
    if re.fullmatch(
        r"(?:(?:please|kiki)\s+)*(?:resume|continue|unpause|play)"
        r"(?:\s+(?:the\s+)?(?:song|music|playback))?(?:\s+please)?",
        utterance,
    ):
        return "control_music", {"action": "resume"}

    timer_match = re.fullmatch(
        r"(?:(?:please|kiki)\s+)*(?:set|start)\s+(?:a\s+)?timer"
        r"(?:\s+for)?\s+(.+?)(?:\s+please)?",
        utterance,
    )
    if timer_match:
        try:
            from core.media_manager import parse_duration_seconds
            return "set_timer", {"duration": parse_duration_seconds(timer_match.group(1))}
        except ValueError:
            pass

    mode_match = re.search(
        r"\b(?:switch|change|go|set|activate|use)(?: me)?(?: back)?(?: in| into| to)? "
        r"([a-z0-9][a-z0-9 _-]*?) mode\b",
        utterance,
    )
    if mode_match:
        requested_mode = mode_match.group(1).strip()
        exact_mode = resolve_mode_name(requested_mode)
        return "switch_mode", {"mode": exact_mode or requested_mode}

    # Also catch commands where the user/model omitted the trailing word
    # "mode" ("switch to Arinab", "sound like Alex") and pass the exact
    # config key into the tool call.
    has_mode_intent = re.search(
        r"\b(?:switch|activate|become)\b"
        r"|\bchange\b.{0,25}\b(?:mode|voice|personality)\b"
        r"|\b(?:talk|sound|act) like\b",
        utterance,
    )
    if has_mode_intent:
        resolved_mode = resolve_mode_name(utterance)
        if resolved_mode is not None:
            return "switch_mode", {"mode": resolved_mode}

    if re.search(r"\b(?:disable|turn off|stop) (?:the )?follow[ -]?ups?\b", utterance) \
            or re.search(r"\bonly listen (?:for|after) (?:the )?(?:hotword|wake word)\b", utterance):
        return "set_followups", {"enabled": False}
    if re.search(r"\b(?:enable|turn on|resume) (?:the )?follow[ -]?ups?\b", utterance):
        return "set_followups", {"enabled": True}

    exact_volume = re.search(r"\bset (?:the )?volume (?:to )?(\d{1,3})(?: percent|%)?\b", utterance)
    if exact_volume:
        return "adjust_volume", {"action": "set", "amount": int(exact_volume.group(1))}
    increase = re.search(
        r"\b(?:increase|raise|turn up) (?:the )?volume(?: by (\d{1,3}))?\b|\bspeak louder\b",
        utterance,
    )
    if increase:
        args = {"action": "increase"}
        if increase.group(1):
            args["amount"] = int(increase.group(1))
        return "adjust_volume", args
    decrease = re.search(
        r"\b(?:decrease|lower|turn down) (?:the )?volume(?: by (\d{1,3}))?\b"
        r"|\bspeak (?:quieter|softer)\b",
        utterance,
    )
    if decrease:
        args = {"action": "decrease"}
        if decrease.group(1):
            args["amount"] = int(decrease.group(1))
        return "adjust_volume", args

    return None
