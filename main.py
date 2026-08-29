"""
KikiFast Voice Assistant — Main Orchestrator

Flow:
  1. Hotword thread listens in background. STT is started but muted.
  2. On 'heyy', STT unmutes immediately for low latency.
  3. STT listens for user speech.
  4. On endpoint → mute mic, start thinking sounds.
  5. Stream LLM response, extract neck-gesture tags, queue sentences to TTS.
  6. TTSStreamer pre-fetches audio in background.
  7. When first queue ready → stop thinking sounds, play.
  8. Sentences play back-to-back with no gaps (pre-fetched).
  9. After TTS finishes → buffer the exchange for Unified Idle Mind.
  10. Unmuted STT waits up to 10s for another query.
  11. If 10s passes with no speech, STT is muted until next 'heyy'.

Additional background tasks:
  - Face event listener via KikiController (greeting injection)
  - Unified Idle Mind (cloud reflection, research, and deliberate actions)
  - Knowledge base context injection

All modules are imported eagerly for minimum latency.
"""
#YOU CAN FREELY EXPLORE SURROUNDINGS

import asyncio
import signal
import sys
import time
import re
import json
import threading
import subprocess
import concurrent.futures
import datetime
from tools_and_config.tools import should_skip_followup

from hotwords.hotword_recog import HotwordRecognizer

# Logging tee FIRST so every print from every module/thread lands in the log
# file too (when logging.enabled in config.json).
from tools_and_config.config_loader import get_full_config as _get_full_config_early
from tools_and_config.logger import setup_logging
setup_logging(_get_full_config_early())

# --- Eager imports: triggers pre-initialization in each module ---
print("[Main] Initializing modules...")
t0 = time.time()

from tools_and_config.config_loader import get_llm_config, get_full_config
from core.stt import STTEngine
from core.llm import (stream_response, execute_tool_calls, warmup as llm_warmup,
                      register_history, hot_inject, last_turn_used_instant_vision,
                      speaking_is_local, _extract_sentences)
from core.tts import LCDOnlyStreamer, TTSStreamer, get_tts_system_prompt_note
from core import local_llm
from core.gesture_controls import (
    is_output_muted,
    mark_activities_stopped,
    toggle_output_mute,
)
from sound_effects.sound_effects import ThinkingSoundPlayer

# One cloud background mind; local speaking model remains isolated.
from core.brain.knowledge_base import get_knowledge_summary
from core.brain.summary_manager import (
    save_summary, load_latest_conversation,
    save_summary_to_conversations_folder, generate_past_conversations_summary
)
from core.brain.unified_idle_mind import UnifiedIdleMindManager
from core.brain.ambient_listening import AmbientListeningManager
from core.vision.vision_handler import VisionHandler
from core.vision import instant_vision
from robot.face_handler import face_event_listener
from core.brain import token_counter
from robot.neck import extract_neck_tags, strip_neck_tags, apply_neck
from robot.oled_tags import strip_oled_tags
from core.observability import get_recorder
from webui.server import start_webui
from core.workers.worker_manager import get_worker_manager
from core.workers.worker_brain import get_face_history, get_vision_history
from core.self_extend.skill_manager import SkillManager
import random
from core.lcd_display import lcd_manager
from core.oled_display import get_oled_tag_prompt_note, oled_manager
from core.ir_controls import IRControls
from core.runtime_controls import (
    apply_active_voice,
    context_enabled,
    followups_enabled,
    get_active_mode,
    get_active_system_prompt,
    get_context_policy,
    get_mode_revision,
    is_shut_up_command,
)

print(f"[Main] All modules initialized in {time.time() - t0:.2f}s\n")

# Tap stdout/stderr so interesting log lines (face events, neck toggles, vision,
# music, ...) flow onto the OLED's live background-activity feed.
from core.oled_log_feed import install_oled_log_tap
install_oled_log_tap()

# Global reference to stop active TTS gracefully
active_tts_streamer = None

# Filler lines (from config) spoken immediately when the model triggers a tool but
# says nothing itself, so the user never hears dead air while the tool runs.
_TOOL_CALLING_CFG = get_llm_config().get("tool_calling", {})
TOOL_FILLER_ENABLED = _TOOL_CALLING_CFG.get("speak_filler", True)
TOOL_FILLERS = _TOOL_CALLING_CFG.get("fillers", [
    "Let me check that.", "One sec.", "Hold on, looking that up.",
    "Give me a moment.", "Let me find out.", "Checking now.",
])
# Hard ceiling on how long a turn will wait for background tool execution before
# giving up and continuing. Safety net against any tool hanging (e.g. a network
# call with no timeout) — must never deadlock the turn, or wake-word detection
# never re-enables and Kiki becomes unwakeable. Slightly above search_web's 12s.
TOOL_EXEC_TIMEOUT = _TOOL_CALLING_CFG.get("exec_timeout_s", 15.0)
# How many EXTRA tool calls a turn will honour after the first one, when the
# model emits them while answering a tool result. 1 covers the case this exists
# for — the model retrying a call the schema gate rejected — without letting a
# looping model hold the turn, and therefore the wake word, indefinitely.
MAX_FOLLOWUP_TOOL_ROUNDS = int(
    _TOOL_CALLING_CFG.get("max_followup_tool_rounds", 1))
# Some tools legitimately need longer than a lookup. complex_query runs a whole
# multi-step agent (core/brain/action_agent.py), which enforces its own shorter
# internal deadline so it reports partial progress instead of being cut off
# mid-action — being truncated is what turns "I sent it" into a lie.
TOOL_EXEC_TIMEOUT_OVERRIDES = _TOOL_CALLING_CFG.get("exec_timeout_overrides", {})
# Longer, more honest fillers for tools that will visibly take a few seconds.
TOOL_COMPLEX_FILLERS = _TOOL_CALLING_CFG.get("complex_fillers", [
    "On it, this will take a few seconds.", "Alright, let me go and do that.",
    "Give me a moment, working on it.",
])
# Tools whose wait is long enough to deserve the honest filler above.
_SLOW_TOOLS = frozenset(TOOL_EXEC_TIMEOUT_OVERRIDES) | {"complex_query"}


def tool_exec_timeout(calls):
    """Longest configured wait among the tools about to run."""
    timeout = TOOL_EXEC_TIMEOUT
    for call in calls or []:
        override = TOOL_EXEC_TIMEOUT_OVERRIDES.get(call.get("name"))
        if override:
            timeout = max(timeout, float(override))
    return timeout


def tool_result_note(calls, tool_result):
    """The system note that turns a tool result into the spoken answer.

    An ordinary lookup (search_web, get_current_time) returns raw data that the
    model must distil into a sentence or two. `complex_query` is different: the
    action agent already did the work and wrote a finished, spoken-style reply,
    so the old "answer briefly in one or two sentences" instruction was
    actively throwing away the detail the agent was asked to gather.

    This note is the LAST thing before generation on a tool turn, so it is also
    the strongest single influence on it — and it is stored in history, where it
    keeps influencing every later turn. It therefore comes in two forms:

    - A CHARACTER mode (its own `system_prompt`, e.g. `rohan`) gets the
      in-character wording added in 704acd8, because "answer the user directly
      and briefly" is service-desk register that flattened a 253-char persona.
    - `default` gets the DIRECTIVE wording back. Kiki's own 7.9k prompt already
      carries her voice, so the persona softener bought nothing here and cost
      precision: on 2026-07-29 00:10 and 00:11, asked to "play the last song"
      twice more, Kiki replied "Playing Maafi again" with NO tool call either
      time (verified: no tool event in events.jsonl) — copying the prose answer
      this note had produced on the previous turn. The two closing clauses spell
      out what the prose form left implicit: report only this result, and a
      repeat request still needs its own tool call.
    """
    if any(call.get("name") == "update_care_plan" for call in calls or []):
        return (
            "This is the authoritative care-plan result. Say it was saved ONLY if "
            "the result contains SUCCESS. NEEDS_CLARIFICATION means it was not "
            "saved yet; ask only for the missing detail. ERROR means it was not "
            "saved. PARTIAL means it was saved but will not reliably trigger. "
            "Never claim success before or after an error. Stay in your normal "
            "voice and do not mention tools or these instructions:\n" + tool_result
        )
    if any(call.get("name") == "complex_query" for call in calls or []):
        return (
            "You just carried out that multi-step request, and this is the full "
            "outcome. Relay it to Alex conversationally and COMPLETELY — keep "
            "the detail, the names, the times and anything he needs to act on. "
            "Do not shorten it to one line and do not add facts that are not "
            "here. If it says the action did not happen, say so plainly and "
            "never claim it worked. Do not mention tools, agents, or these "
            "instructions, and do not think out loud:\n" + tool_result
        )
    try:
        from core.runtime_controls import mode_has_own_character
        in_character = mode_has_own_character()
    except Exception:
        in_character = False
    if in_character:
        return (
            "Here is the result of a quick lookup you just did. Answer in YOUR OWN "
            "VOICE, fully in character, the way you normally talk. Keep it short "
            "and spoken. Do not mention searching or tools, and do not think out "
            "loud. Say only what this result says; if the user asks for the same "
            "thing again, call the tool again:\n" + tool_result
        )
    return (
        "Here is the result of a quick lookup you just did. Answer the user "
        "directly and briefly in one or two spoken sentences. Report only what "
        "this result says — do not claim anything it does not. This answers the "
        "current request ONLY: if the user asks for the same thing again, call "
        "the tool again rather than repeating this answer. Do not mention "
        "searching or tools, and do not think out loud:\n" + tool_result
    )


def deterministic_care_plan_failure_reply(calls, tool_result):
    """Return a truthful spoken reply for failed care-plan writes.

    Prompt instructions alone did not hold: the live model was handed an
    explicit ``ERROR: No change was saved`` and still said the reminder was
    set. Failed writes therefore bypass another generation entirely.
    """
    if not any(call.get("name") == "update_care_plan" for call in calls or []):
        return None
    result = str(tool_result or "")
    if "NEEDS_CLARIFICATION:" in result:
        if "what time" in result.lower():
            return "अभी यह सेव नहीं हुआ है। कृपया बताइए, मैं आपको कितने बजे याद दिलाऊँ?"
        return "अभी यह सेव नहीं हुआ है। कृपया बाकी जानकारी भी बता दीजिए।"
    if "PARTIAL:" in result:
        return ("यह केयर प्लान में लिखा गया है, लेकिन इसका रिमाइंडर चालू नहीं हो पाया। "
                "इसलिए मैं अभी यह वादा नहीं करूँगी कि यह समय पर बजेगा।")
    if "ERROR:" in result:
        return ("माफ़ कीजिए, यह केयर प्लान में सेव नहीं हुआ। "
                "कृपया समय और काम एक बार फिर बता दीजिए।")
    return None


def is_direct_care_complex_call(calls, user_text=""):
    """Care complex-agent results are already voice-ready; skip a second LLM."""
    complex_calls = [call for call in (calls or [])
                     if call.get("name") == "complex_query"]
    if not complex_calls:
        return False
    try:
        from core.brain.action_agent import is_care_request, _active_care_session
        for call in complex_calls:
            raw = call.get("arguments") or "{}"
            try:
                args = raw if isinstance(raw, dict) else json.loads(raw)
            except Exception:
                args = {}
            if is_care_request(str(args.get("request", "")),
                               str(args.get("context", ""))):
                return True
        return bool(is_care_request(str(user_text or "")) or _active_care_session())
    except Exception:
        return False


def direct_complex_reply(calls, tool_result):
    """Extract one complex agent's finished spoken answer from tool aggregation."""
    calls = list(calls or [])
    if len(calls) != 1 or calls[0].get("name") != "complex_query":
        return ""
    text = str(tool_result or "").strip()
    prefix = "- complex_query:"
    text = text[len(prefix):].strip() if text.startswith(prefix) else text
    failure_marker = "CARE_ACTION_FAILED:"
    return (text[len(failure_marker):].strip()
            if text.startswith(failure_marker) else text)


def queue_voice_ready_text(tts_streamer, text):
    """Queue an already-authored reply sentencewise for synth-ahead/low TTFW."""
    complete, tail = _extract_sentences(str(text or "").strip())
    queued = 0
    for sentence in complete + ([tail] if tail.strip() else []):
        sentence = sentence.strip()
        if sentence:
            tts_streamer.add_sentence(sentence)
            queued += 1
    return queued


_SUMMARY_KIND_PREFIX = {
    "user": "USER: ",
    "kiki": "ASSISTANT: ",
    "tool_call": "[KIKI USED] ",
    "tool_result": "[TOOL RESULT] ",
    "time": "[TIME] ",
    # Carry the PREVIOUS summary forward. Without it each new summary covered
    # only what happened since the last one, so memory reset at every
    # compaction instead of accumulating.
    "memory": "[EARLIER MEMORY] ",
    "context": "[CONTEXT] ",
}


def build_summary_input(history):
    """The conversation text the background summariser is asked to remember.

    This used to keep ONLY user/assistant rows plus the [TIME] anchors and
    `continue` past every other system message — which is exactly where main.py
    files every tool result. So anything Kiki learned by CALLING a tool (the
    music URL, a search hit, an email body, a chat summary) never reached her
    long-term memory: it survived verbatim until the token limit tripped, then
    vanished. That is why "what was that link you found earlier" failed as a
    cliff rather than a fade.

    It also fed the raw <tool_call> tags in as though they were speech, so the
    summariser was partly summarising XML.

    Shared with the action agent through history_view, so the two readers of
    Kiki's history can never drift apart again.
    """
    from core.brain.history_view import render as render_history

    return "\n".join(
        _SUMMARY_KIND_PREFIX.get(rec["kind"], "") + rec["text"]
        for rec in render_history(history)
    )


def pick_tool_fillers(calls):
    """Filler pool matching how long this tool call will actually take.

    "One sec." is fine for a web search; it reads as a stall when the agent is
    about to spend eight seconds sending a file.
    """
    if TOOL_COMPLEX_FILLERS and any(
            call.get("name") in _SLOW_TOOLS for call in calls or []):
        return TOOL_COMPLEX_FILLERS
    return TOOL_FILLERS

# Bridge connector spoken between the model's pre-tool-call speech and the
# tool-result follow-up answer. The follow-up is generated in the background
# while the initial speech is still playing and queued onto the SAME streamer;
# on a slow/loaded box its prefill+gen can outlast the remaining queued audio,
# surfacing as dead air. A short natural connector queued right before the
# follow-up absorbs that latency (and reads as a normal lead-in when the box is
# fast). TTS-only — never written to message_history, so the KV cache is intact.
TOOL_BRIDGE_ENABLED = _TOOL_CALLING_CFG.get("speak_bridge", True)
TOOL_BRIDGES = _TOOL_CALLING_CFG.get("bridges", [
    "Okay, so.", "Right.", "So, here's what I found.", "Alright.",
    "Okay, let's see.", "So.",
])




import asyncio
from kiki_control_client import quick_command

def set_neck_active(state: bool):
    """
    Fire-and-forget NECK-tracking power toggle (formerly the mis-named "motor relay").

    Despite the old name this never controlled chassis wheels — it enables/disables the
    controller's autonomous neck-tracking (`neck_movement` on/off). Kept on the wake/sleep
    path: neck tracking comes alive when Kiki wakes, idles off when it sleeps.

    Runs the (potentially slow/blocking) KikiController call in a daemon thread so it NEVER
    stalls the caller. This matters on the wake path: previously this ran inline on the
    hotword thread and, when KikiController was unreachable, blocked for ~5s before STT
    could unmute — so the user's first words were lost.
    """
    command = "on" if state else "off"
    host = get_full_config().get("controller", {}).get("host", "192.0.2.20")

    def _worker():
        try:
            asyncio.run(quick_command(host=host, neck_movement=command))
            print(f"[Neck] Neck tracking set to {command.upper()} via KikiController")
        except Exception as e:
            print(f"[Neck] Warning: Could not set neck tracking via KikiController: {e}")

    threading.Thread(target=_worker, daemon=True).start()


# ============================================================================
# Speculative turn (start the reply pipeline BEFORE the endpoint commits)
# ============================================================================

class SpeculativeTurn:
    """Runs the speaking pipeline early, on the speculative ASR result that
    resolves inside the STT silence window — LLM stream + TTS synthesis start
    100-300ms before the endpoint commits, but NOTHING plays until the commit
    confirms the exact same transcript (hold_playback). What Kiki says is
    therefore byte-identical to a non-speculative turn; only the start time
    moves. If speech resumes, abort() kills the stream (freeing the box's
    single slot) and the held audio is discarded unheard.

    Tool calls are recorded but NEVER executed speculatively (play_music etc.
    have side effects); the adopting turn starts them at commit."""

    def __init__(self, utterance, messages):
        self.utterance = utterance
        self.messages = messages            # snapshot INCLUDING the user msg
        self.abort_event = threading.Event()
        self.result = {"full": "", "raw": "", "neck": [], "tool_calls": None,
                       "ok": False, "local_fail": False, "vision": False}
        self.done = threading.Event()
        self.t0 = time.time()
        self.tts = TTSStreamer()
        if hasattr(self.tts, "hold_playback"):
            self.tts.hold_playback()
        self.tts.start()
        threading.Thread(target=self._run, daemon=True, name="spec-turn").start()

    def _run(self):
        print(f"[Spec] ▶ speculative turn started: {self.utterance[:60]!r}")
        try:
            for evt, data in stream_response(self.messages,
                                             abort_event=self.abort_event,
                                             local_only=True):
                if self.abort_event.is_set():
                    break
                if evt == "sentence":
                    g = extract_neck_tags(data)
                    if g:
                        self.result["neck"].extend(g)
                    clean = strip_neck_tags(data)
                    if clean:
                        self.tts.add_sentence(clean)
                    self.result["full"] += data + " "
                elif evt == "tool_calls":
                    self.result["tool_calls"] = data
                elif evt == "vision_requested":
                    # Local model asked for live vision. Never adopt this blind
                    # spec — the real turn will re-run and route to Groq.
                    self.result["vision"] = True
                    break
                elif evt == "local_unavailable":
                    self.result["local_fail"] = True
                elif evt == "done":
                    self.result["raw"] = data
                    clean_data = re.sub(r'<tool_call>.*?</tool_call>', '', data,
                                        flags=re.DOTALL).strip()
                    self.result["full"] = clean_data
                    self.result["ok"] = True
        except Exception as e:
            print(f"[Spec] stream error: {e}")
        finally:
            self.done.set()

    def abort(self, reason=""):
        print(f"[Spec] ✖ speculative turn aborted{' (' + reason + ')' if reason else ''}")
        self.abort_event.set()
        try:
            self.tts.abort()
        except Exception:
            pass
        # The spec request extended the KV cache past the real prefix; a
        # coalesced rewarm restores the registered prefix when the box frees up
        # (pollution is benign either way — common-prefix matching keeps it).
        # Only meaningful when the box actually generates replies.
        if speaking_is_local():
            local_llm.schedule_rewarm()


# ============================================================================
# STT Stream Bridge (thread → asyncio.Queue)
# ============================================================================

def stt_stream_worker(stt: STTEngine, event_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """Run STT stream in a thread, push events to asyncio queue."""
    try:
        for event, text in stt.stream():
            loop.call_soon_threadsafe(event_queue.put_nowait, (event, text))
    except Exception as e:
        loop.call_soon_threadsafe(event_queue.put_nowait, ("error", str(e)))


# ============================================================================
# Main Async Loop
# ============================================================================

async def main():
    global active_tts_streamer
    lcd_manager.update_status("Startup")

    cfg = get_llm_config()
    full_config = get_full_config()
    system_prompt = get_active_system_prompt()
    # Provider-specific TTS constraints (e.g. local TTS supports only a fixed
    # tag set). Appended BEFORE warmup so the cached prefix matches real turns.
    system_prompt += get_tts_system_prompt_note()
    # The <oled:> expression vocabulary, generated from what the display can
    # actually draw. Appended here (not in config.json) so it reaches every
    # persona mode — mode prompts REPLACE llm.system_prompt wholesale. Static
    # text, so it becomes part of the warmed prefix and costs no per-turn
    # prefill.
    system_prompt += get_oled_tag_prompt_note()
    mode_revision = get_mode_revision()
    try:
        mode_voice = apply_active_voice()
        print(f"[Mode] Started in '{get_active_mode()}' mode "
              f"with voice '{mode_voice or 'default'}'.")
    except Exception as e:
        print(f"[Mode] Could not apply startup voice: {e}")
    agent_config = full_config.get("agent", {})
    prompts_config = full_config.get("prompts", {})

    # (Warmup is fired later, once the FULL message_history prefix — system prompt
    # + memory/KB/skills context — is built, so the entire prefix gets cached.)

    # --- STT and Async state ---
    loop = asyncio.get_running_loop()
    stt_queue = asyncio.Queue()

    # 0 (not now): forces a time injection on the FIRST user turn so Kiki always
    # has a time/date anchor from the start of a conversation (it used to wait
    # 5 min, so early on it would say it didn't know when the chat began).
    # Time anchor + workers context state now lives on idle_mgr (the idle thread
    # pre-injects + re-warms it between turns; main.py's turn path delegates via
    # idle_mgr.maybe_inject_time).
    summarizing = False
    turn_active = False  # a speaking turn is mid-flight (user msg appended, assistant not yet)
    turn_counter = 0
    # Guided exercise: how many care turns in a row ran with no reply from the
    # person. Bounded so a routine can lead itself through a set of movements
    # without stalling, but can never become Kiki talking to an empty room
    # indefinitely. Any real utterance resets it.
    care_auto_continues = 0
    current_system_context = ""
    # Cooperative kill switch for the CURRENT speaking turn's LLM stream(s):
    # the IR push-to-talk barge-in sets it so generation stops the instant the
    # user's hand lands on a sensor. Re-created fresh at every turn start so a
    # stale set() can never abort the next turn.
    turn_abort_event = threading.Event()
    ir_idle_requested = threading.Event()
    mute_interrupted_turn = threading.Event()
    spec_turn = None
    ir_controls = None  # assigned below; referenced by mute_stt's hold guard

    # Initialize task variables for safe cleanup in finally block
    face_task = None
    periodic_q_task = None
    peeping_task = None

    # --- Peeping Config ---
    peeping_cfg = full_config.get("peeping", {})
    peeping_interval = peeping_cfg.get("interval_seconds", 0)
    peeping_listen_duration = peeping_cfg.get("listen_duration_seconds", 10)
    peeping_active = False
    peep_sentences = []

    # Always-listen is a startup mode because it changes ownership of the single
    # STT event stream.  The cloud cadence/tunables live in always_listen_config.
    ambient_listener = AmbientListeningManager(full_config)
    always_listen_enabled = ambient_listener.enabled
    query_listening = threading.Event()
    ambient_listener.start(loop)

    # --- Vision Injection Config ---
    vision_injection_cfg = full_config.get("vision_injection", {})
    vision_injection_enabled = vision_injection_cfg.get("enabled", False)
    main_vision_cfg = vision_injection_cfg.get("main_llm", {})
    main_vision_enabled = vision_injection_enabled and main_vision_cfg.get("enabled", False)
    main_vision_every_n = main_vision_cfg.get("every_n_turns", 3)
    # When true, run image recognition after EVERY message (forced), not just on
    # the periodic timer — so Kiki always has a fresh view of the scene.
    vision_every_message = vision_injection_enabled and vision_injection_cfg.get("every_message", False)

    # --- Load Contexts ---
    print("[Main] Loading persistent memory contexts... this may take a moment.")
    
    # 1. Knowledge Base
    kb_summary = get_knowledge_summary(max_lines=full_config.get("knowledge_base", {}).get("max_context_lines", 50))
    
    # 2. Past Conversation Summaries — loaded in the BACKGROUND (this LLM call
    # used to block startup for seconds; the summary is spliced into the system
    # context when ready and the prefix re-warmed).
    past_conversations_count = agent_config.get("past_conversations_count", 5)
    past_summary = None

    # 3. Latest Conversation (very last session)
    past_conversation = load_latest_conversation()

    # The active mode decides which of these it is allowed to see. A roleplay
    # character is a different PERSON: Kiki's knowledge base and past sessions
    # are ~10k chars of Alex's life, and injected verbatim they turn "you are
    # Rohan" into a briefing about the IMS portal no persona line can outweigh.
    context_policy = get_context_policy()
    disabled = [name for name, on in sorted(context_policy.items()) if not on]
    if disabled:
        print(f"[Context] '{get_active_mode()}' mode suppresses: {', '.join(disabled)}")

    # Build context components
    additional_context = ""
    if kb_summary and context_policy.get("memory", True):
        kb_context_prompt = prompts_config.get("knowledge_context", "Things you remember about people and the world (your long-term memory):\n{knowledge_summary}")
        additional_context += f"\n\n## YOUR MEMORY (Knowledge Base)\n{kb_context_prompt.format(knowledge_summary=kb_summary)}"
    if past_summary and context_policy.get("memory", True):
        prev_sum_prompt = prompts_config.get("previous_summary_context", "Your memories from past conversations:\n{summary}")
        additional_context += f"\n\n## PAST CONVERSATIONS SUMMARY\n{prev_sum_prompt.format(summary=past_summary)}"
    if past_conversation and context_policy.get("memory", True):
        additional_context += f"\n\n## MOST RECENT CONVERSATION\n{past_conversation}"

    # --- Skills injection ---
    se_cfg = full_config.get("self_extend", {})
    if (se_cfg.get("enabled", True) and se_cfg.get("auto_inject_skills", True)
            and context_policy.get("skills", True)):
        try:
            skill_manager = SkillManager()
            skills_summary = skill_manager.get_skills_summary()
            if skills_summary:
                print(skills_summary)
                additional_context += f"\n\n{skills_summary}"
                print(f"[SelfExtend] Injected {len(skill_manager.list_skills())} skill(s) into context")
        except Exception as e:
            print(f"[SelfExtend] Warning: Could not load skills: {e}")
    # NOTE: We deliberately do NOT tell the model how to handle tools here. Doing
    # so made it "plan out loud" (spoken reasoning like "Plan: 1. Say a filler…").
    # Instead the model just emits a silent tool call, and main.py speaks a canned
    # filler (TOOL_FILLERS) on its behalf — clean and reliable.
    current_system_context = additional_context.strip()

    # Conversation history
    message_history = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                    # "cache_control": {"type": "ephemeral"}
                }
            ]
        }
    ]
    if current_system_context:
        message_history.append({"role": "system", "content": current_system_context})

    def sync_mode_prompt(rewarm=True):
        """Apply a spoken mode change to the live prompt at a safe boundary."""
        nonlocal mode_revision
        latest_revision = get_mode_revision()
        if latest_revision == mode_revision:
            return False
        prompt = (get_active_system_prompt() + get_tts_system_prompt_note()
                  + get_oled_tag_prompt_note())
        first_content = message_history[0].get("content")
        if isinstance(first_content, list) and first_content:
            first_content[0]["text"] = prompt
        else:
            message_history[0]["content"] = prompt
        mode_revision = latest_revision
        new_mode = get_active_mode()
        print(f"[Mode] System prompt changed to '{new_mode}' mode.")
        # Senior Citizen Mode: (de)activate care workers at this cache-safe boundary.
        try:
            from core.senior.senior_care_manager import get_senior_care_manager
            _mgr = get_senior_care_manager()
            if _mgr is not None:
                if new_mode == "senior":
                    _mgr.activate()
                elif _mgr.is_active():
                    _mgr.deactivate()
        except Exception as _senior_err:
            print(f"[SeniorCare] mode-change hook failed: {_senior_err}")
        if rewarm:
            register_history(message_history)
        return True

    # Session-start time anchor: baked into the prefix (append-only, KV-safe) so
    # from the very first words Kiki knows when this conversation began — it kept
    # saying it didn't know when the chat started.
    _session_start_str = datetime.datetime.now().strftime("%I:%M %p on %A, %B %d, %Y")
    message_history.append({
        "role": "system",
        "content": f"This conversation began at {_session_start_str}. "
                   f"Track how much time passes during it."
    })

    # SWA NOTE: Only ONE warmup should fire at startup to avoid cache
    # invalidation from competing prefill requests. The single warmup fires
    # AFTER both the past-summary and startup workers have injected their
    # context (see the warmup after worker_manager.fire_event below).
    # Previously, up to THREE warmups raced at startup, each invalidating
    # the previous one's SWA cache → the first real user query paid a full
    # ~50s prefill.

    # Load past-conversations summary in the background and splice it in when ready.
    # prefer_cache=True: returns the (possibly stale) cached summary instantly so
    # the warmup isn't blocked behind a ~60s LLM call on the box. A background
    # refresher below regenerates the cache for the NEXT startup without touching
    # the live context (mid-session splices invalidate the KV cache).
    async def load_past_summary_bg():
        nonlocal past_summary, current_system_context
        if past_conversations_count <= 0:
            return
        if not context_enabled("memory"):
            # Nothing to splice, and generating it would burn a cloud/box call
            # for text this mode is never allowed to see.
            print("[Main] Past-summary load skipped (mode suppresses memory).")
            return
        try:
            print(f"[Main] Loading past conversations summary (N={past_conversations_count})")
            s = await generate_past_conversations_summary(past_conversations_count,
                                                          prefer_cache=True)
            if s:
                past_summary = s
                prev_sum_prompt = prompts_config.get("previous_summary_context", "Your memories from past conversations:\n{summary}")
                current_system_context += f"\n\n## PAST CONVERSATIONS SUMMARY\n{prev_sum_prompt.format(summary=s)}"
                if len(message_history) > 1 and message_history[1]["role"] == "system" and isinstance(message_history[1].get("content"), str):
                    message_history[1]["content"] = current_system_context
                else:
                    message_history.insert(1, {"role": "system", "content": current_system_context})
                print("[Main] Past conversations summary loaded (background).")
        except Exception as e:
            print(f"[Main] Past summary background load failed: {e}")
        finally:
            # Warmup is deferred to AFTER startup workers fire (see below)
            # to avoid competing prefills on the SWA box.
            pass

    past_summary_task = asyncio.create_task(load_past_summary_bg())

    # Refresh the past-summary CACHE in the background (for the next startup).
    # Never splices into the live context. Waits a while so the warmup + any
    # immediate conversation get the box first; generate_background refuses
    # while the conversation is hot, so we retry a few times.
    async def refresh_past_summary_cache_bg():
        if past_conversations_count <= 0:
            return
        for attempt in range(4):
            await asyncio.sleep(180 if attempt == 0 else 600)
            try:
                s = await generate_past_conversations_summary(
                    past_conversations_count, force_refresh=True)
                if s:
                    print("[Main] Past-summary cache refreshed for next startup.")
                    return
            except Exception as e:
                print(f"[Main] Past-summary cache refresh failed: {e}")
        print("[Main] Past-summary cache refresh skipped (box busy all session).")

    asyncio.create_task(refresh_past_summary_cache_bg())

    # Workers context will be injected dynamically at each response cycle
    # (see step 0 in the main loop)

    # Initialize modules
    stt = STTEngine()
    # SFX player dynamically loads filler wavs from sound_effects/soundeffects/fillers
    sfx = ThinkingSoundPlayer()
    if always_listen_enabled and followups_enabled():
        stt.set_capture_mode("ambient")
        stt.unmute()
    else:
        stt.mute()

    # Initialize hotword recognizer in the main thread to avoid ONNX/PyAudio threading issues
    try:
        recognizer = HotwordRecognizer(device_index=full_config.get("stt", {}).get("device_index", 2))
    except Exception as e:
        print(f"[Main] Error initializing HotwordRecognizer: {e}")
        recognizer = None

    # Graceful shutdown is handled by catching KeyboardInterrupt in main loop and executing finally block.

    print("=" * 50)
    print("  KikiFast Voice Assistant")
    print("  + Unified Idle Mind | + Tools | + Face Events | + Workers")
    print("=" * 50)
    print(f"  LLM:   {cfg['model']}")
    idle_cfg = full_config.get("idle_mind", {})
    print(f"  Brain: unified cloud idle mind "
          f"({'ENABLED' if idle_cfg.get('enabled', True) else 'DISABLED'})")
    print(f"  KB:    {'Loaded' if kb_summary else 'Empty'}")
    print(f"  Press Ctrl+C to quit")
    print("=" * 50)
    print()

    # Timer logic for returning to hotword mode
    mute_timer = None
    mute_timer_lock = threading.Lock()
    # Hard ceiling on how long the STT interim heartbeat may keep an explicit
    # listening window open (see extend_listen_window). 0 disables the cap.
    max_listen_window_s = (
        float(full_config.get("stt", {}).get("max_listen_window_s", 45) or 0)
        or float("inf")
    )
    listen_window_opened_at = 0.0

    def return_to_background_listening(hotword_only=False):
        """Close the explicit query window and resume passive capture if enabled."""
        hotword_only = hotword_only or not followups_enabled()
        query_listening.clear()
        set_neck_active(False)
        if always_listen_enabled and not hotword_only:
            stt.set_capture_mode("ambient")
            if stt.is_muted:
                stt.unmute()
            lcd_manager.update_status("Idle", "Listening in background")
            print("\n[AlwaysListen] Query window ended; passive listening resumed.")
        else:
            lcd_manager.update_status("Idle", "Waiting for 'heyy'")
            stt.mute()

    def mute_stt():
        # Push-to-talk: a hand on an IR sensor means "keep listening" no matter
        # how long the silence — re-arm the timer instead of muting.
        if ir_controls is not None and ir_controls.hold_active:
            reset_mute_timer()
            return
        if always_listen_enabled and followups_enabled():
            print("\n[Main] ⏱️ 15 seconds empty. Returning to background listening.")
        else:
            print("\n[Main] ⏱️ 15 seconds empty. Muting STT (Back to hotword mode).")
        return_to_background_listening()

    def reset_mute_timer():
        nonlocal mute_timer
        with mute_timer_lock:
            if mute_timer is not None:
                mute_timer.cancel()
            mute_timer = threading.Timer(15.0, mute_stt)
            mute_timer.daemon = True
            mute_timer.start()

    def cancel_mute_timer():
        nonlocal mute_timer, listen_window_opened_at
        with mute_timer_lock:
            if mute_timer is not None:
                mute_timer.cancel()
                mute_timer = None
        listen_window_opened_at = 0.0

    def open_listen_window():
        """Start a fresh explicit-listening budget (wake word, IR, follow-up)."""
        nonlocal listen_window_opened_at
        listen_window_opened_at = time.time()
        reset_mute_timer()

    def extend_listen_window():
        """Keep the window alive while the user is being transcribed — bounded.

        The 15s mute timer is reset by the STT interim heartbeat so a long
        sentence isn't cut off mid-thought. In a crowded room that heartbeat
        never stops (room babble is speech to the VAD), so the timer was reset
        forever and Kiki listened indefinitely. Past max_listen_window_s the
        heartbeat stops counting and the ordinary silence timer takes over.
        """
        nonlocal listen_window_opened_at
        now = time.time()
        if listen_window_opened_at <= 0.0:
            listen_window_opened_at = now
        if now - listen_window_opened_at < max_listen_window_s:
            reset_mute_timer()
            return True
        return False

    # Created after the worker manager; this late-bound checker lets earlier
    # consumers observe whether its cloud session is active.
    idle_mgr = None
    whatsapp_mcp_mgr = None

    def is_idle_mind_active():
        return idle_mgr.is_thinking if idle_mgr is not None else False

    # Hotword background thread
    def hotword_thread_func():
        if not recognizer:
            print("[Hotword] Error: Recognizer not initialized. Thread exiting.")
            return

        try:
            for hotword in recognizer.listen():
                if hotword == "heyy":
                    ir_idle_requested.clear()
                    query_listening.set()
                    # The wake path never waits for the cloud.  Completed ambient
                    # sentences are snapshotted immediately while STT is cut over to
                    # a clean explicit-query utterance.
                    if always_listen_enabled:
                        stt.set_capture_mode("query")
                    # UNMUTE FIRST — everything else on the wake path must come
                    # after, so the user's words right behind "kiki" are never
                    # lost. (preempt_background once blocked here for ~8s.)
                    lcd_manager.update_status("Wake word detected")
                    if stt.is_muted:
                        print("[Hotword] Wake word detected. Unmuting STT immediately! 🔊")
                        stt.unmute()
                    open_listen_window()
                    lcd_manager.update_status("Listening", "user speech...")
                    set_neck_active(True)  # neck tracking on when waking up
                    # Free the local box the INSTANT the wake word fires (not at
                    # endpoint): aborts any background prefill/generation so the
                    # whole utterance duration is available for abort + rewarm.
                    # (Non-blocking: socket shutdown happens on a side thread.)
                    local_llm.preempt_background()
                    if idle_mgr is not None:
                        if idle_mgr.is_thinking:
                            idle_mgr.interrupt(reason="hotword")
                        else:
                            idle_mgr.mark_activity()
                elif hotword in ["stop_music", "stop_it"]:
                    print(f"\n[Hotword] '{hotword}' detected. Running pkill mpv...")
                    subprocess.Popen(["pkill", "-9", "mpv"])
                    if active_tts_streamer:
                        try:
                            if hasattr(active_tts_streamer, "abort"):
                                # Local provider plays via a long-lived aplay
                                # pipe; pkill mpv doesn't touch it.
                                active_tts_streamer.abort()
                            else:
                                active_tts_streamer._sentence_queue.put(None)
                                while not active_tts_streamer._sentence_queue.empty():
                                    active_tts_streamer._sentence_queue.get()
                        except:
                            pass
        except Exception as e:
            print(f"[Hotword] Error: {e}")

    hw_thread = threading.Thread(target=hotword_thread_func, daemon=True)
    hw_thread.start()

    # --- Vision and Autonomy State ---
    vision_handler = VisionHandler(
        full_config, loop, stt_queue, is_idle_mind_active)

    def _vision_history_inject(ctx: str) -> bool:
        # Bake silent vision context into the warm prefix NOW (append-only +
        # rewarm) so the next turn's prefill is a pure cache hit instead of
        # paying these tokens on the speaking path. Refused while a turn is
        # mid-flight or summarization is rebuilding the history — the caller
        # then falls back to pending_vision_context (old behavior).
        if turn_active or summarizing:
            return False
        if not context_enabled("vision"):
            # Returning True (rather than False) deliberately DROPS the context
            # instead of parking it in pending_vision_context, which the turn
            # path would then inject anyway.
            return True
        message_history.append({"role": "system", "content": ctx})
        register_history(message_history)
        return True

    vision_handler.history_inject_fn = _vision_history_inject

    # --- Workers System ---
    worker_manager = get_worker_manager(loop, message_history=message_history)

    # --- Senior Citizen Mode: bridge the care plan onto the workers scheduler ---
    # Additive: only does anything while the 'senior' assistant mode is active.
    senior_care_mgr = None
    try:
        from core.senior.care_plan import get_care_plan_store
        from core.senior.senior_care_manager import get_senior_care_manager
        senior_care_mgr = get_senior_care_manager(
            worker_manager, get_care_plan_store(), full_config.get("senior_mode", {}))
        _queue_care_session = (
            lambda event_id: loop.call_soon_threadsafe(
                stt_queue.put_nowait, ("care_session_start", event_id)))
        worker_manager.set_care_session_callback(_queue_care_session)
        # Same route for "start my exercise now" — the start_care_session tool
        # opens the session, then hands the speaking turn to this lifecycle
        # rather than voicing anything from the tool thread.
        from core.senior.senior_care_manager import set_foreground_hook
        set_foreground_hook(_queue_care_session)
        if get_active_mode() == "senior":
            senior_care_mgr.activate()
    except Exception as _senior_err:
        print(f"[SeniorCare] init skipped: {_senior_err}")

    # Start only after care events have a foreground callback. A due worker in
    # the small gap before registration used to fall back to worker-owned TTS.
    worker_manager.start_scheduler()

    face_history = get_face_history()
    vision_history = get_vision_history()

    # --- Fire startup workers ---
    await worker_manager.fire_event("startup")

    # SINGLE startup warmup — fires AFTER both the past-summary background task
    # and startup workers have injected their context into message_history. On
    # SWA models, competing warmups invalidate each other's cache, so this is
    # the ONE AND ONLY warmup at startup. Previously this fired immediately
    # while the past summary was still loading; the summary then spliced into
    # message_history[1] AFTER the warmup, so the first real user turn paid the
    # full cold prefill anyway (observed: warmup 0.0s, then a second 60s one).
    if cfg.get("warmup_on_start", True):
        async def warmup_when_context_ready():
            # With prefer_cache the summary load is near-instant; the long
            # timeout only matters on a first-ever run with no cache file.
            try:
                await asyncio.wait_for(asyncio.shield(past_summary_task), timeout=180)
            except Exception:
                pass  # summary slow/failed — warm what we have
            # Bake the single pending next-turn note before startup warmup.
            if idle_mgr is not None:
                note = idle_mgr.get_pending_injection()
                if note and context_enabled("idle_notes"):
                    message_history.append({"role": "system", "content": note})
                    print(
                        "[IdleMind] Next-turn note baked into startup context "
                        "(pre-warmup)")
                # Bake the time anchor (+changed workers context) into the prefix
                # too, so the FIRST user turn prefills only the user's words. The
                # idle pre-injection (maybe_inject_time(rewarm=True)) normally keeps
                # the anchor hot, but it hasn't run yet on a fresh boot — without
                # this, turn 1 pays ~55 tokens for the anchor on the speaking path
                # (observed: 74-token prefill for "Can you hear me?"). rewarm=False
                # = append only; the llm_warmup below caches it in one shot. The
                # last_time_injected gate then suppresses main.py's turn-time
                # re-injection for the next time_injection_threshold_minutes.
                idle_mgr.maybe_inject_time(rewarm=False)
            lcd_manager.update_status("Warming up model", "Please wait...")
            print("[Main] 🔥 Warming up local model (caching full prompt prefix)...")
            await loop.run_in_executor(None, llm_warmup, message_history)
            lcd_manager.update_status(
                "Idle",
                ("Listening in background"
                 if always_listen_enabled and followups_enabled()
                 else "Waiting for 'heyy'"),
            )
        asyncio.create_task(warmup_when_context_ready())

    # --- Unified Idle Mind ---
    idle_mgr = UnifiedIdleMindManager(
        loop, message_history, worker_manager, full_config,
        ambient_listener=ambient_listener)
    vision_handler.proactive_prompt_fn = idle_mgr.get_proactive_injection
    idle_mgr.start_monitor()

    # WhatsApp is a long-lived local MCP session on its own daemon thread.
    # This returns immediately: MCP imports, process startup, bridge readiness
    # and every later tool call stay completely outside the wake/speaking path.
    async def _start_whatsapp_off_path():
        nonlocal whatsapp_mcp_mgr
        try:
            def start():
                from core.self_extend.whatsapp_mcp import start_whatsapp_mcp_background
                return start_whatsapp_mcp_background(full_config)
            whatsapp_mcp_mgr = await loop.run_in_executor(None, start)
        except Exception as _e:
            print(f"[WhatsApp] Background MCP startup skipped: {_e}")

    asyncio.create_task(_start_whatsapp_off_path())

    # --- Web UI dashboard (live config + observability) on the LAN, :8090 ---
    # Runs in its own daemon thread; strictly off the speaking path. Reads the
    # in-memory observability recorder and edits config.json via config_loader.
    try:
        start_webui(status_provider=lambda: {
            "status": (
                "speaking" if active_tts_streamer is not None
                else "listening" if query_listening.is_set()
                else "background-listening" if always_listen_enabled and followups_enabled()
                else "idle"
            ),
            "messages_in_context": len(message_history),
            "ambient_sentences_pending": ambient_listener.pending_count,
            "whatsapp_mcp_ready": bool(
                whatsapp_mcp_mgr is not None and whatsapp_mcp_mgr.ready),
        })
    except Exception as _e:
        print(f"[WebUI] not started: {_e}")

    # --- IR sensor controls (left=GPIO22 / right=GPIO17) ---
    # Push-to-talk hold: a hand over EITHER sensor instantly interrupts
    # whatever Kiki is saying/doing (music, thinking filler, speech, the
    # in-flight LLM generation) and opens the mic; the STT endpoint is held
    # open for as long as the hand stays put, and lifting the hand commits
    # the utterance immediately — no silence-window wait — so Kiki replies.
    # Both-hold >=1.2s -> opens the LCD settings menu (inside IRControls).
    def ir_talk_hold_start():
        ir_idle_requested.clear()
        # 1) Instant barge-in: stop music, the thinking filler and any speech,
        # and abort the current turn's LLM stream (partial reply is kept in
        # history; the KV cache stays prefix-consistent).
        try:
            subprocess.Popen(["pkill", "-9", "mpv"])
        except Exception:
            pass
        turn_abort_event.set()
        try:
            sfx.stop()
        except Exception:
            pass
        if active_tts_streamer:
            try:
                if hasattr(active_tts_streamer, "abort"):
                    active_tts_streamer.abort()
                else:
                    active_tts_streamer._sentence_queue.put(None)
                    while not active_tts_streamer._sentence_queue.empty():
                        active_tts_streamer._sentence_queue.get()
            except Exception:
                pass
        # 2) Listen for as long as the hand is present: suspend the endpoint
        # and the 15s idle-mute, then unmute (same ordering as the wake path).
        query_listening.set()
        if always_listen_enabled:
            stt.set_capture_mode("query")
        stt.hold_open()
        cancel_mute_timer()
        if stt.is_muted:
            print("[IR] Hand on sensor - unmuting STT immediately! 🔊")
            stt.unmute()
        lcd_manager.update_status("Listening", "hand held - talk!")
        set_neck_active(True)
        local_llm.preempt_background()
        if idle_mgr is not None:
            if idle_mgr.is_thinking:
                idle_mgr.interrupt(reason="ir_hold")
            else:
                idle_mgr.mark_activity()

    def ir_talk_hold_end():
        # Hand lifted -> commit NOW (the endpointer skips the silence wait).
        # If nothing was said during the hold, this is a no-op endpoint and
        # normal listening (with the 15s timer) resumes.
        stt.hold_release(commit=True)
        open_listen_window()

    def ir_talk_hold_cancel():
        # The hold turned into a both-hands settings entry: drop the hold
        # without committing (on_enter_settings mutes the mic, which resets
        # the endpointer and discards the buffered audio).
        stt.hold_release(commit=False)

    def ir_enter_settings():
        # Make the settings menu modal: silence the mic and the wake word.
        cancel_mute_timer()
        query_listening.clear()
        try:
            stt.mute()
        except Exception:
            pass
        try:
            recognizer.pause()
        except Exception:
            pass

    def ir_exit_settings():
        try:
            recognizer.request_resume()
        except Exception:
            pass
        return_to_background_listening(hotword_only=not followups_enabled())

    def ir_return_to_idle():
        """Double-tap either sensor: terminate the interaction and go idle."""
        print("[IR] Double-tap requested idle mode.")
        ir_idle_requested.set()
        turn_abort_event.set()
        try:
            subprocess.Popen(["pkill", "-9", "mpv"])
        except Exception:
            pass
        try:
            sfx.stop()
        except Exception:
            pass
        if active_tts_streamer:
            try:
                if hasattr(active_tts_streamer, "abort"):
                    active_tts_streamer.abort()
                else:
                    active_tts_streamer._sentence_queue.put(None)
            except Exception:
                pass
        try:
            stt.hold_release(commit=False)
        except Exception:
            pass
        cancel_mute_timer()
        return_to_background_listening()
        try:
            recognizer.request_resume()
        except Exception:
            pass
        try:
            oled_manager.set_state("idle")
        except Exception:
            pass
        local_llm.preempt_background()
        if idle_mgr is not None:
            if idle_mgr.is_thinking:
                idle_mgr.interrupt(reason="ir_double_tap")
            else:
                idle_mgr.mark_activity()
        loop.call_soon_threadsafe(
            stt_queue.put_nowait, ("ir_idle", None))

    def handle_hand_gesture(event):
        """Apply a debounced camera gesture without polling the speaking path."""
        nonlocal spec_turn

        def stop_audio_processes():
            # mpv: music/fillers/cloud TTS; aplay: local TTS; play: timer alarm.
            # Every call is fire-and-forget and runs only after a gesture event.
            for process_name in ("mpv", "aplay", "play"):
                try:
                    subprocess.Popen(["pkill", "-9", process_name])
                except Exception:
                    pass

        gesture = str(event.get("gesture", "")).strip().lower()
        confidence = event.get("confidence")
        confidence_text = (
            f" ({float(confidence):.0%})"
            if isinstance(confidence, (int, float)) and confidence > 0
            else ""
        )
        print(f"[Gesture] {gesture}{confidence_text}")

        if gesture == "mute":
            muted = toggle_output_mute()
            # A speculative stream may have selected its output type before
            # this toggle. Discard it so endpoint adoption uses the new mode.
            if spec_turn is not None:
                spec_turn.abort("output mute toggled")
                spec_turn = None
            if muted:
                mark_activities_stopped()
                stop_audio_processes()
                try:
                    sfx.stop()
                except Exception:
                    pass
                if active_tts_streamer:
                    mute_interrupted_turn.set()
                    try:
                        if hasattr(active_tts_streamer, "abort"):
                            active_tts_streamer.abort()
                        else:
                            active_tts_streamer._sentence_queue.put(None)
                    except Exception:
                        pass
                lcd_manager.update_status("Muted", "LCD text only")
                print("[Gesture] Output muted; future replies use LCD only.")
            else:
                lcd_manager.update_status("Sound on", "Voice restored")
                print("[Gesture] Output unmuted; voice restored.")
            return

        if gesture == "open_palm":
            mark_activities_stopped()
            if spec_turn is not None:
                spec_turn.abort("open-palm stop")
                spec_turn = None
            # Stop the current generation and every local audio source now.
            turn_abort_event.set()
            stop_audio_processes()
            try:
                sfx.stop()
            except Exception:
                pass
            if active_tts_streamer:
                try:
                    if hasattr(active_tts_streamer, "abort"):
                        active_tts_streamer.abort()
                    else:
                        active_tts_streamer._sentence_queue.put(None)
                except Exception:
                    pass
            set_neck_active(False)
            local_llm.preempt_background()
            if idle_mgr is not None and idle_mgr.is_thinking:
                idle_mgr.interrupt(reason="open_palm")
            lcd_manager.update_status("Stopped", "Open palm")
            print("[Gesture] Open palm stopped the active output/activity.")
            return

        if gesture == "thumbs_up":
            # "I'm done — answer now." This is the escape hatch for crowded
            # rooms, where room babble can keep the VAD from ever seeing enough
            # trailing silence to end the utterance on its own.
            #
            # Force-committing goes through the same path as an IR hand release:
            # the endpointer stops waiting out the silence window and emits the
            # final + endpoint immediately, so the turn starts right away. If
            # nothing committable was captured, STT still emits a bare endpoint
            # and main flushes whatever sentences it already collected.
            if ir_controls is not None:
                # A hand still on the IR sensor would otherwise veto the
                # endpoint (both in the endpointer and in main's race guard),
                # so clear the hold as part of the same commit.
                ir_controls.clear_hold()
            if idle_mgr is not None:
                idle_mgr.mark_activity()
            try:
                stt.commit_now()
            except Exception as e:
                print(f"[Gesture] Thumbs-up commit failed: {e}")
            lcd_manager.update_status("Got it", "thinking...")
            print("[Gesture] Thumbs up — ending listening, answering now.")
            return

        if gesture == "peace":
            mark_activities_stopped()
            stop_audio_processes()
            if spec_turn is not None:
                spec_turn.abort("peace gesture idle")
                spec_turn = None
            print("[Gesture] Peace gesture requested hotword/idle mode.")
            ir_return_to_idle()

    ir_controls = IRControls(
        on_talk_hold_start=ir_talk_hold_start,
        on_talk_hold_end=ir_talk_hold_end,
        on_talk_hold_cancel=ir_talk_hold_cancel,
        on_enter_settings=ir_enter_settings,
        on_exit_settings=ir_exit_settings,
        on_double_tap=ir_return_to_idle,
    )
    ir_controls.start()

    # --- Start face event listener (background async task) ---
    face_task = asyncio.create_task(face_event_listener(
        message_history,
        stt,
        idle_mgr,
        loop,
        stt_queue,
        face_history,
        gesture_event_handler=handle_hand_gesture,
    ))

    # --- Senior Care: CLIP activity events (gates 04-05) ---
    # Additive and fail-open, like the care-plan bridge above: the Hailo process
    # publishes care_activity only for candidates that already survived
    # threshold/margin/persistence/dwell, and this consumer re-checks each one
    # against the live frame before it is allowed to reach the history.
    care_events_consumer = None
    try:
        from core.senior.health_events import HealthEventConsumer
        from core.senior.care_plan import get_care_plan_store as _get_care_plan
        from core.llm import hot_inject as _hot_inject
        _care_cfg = full_config.get("senior_mode", {}).get("health_events", {})

        def _care_inject(text):
            _hot_inject(message_history, {"role": "system", "content": text})

        def _care_speak(activity, record):
            """Actually start a turn, rather than leaving a note for the next one.

            An earlier version hot_injected an instruction here. That only
            reaches the model when the user next speaks, so a confirmed
            heat_distress sat silent while the WebUI reported it as "spoken" --
            worse than staying quiet, because the telemetry lied. Queuing
            autonomous_vision is the same route the periodic spoken question
            uses: main.py appends the text and opens a turn immediately.
            """
            prompt = (
                f"You have just seen this yourself: "
                f"{record.get('description', '')[:200]} "
                f"That looks like {activity.replace('_', ' ')}. Check in with "
                f"them about it right now — one short, warm question, no preamble."
            )
            try:
                asyncio.run_coroutine_threadsafe(
                    stt_queue.put(("autonomous_vision", prompt)), loop)
            except Exception as exc:
                print(f"[Care] could not queue the proactive check-in: {exc}")

        def _care_scene(description):
            """Route a care-adjudication description into normal vision context.

            Gate 04 pays for a VLM look at the room on every candidate, and most
            candidates are rejected -- that description used to be discarded.
            Reusing _vision_history_inject means it lands in exactly the format
            and with exactly the guards the periodic vision path already uses
            ([WHAT KIKI SEES]: ..., turn-safe, prefix-rewarmed, gated on the
            'vision' source), rather than becoming a second kind of scene note.
            """
            context = f"{vision_handler.vision_prefix}{description}"
            if not _vision_history_inject(context):
                vision_handler.pending_vision_context = context
            try:
                from core.workers.worker_brain import get_vision_history
                get_vision_history().record_vision(context)
            except Exception:
                pass

        care_events_consumer = HealthEventConsumer(
            config=_care_cfg,
            care_plan=_get_care_plan(),
            speak=_care_speak,
            inject=_care_inject,
            see=_care_scene,
            host=_care_cfg.get("event_host", "127.0.0.1"),
            port=int(_care_cfg.get("event_port", 5556)),
        )
        if care_events_consumer.start():
            print("[Care] activity events wired to the conversation")
    except Exception as _care_err:
        print(f"[Care] health events skipped: {_care_err}")

    # --- Vision Logic ---
    vision_task_ref = None

    # --- Periodic Question Background Loop ---
    async def periodic_question_loop():
        """Independent timer that triggers autonomous questions at configured intervals."""
        nonlocal vision_task_ref
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            if is_idle_mind_active():
                continue
            now = time.time()
            elapsed = now - vision_handler.last_question_time
            if elapsed >= vision_handler.next_question_interval:
                print(f"[Periodic] Question interval reached ({elapsed:.0f}s >= {vision_handler.next_question_interval}s). Triggering vision update...")
                if not vision_task_ref or vision_task_ref.done():
                    # force_qa makes this a SPOKEN autonomous question. Without
                    # it (and with traditional_context_enabled=false) the call
                    # returned immediately and NO request was ever sent — the
                    # periodic-question feature was silently dead.
                    vision_task_ref = asyncio.create_task(
                        vision_handler.run_vision_update(force_trigger=True, force_qa=True)
                    )
                    # Reset the timer HERE (it was only reset on the in-convo
                    # path, so this loop re-fired every 30s once due).
                    vision_handler.last_question_time = now
                    vision_handler.next_question_interval = random.randint(
                        min(vision_handler.question_min_interval, vision_handler.question_max_interval),
                        max(vision_handler.question_min_interval, vision_handler.question_max_interval)
                    )

    periodic_q_task = asyncio.create_task(periodic_question_loop())

    # --- Peeping Background Loop ---
    async def peeping_loop():
        """Periodically unmute STT to passively listen for ambient speech."""
        nonlocal peeping_active
        if always_listen_enabled:
            print("[Peeping] Disabled because always-listen owns passive capture")
            return
        if peeping_interval <= 0:
            print("[Peeping] Disabled (interval_seconds=0)")
            return
        print(f"[Peeping] Enabled: every {peeping_interval}s, listen for {peeping_listen_duration}s")
        while True:
            await asyncio.sleep(peeping_interval)
            # Avoid competing camera work while the unified mind is active.
            if is_idle_mind_active():
                continue
            # Skip if STT is already unmuted (user is actively talking)
            if not stt.is_muted:
                print("[Peeping] STT already unmuted (active conversation), skipping peep")
                continue
            print(f"[Peeping] Starting peep (listening for {peeping_listen_duration}s)...")
            peep_sentences.clear()
            peeping_active = True
            stt.unmute()
            await asyncio.sleep(peeping_listen_duration)
            stt.mute()
            peeping_active = False
            # Collect results
            heard = list(peep_sentences)
            peep_sentences.clear()
            if heard and not context_enabled("peeping"):
                print("[Peeping] Discarded (mode suppresses peeping context)")
            elif heard:
                heard_text = " ".join(heard)
                print(f"[Peeping] Collected: {heard_text}")
                # Append + hot-load so the overheard context is prompt-cached.
                hot_inject(message_history, {
                    "role": "system",
                    "content": f"[Peeping — passively listening to surroundings to see what is going on right now]: {heard_text}"
                })
                print(f"[Peeping] Injected into history")
            else:
                print("[Peeping] Nothing heard")

    peeping_task = asyncio.create_task(peeping_loop())

    stt_thread = threading.Thread(
        target=stt_stream_worker,
        args=(stt, stt_queue, loop),
        daemon=True
    )
    stt_thread.start()

    collected_sentences = []
    thinking_start_time = None

    # Speculative-turn state: at most ONE in flight, created on the STT
    # "speculative" event and resolved (adopted or aborted) at the endpoint.
    _stt_cfg_main = full_config.get("stt", {})
    # Speculation requires a streamer that can HOLD audio until commit — only
    # the local provider implements hold_playback/release_playback.
    spec_turn_enabled = (bool(_stt_cfg_main.get("spec_turn_enabled", True))
                         and full_config.get("tts", {}).get("provider", "local") == "local")
    partial_prefill_enabled = bool(_stt_cfg_main.get("partial_prefill_enabled", True))
    # Pre-warm the deterministic TTS PCM cache with fragments that repeat all
    # the time (the 13 expression tags, tool fillers, bridges): the tts-server
    # is seed-fixed, so cached playback is byte-identical to a fresh synth.
    # Background thread; yields to any live speaking turn.
    def _prewarm_tts_cache_bg():
        try:
            from core.tts import prewarm_tts_cache, SUPPORTED_TAGS
            frags = [f"[{t}]" for t in sorted(SUPPORTED_TAGS)]
            frags += list(TOOL_FILLERS) + list(TOOL_BRIDGES)
            prewarm_tts_cache(frags,
                              should_pause_fn=lambda: active_tts_streamer is not None)
        except Exception as e:
            print(f"[TTS] cache prewarm skipped: {e}")
    threading.Thread(target=_prewarm_tts_cache_bg, daemon=True).start()

    def trigger_background_summary(reason=""):
        """Kick off a SILENT, cache-preserving background summarization.

        This is the ONLY context-shrinking mechanism. There is deliberately no
        emergency mid-prefix trim: dropping messages out of the cached prefix
        forces the box to fully re-prefill (throwing away --cache-prompt and
        spiking TTFW), which defeats the whole single-slot design. Instead we
        leave message_history untouched on the speaking path and let this task
        replace it with a shorter [system_prompt, summary] prefix — and
        pre-warm that prefix — only once the cloud summary is actually ready.

        The box's real context (-c 7000 ≈ 7168) is sized ABOVE
        agent.token_limit on purpose, so a turn or two can sit over the soft
        limit while this runs. No-op if a summary is already in flight.
        """
        nonlocal summarizing
        if summarizing or not agent_config.get("auto_summarize", True):
            return
        summarizing = True
        if reason:
            print(f"[Summarization] Triggering background summary ({reason}).")

        async def summarize_task(current_history):
            nonlocal message_history, summarizing, current_system_context
            try:
                from core.brain.token_counter import count_tokens, SAFE_CONTEXT_LIMIT

                try:
                    oled_manager.set_progress(
                        "summarizing", "compacting our conversation into memory")
                except Exception:
                    pass

                current_tok = count_tokens(message_history)

                # Tool results and the previous memory block are carried in
                # here — see build_summary_input for why they used to be lost.
                convo_text = build_summary_input(current_history)

                # Truncate convo_text to fit in the model's context
                # Leave room for: summary prompt template (~200 tokens) + output (512 tokens)
                max_convo_chars = (SAFE_CONTEXT_LIMIT - 712) * 4  # ~chars, conservative
                if len(convo_text) > max_convo_chars:
                    # Keep the most recent portion
                    convo_text = "[...earlier conversation truncated...]\n" + convo_text[-max_convo_chars:]
                    print(f"[Summarization] Truncated conversation to {len(convo_text)} chars to fit context")

                sum_prompt = prompts_config.get("summarization_prompt", "Summarize this: {conversation}")
                prompt = sum_prompt.format(conversation=convo_text)

                summary = None
                loop = asyncio.get_running_loop()
                # Summarization always runs on the CLOUD now: it triggers
                # right after a turn (inside the hot window), and we never
                # want it competing for the single-slot box, which must stay
                # warm for speaking. The router (purpose="summary") uses
                # gemini-3-flash via GEMINI_KEY_LIST.
                print("[Summarization] Generating summary (cloud)...")
                from core.brain.generate_llm_resp import generate as generate_summary
                _sum_t0 = time.time()
                get_recorder().record("summarization", name="generate",
                                      phase="start",
                                      context_chars=len(convo_text))
                _sum_sid = get_recorder().start_session(
                    "summarization", name="auto", model="cloud",
                    prompt=prompt, context_chars=len(convo_text))
                summary = await loop.run_in_executor(
                    None,
                    lambda: generate_summary(prompt, purpose="summary")
                )
                get_recorder().record(
                    "summarization", name="generate", phase="end",
                    duration_ms=int((time.time() - _sum_t0) * 1000),
                    ok=bool(summary), summary=str(summary or "")[:600])
                get_recorder().log_step(_sum_sid, "result",
                                        content=str(summary or "(empty)"))
                get_recorder().end_session(
                    _sum_sid, status="done" if summary else "failed",
                    summary=str(summary or "")[:800])

                if not summary:
                    print("[Summarization] Aborted or failed (likely preempted by a user request); keeping current history.")
                if summary:
                    print(f"[Summarization] Generated summary:\n{summary[:200]}...")
                    save_summary_to_conversations_folder(summary)
                    save_summary(summary)
                    # Surface what was just compacted in the OLED background feed.
                    try:
                        oled_manager.log_activity("SUMMARY", summary)
                    except Exception:
                        pass

                    # Create new history keeping only the system prompt and injecting summary
                    new_history = [message_history[0]] # Keep enriched prompt
                    prev_sum_prompt = prompts_config.get("previous_summary_context", "Your memories:\n{summary}")

                    current_system_context = prev_sum_prompt.format(summary=summary).strip()
                    new_history.append({"role": "system", "content": current_system_context})
                    # Mutate IN PLACE: idle_mgr / worker_manager hold
                    # references to this list — rebinding the name silently
                    # froze their view of the conversation.
                    message_history[:] = new_history

                    print("[Summarization] Context replaced with new summary")

                    # The rebuild dropped the injected next-turn note. Re-arm it
                    # and bake it into the new short context.
                    if idle_mgr is not None and context_enabled("idle_notes"):
                        idle_mgr.reset_injected_after_summary()
                        reinject = idle_mgr.get_pending_injection()
                        if reinject:
                            message_history.append({"role": "system", "content": reinject})
                            print(
                                "[Summarization] Re-injected the Unified Idle "
                                "Mind next-turn note")

                    # Pre-warm the new (shorter) prefix so the next user turn
                    # doesn't pay the cold prefill for the rebuilt context.
                    register_history(message_history)
            except Exception as e:
                print(f"[Summarization] Error: {e}")
            finally:
                summarizing = False
                # Hand the screen back only if we're still showing the
                # compaction view (a new turn may have taken over).
                try:
                    if getattr(oled_manager, "_state", None) == "summarizing":
                        oled_manager.set_state("idle")
                except Exception:
                    pass

        asyncio.create_task(summarize_task(message_history.copy()))

    try:
        while True:
            try:
                event, text = await asyncio.wait_for(stt_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                # Safety sweep: a speculative turn whose endpoint/invalidate
                # event never arrived (noise-dropped utterance) must not hold
                # the slot or the held audio forever.
                if spec_turn is not None and time.time() - spec_turn.t0 > 10:
                    spec_turn.abort("stale — no endpoint followed")
                    spec_turn = None
                if not turn_active and not summarizing:
                    sync_mode_prompt(rewarm=True)
                continue

            if event == "error":
                print(f"[STT] Error: {text}")
                break

            if event == "ir_idle":
                # Drop any transcript/endpoints produced by the two short
                # push-to-talk presses so they cannot start a fresh turn after
                # the double-tap has already returned Kiki to idle.
                if spec_turn is not None:
                    spec_turn.abort("IR double-tap idle")
                    spec_turn = None
                collected_sentences = []
                while not stt_queue.empty():
                    try:
                        stt_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                return_to_background_listening()
                ir_idle_requested.clear()
                continue

            # The callback queues "ir_idle" behind any STT events already in
            # flight. Ignore those older events until that marker performs the
            # cleanup, otherwise a first-tap endpoint could start a new turn.
            if ir_idle_requested.is_set():
                continue

            if event == "ambient_final":
                ambient_listener.add_sentence(text)
                continue

            if event == "ambient_endpoint":
                continue

            if event == "speculative":
                # The final ASR resolved inside the silence window → start the
                # reply pipeline now, audio held until the commit confirms.
                #
                # NEVER speculate onto a cloud brain: a spec turn is a FULL
                # duplicate generation. On the box that is free (its own single
                # slot, no quota); on Groq/OpenRouter it doubles spend against a
                # per-minute budget, and the discarded spec is exactly what
                # pushes the REAL turn into a 429. So speculation is gated on
                # speaking_is_local() and stays on for the box only.
                if (spec_turn_enabled and speaking_is_local()
                        and spec_turn is None and not turn_active
                        and not peeping_active and not collected_sentences
                        and event == "speculative" and text
                        and not vision_handler.pending_vision_context
                        and (time.time() - vision_handler.last_question_time)
                            <= vision_handler.next_question_interval):
                    # Same append-only, gated injection the endpoint branch
                    # does; doing it BEFORE the snapshot keeps the two paths
                    # byte-identical (mismatch would just abort the spec).
                    idle_mgr.maybe_inject_time(rewarm=False)
                    snapshot = list(message_history)
                    snapshot.append({"role": "user", "content": text})
                    spec_turn = SpeculativeTurn(text, snapshot)
                continue

            if event == "spec_invalid":
                if spec_turn is not None:
                    spec_turn.abort("speech resumed")
                    spec_turn = None
                continue

            if event == "partial":
                # Rolling mid-speech transcript → incremental prefill so the
                # final turn only pays the last few words' tokens. (No-op under
                # the Groq speaking brain — there's no box KV cache to prefill.)
                if (partial_prefill_enabled and not turn_active
                        and not peeping_active and text and speaking_is_local()):
                    local_llm.prefill_partial(
                        " ".join(collected_sentences + [text]) if collected_sentences else text)
                continue

            if event == "interim":
                if not peeping_active:
                    lcd_manager.display_stream("Question:", text)
                    # The user is actively being transcribed (text on the LCD)
                    # even though no 'final' has landed yet. Keep the listen
                    # window alive — otherwise a long utterance that hasn't hit
                    # an endpoint yet gets cut off by the 15s silence timer
                    # mid-sentence and Kiki never responds.
                    if not stt.is_muted and not extend_listen_window():
                        # Budget spent: room noise has been holding the window
                        # open without ever producing a committed utterance.
                        print("[Main] Listen window budget spent — letting the "
                              "silence timer close it (noisy room?).")
                continue

            if event == "final":
                # During peeping, route to peep buffer instead of conversation
                if peeping_active:
                    print(f"[Peeping] Heard: {text}")
                    peep_sentences.append(text)
                    continue

                print(f"[User] {text}")
                lcd_manager.display_stream("Question:", text)
                collected_sentences.append(text)
                # They spoke, so the routine is not running unattended: give the
                # auto-continue budget back in full.
                care_auto_continues = 0

                if idle_mgr is not None:
                    if idle_mgr.is_thinking:
                        idle_mgr.interrupt(reason="user speech")
                    else:
                        idle_mgr.mark_activity()

                if not stt.is_muted:
                    # A committed utterance is real progress, so the window
                    # budget restarts here (unlike a bare interim heartbeat).
                    open_listen_window()

            elif event in ("endpoint", "autonomous_vision", "face_wake",
                           "meet_stranger", "care_session_start",
                           "care_continue"):
                # During peeping, ignore endpoints entirely (no AI response)
                if peeping_active and event == "endpoint":
                    continue
                # Push-to-talk race guard: a commit that landed in the queue
                # just before the hand hit the sensor must NOT start a turn —
                # the user isn't done. Keep the collected sentences; the
                # hand-release commit flushes them (bare endpoints re-fire).
                if (event == "endpoint" and ir_controls is not None
                        and ir_controls.hold_active):
                    print("[Main] Endpoint while hand held — keeping the turn open.")
                    continue
                if event == "endpoint" and not collected_sentences:
                    continue

                # "Shut up" is a terminal conversation command, not a prompt.
                # Handle it before muting-for-TTS / LLM generation so Kiki says
                # nothing, does not reopen the follow-up window, and immediately
                # returns to wake-word-only mode. A speculative reply may already
                # be warming, but its audio is held and can be discarded safely.
                if (event == "endpoint"
                        and is_shut_up_command(" ".join(collected_sentences))):
                    print("[Main] 'shut up' command — returning to hotword mode.")
                    if spec_turn is not None:
                        spec_turn.abort("shut up command")
                        spec_turn = None
                    collected_sentences = []
                    cancel_mute_timer()
                    return_to_background_listening(hotword_only=True)
                    oled_manager.play_oneshot("sad", fallback="idle")
                    continue

                if event == "face_wake":
                    # Known face woke us up → react out loud (force a vision QA).
                    if not vision_task_ref or vision_task_ref.done():
                        vision_task_ref = asyncio.create_task(
                            vision_handler.run_vision_update(force_trigger=True, force_qa=True)
                        )
                    continue

                cancel_mute_timer()
                if ir_idle_requested.is_set():
                    collected_sentences = []
                    return_to_background_listening()
                    continue
                turn_active = True
                guided_care_turn = False
                mute_interrupted_turn.clear()
                # Fresh barge-in kill switch for THIS turn's LLM stream(s): a
                # set() left over from a previous hold must not abort us.
                turn_abort_event = threading.Event()
                if ir_idle_requested.is_set():
                    turn_abort_event.set()
                thinking_start_time = time.time()
                lcd_manager.update_status("Thinking")
                # Narrate on the OLED what Kiki is doing right now.
                try:
                    oled_manager.push_status("about what you just said")
                except Exception:
                    pass

                # User needs us NOW → abort any background work (summary / vision)
                # still occupying the single-slot local box, so the reply is instant.
                local_llm.preempt_background()

                # Push-to-talk: if a hand landed on a sensor in the instant this
                # turn was being set up, the user wants to keep talking — leave
                # the mic open (the barge-in has already aborted our streams).
                if not (ir_controls is not None and ir_controls.hold_active):
                    stt.mute()
                # Suspend wake-word detection for the ENTIRE speaking duration
                # (thinking sounds + TTS, both providers) so the bot's own audio
                # can't trigger its hotword and loop on itself.
                if recognizer:
                    recognizer.pause()
                if is_output_muted():
                    # Mute mode has no filler or TTS audio; response text starts
                    # appearing on the LCD as soon as sentence one is ready.
                    sfx.stop()
                else:
                    sfx.start()

                # 3. Context Injection (Time + workers context).
                # Single source of truth lives in idle_mgr.maybe_inject_time():
                # the idle thread pre-injects + re-warms this anchor BETWEEN turns
                # (idle_time_injection), so on most turns it's already hot and this
                # call is a no-op (the shared last_time_injected gate). When idle
                # didn't get to it, we inject here (rewarm=False) and the speaking
                # request prefills it. Full date + time so Kiki can anchor events
                # to a DAY; the summarizer keys off the "right now it's" marker.
                idle_mgr.maybe_inject_time(rewarm=False)

                if event == "meet_stranger" and not context_enabled("face_events"):
                    # This instruction says "introduce yourself as Kiki", which
                    # breaks a character outright. Drop the whole turn rather
                    # than let a roleplay mode answer a prompt aimed at Kiki.
                    print("[Meet Stranger] Skipped (mode suppresses face events)")
                    collected_sentences = []
                    continue
                if event == "meet_stranger":
                    # Proactively ask a freshly-met stranger their real name,
                    # grounded in what/when we met (clothing + date from the KB).
                    user_utterance = ""
                    collected_sentences = []
                    temp_name = text
                    print(f"\n{'─' * 40}")
                    print(f"[Meet Stranger] Asking real name for {temp_name}")
                    print(f"{'─' * 40}")
                    clothing = first_met = ""
                    try:
                        from core.brain.knowledge_base import get_knowledge_base
                        _p = get_knowledge_base().get_person(temp_name) or {}
                        clothing = _p.get("clothing", "")
                        first_met = _p.get("first_met", "")
                    except Exception:
                        pass
                    when = ""
                    if first_met:
                        try:
                            from datetime import datetime as _dt
                            when = _dt.fromisoformat(first_met).strftime("%A around %H:%M")
                        except Exception:
                            when = ""
                    detail = []
                    if clothing:
                        detail.append(f"they're wearing {clothing}")
                    if when:
                        detail.append(f"you first noticed them {when}")
                    detail_str = ("; ".join(detail)) if detail else "you just met them"
                    instruction = (
                        f"[FRIEND MODE] There's a person in front of you that you recently started "
                        f"remembering under the temporary tag '{temp_name}' ({detail_str}), but you "
                        f"don't know their real name yet. Warmly introduce yourself as Kiki and ask "
                        f"their name in a natural, non-creepy way. When they answer and confirm it, "
                        f"call the set_person_real_name tool with temp_name='{temp_name}' and their "
                        f"real name so you remember them properly."
                    )
                    message_history.append({"role": "system", "content": instruction})
                    print(f"[Context] Injected meet-stranger name request for {temp_name}")
                elif event == "care_continue":
                    # The hold finished, the short reply window passed, and the
                    # person said nothing — because they are exercising, not
                    # chatting. Drive the routine on from the camera instead of
                    # waiting to be spoken to.
                    user_utterance = ""
                    collected_sentences = []
                    guided_care_turn = True
                    message_history.append({
                        "role": "system",
                        "content": ("[CARE ROUTINE CONTINUES]: the person did not "
                                    "reply during the listening window."),
                    })
                    print(f"[Care] Auto-continuing routine "
                          f"(no reply; {care_auto_continues} consecutive)")
                elif event == "care_session_start":
                    # The scheduler supplied timing and an event id, never a
                    # fake user utterance. This now follows the exact normal
                    # mute→recognizer-pause→model→TTS→listen lifecycle.
                    user_utterance = ""
                    collected_sentences = []
                    guided_care_turn = True
                    message_history.append({
                        "role": "system",
                        "content": ("[CARE SESSION STARTED BY SCHEDULE]: event id "
                                    + str(text)),
                    })
                    print(f"[Care] Foreground session queued for event {text}")
                elif event == "autonomous_vision" and not context_enabled("vision"):
                    print("[Autonomous Vision] Skipped (mode suppresses vision)")
                    collected_sentences = []
                    continue
                elif event == "autonomous_vision":
                    user_utterance = ""
                    collected_sentences = []

                    print(f"\n{'─' * 40}")
                    print(f"[Autonomous Vision Query Triggered]")
                    print(f"{'─' * 40}")

                    message_history.append({
                        "role": "system",
                        "content": text
                    })
                    print("[Context] Injected source-grounded autonomous prompt")
                else:
                    # --- Build full user utterance ---
                    user_utterance = " ".join(collected_sentences)
                    collected_sentences = []

                    print(f"\n{'─' * 40}")
                    print(f"[User Query] {user_utterance}")
                    print(f"{'─' * 40}")
                    
                    if vision_handler.pending_vision_context:
                        if context_enabled("vision"):
                            message_history.append({
                                "role": "system",
                                "content": vision_handler.pending_vision_context
                            })
                            print(f"[Context] Injected pending vision context")
                        vision_handler.pending_vision_context = None

                    now = time.time()
                    elapsed = now - vision_handler.last_question_time
                    if elapsed > vision_handler.next_question_interval:
                        opener = (
                            idle_mgr.get_proactive_injection()
                            if idle_mgr and context_enabled("idle_notes") else None
                        )
                        if opener:
                            message_history.append({
                                "role": "system",
                                "content": opener
                            })
                            print("[Context] Injected source-grounded proactive prompt")
                        else:
                            print("[Context] No worthwhile grounded proactive topic; "
                                  "answering the user without an extra question")
                        vision_handler.last_question_time = now
                        vision_handler.next_question_interval = random.randint(
                            min(vision_handler.question_min_interval, vision_handler.question_max_interval), 
                            max(vision_handler.question_min_interval, vision_handler.question_max_interval)
                        )

                    # 4. Add user message to history
                    message_history.append({"role": "user", "content": user_utterance})
                    try:
                        from core.senior.care_plan import get_care_plan_store
                        guided_care_turn = (
                            get_care_plan_store().care_session_state().get("status")
                            == "active")
                    except Exception:
                        guided_care_turn = False

                # --- Vision Injection (every N turns) ---
                turn_counter += 1

                # The single next-turn note is baked into the warm prefix between
                # turns, never added on the latency-sensitive speaking path.

                # --- Speculative turn resolution ---
                # Adopt only if the fully-built context (injections + user
                # message) is byte-identical to the speculative snapshot; any
                # divergence — different transcript, an injection that landed
                # between spec start and commit, a non-endpoint event — aborts
                # the spec and takes the normal path. Nothing was played either
                # way, so the spoken reply is unaffected.
                adopted_spec = None
                if spec_turn is not None:
                    # A speculative turn pre-generated BLIND on the local box. If
                    # this turn is actually a live-image question, it must go to
                    # the Groq vision path — never adopt the blind local guess.
                    _last_user_msg = next(
                        (m.get("content", "") for m in reversed(message_history)
                         if isinstance(m, dict) and m.get("role") == "user"), "")
                    _last_user_msg = (_last_user_msg if isinstance(_last_user_msg, str)
                                      else "")
                    _spec_is_image = (
                        spec_turn.result.get("vision")
                        or (instant_vision.enabled()
                            and instant_vision.is_instant_image_query(_last_user_msg)))
                    if (event == "endpoint"
                            and not guided_care_turn
                            and not _spec_is_image
                            and not spec_turn.result["local_fail"]
                            and not spec_turn.abort_event.is_set()
                            and message_history == spec_turn.messages):
                        adopted_spec = spec_turn
                    else:
                        spec_turn.abort("image query → Groq path" if _spec_is_image
                                        else "context mismatch at commit")
                    spec_turn = None

                # 6. Create TTS streamer (or adopt the speculative one, whose
                # audio was synthesizing ahead and is released to play NOW).
                if adopted_spec is not None:
                    tts_streamer = adopted_spec.tts
                    active_tts_streamer = tts_streamer
                    if hasattr(tts_streamer, "release_playback"):
                        tts_streamer.release_playback()
                    print(f"[Spec] ✔ adopted speculative turn "
                          f"(pipeline head start: {time.time() - adopted_spec.t0:.2f}s)")
                else:
                    tts_streamer = TTSStreamer()
                    active_tts_streamer = tts_streamer
                    tts_streamer.start()

                # Observability: snapshot the EXACT context fed to the local LLM
                # this turn + open a grouped "turn" session (the Web UI shows the
                # whole turn end to end: context, tool calls, the reply). Off the
                # hot path; never raises.
                _turn_sid = None
                try:
                    _rec = get_recorder()
                    def _flatten_content(c):
                        if isinstance(c, list):
                            return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
                        return str(c)
                    _last_user = next(
                        (m.get("content") for m in reversed(message_history)
                         if isinstance(m, dict) and m.get("role") == "user"), "")
                    _rec.set_context(message_history, messages=len(message_history))
                    _rec.record("turn", name="start", phase="start",
                                last_user=_flatten_content(_last_user)[:300])
                    _turn_sid = _rec.start_session(
                        "turn", name="speaking",
                        last_user=_flatten_content(_last_user)[:400])
                    _ctx_str = "\n\n".join(
                        f"[{m.get('role', '?')}]\n{_flatten_content(m.get('content'))}"
                        for m in message_history if isinstance(m, dict))
                    _rec.log_step(_turn_sid, "context",
                                  messages=len(message_history), content=_ctx_str)
                except Exception:
                    pass

                def stop_sfx_on_first_play():
                    tts_streamer.first_play_event.wait()
                    # Snapshot the clock BEFORE stopping the filler: sfx.stop()
                    # used to take 300-500ms (mpv terminate+wait) and inflated
                    # every printed TTFW by that much.
                    _t_first_play = time.time()
                    sfx.stop()
                    # LCD-only output has no spoken first word and must not
                    # contaminate the normal TTFW metric.
                    if getattr(tts_streamer, "is_silent", False):
                        return
                    # A barge-in abort sets first_play_event without any audio
                    # having played — no first word, so no TTFW to report.
                    if getattr(tts_streamer, "_aborted", False):
                        return
                    # Time-to-first-word: end-of-speech → first audio out. This is
                    # the REAL perceived latency (prompt processing + first-sentence
                    # generation + TTS synth of sentence 1), separate from QA Time
                    # which also includes the whole spoken reply + playback.
                    if thinking_start_time is not None:
                        ttfw = _t_first_play - thinking_start_time
                        print(f"[Latency] ⚡ Time to first word: {ttfw:.2f}s")
                        lcd_manager.update_status("Speaking", f"TTFW: {ttfw:.1f}s")
                        try:
                            get_recorder().record("ttfw", name="first word",
                                                  duration_ms=int(ttfw * 1000))
                            if _turn_sid:
                                get_recorder().log_step(_turn_sid, "ttfw",
                                                        ms=int(ttfw * 1000))
                        except Exception:
                            pass
                    # Real speech just started → switch the OLED to the live
                    # voice-waveform animation.
                    oled_manager.set_state("speaking")

                sfx_stopper = threading.Thread(target=stop_sfx_on_first_play, daemon=True)
                sfx_stopper.start()

                # --- Context-size check (cache-preserving, NO trimming) ---
                # The old code hard-dropped the oldest messages here to fit the
                # box's limit. But dropping from the middle of the cached prefix
                # forces a FULL re-prefill (the 1800-token reprocess + 16s TTFW
                # we used to see), throwing away --cache-prompt — the exact thing
                # the single-slot design exists to protect. So we leave the
                # prefix byte-for-byte intact and lean on the silent background
                # summarizer instead. The box's real context (-c 7000 ≈ 7168) is
                # sized above agent.token_limit precisely so a turn or two can
                # sit over the soft limit while that summary lands.
                from core.brain.token_counter import would_exceed_context, SAFE_CONTEXT_LIMIT
                exceeds, pre_tok = would_exceed_context(message_history)
                if exceeds:
                    print(f"[Main] Context at {pre_tok} tokens (soft limit "
                          f"{SAFE_CONTEXT_LIMIT}) — keeping prefix intact; "
                          f"ensuring a background summary is running.")
                    trigger_background_summary(f"pre-turn {pre_tok} tok over soft limit")

                # 6. Stream LLM response in a thread (it's sync/blocking)
                full_response = ""
                raw_first = ""      # verbatim 1st-gen text (KEEPS <neck>/<tool_call> tags)
                raw_followup = ""   # verbatim tool-result follow-up text
                tool_turn = False   # a <tool_call> fired and was answered this turn
                synthetic_tool_followup = False  # deterministic failure reply, not box-generated
                direct_care_turn = False  # complex care agent already authored speak_text
                pending_neck = []
                pending_tool_calls = None
                followup_tool_calls = None  # a call emitted by a follow-up generation

                tool_result_ready = threading.Event()
                tool_result_data = None

                def llm_and_tts():
                    nonlocal full_response, raw_first, pending_neck, pending_tool_calls
                    nonlocal tool_result_data, direct_care_turn
                    first_sentence = True
                    try:
                        # abort_event: the IR push-to-talk barge-in kills this
                        # stream mid-generation (socket close frees the slot);
                        # whatever was generated so far stays in full_response.
                        for evt, data in stream_response(message_history,
                                                         abort_event=turn_abort_event):
                            if evt == "sentence":
                                if first_sentence:
                                    first_sentence = False
                                    print(f"\n[TTS] Queueing sentences...")

                                # Extract neck-gesture tags before TTS
                                gestures = extract_neck_tags(data)
                                if gestures:
                                    pending_neck.extend(gestures)
                                    print(f"[Neck] Found gestures: {gestures}")

                                # Strip neck tags for TTS
                                clean_text = strip_neck_tags(data)
                                if clean_text:
                                    # LCD display is driven by the TTS playback
                                    # clock (see core/tts.py), NOT here — the LLM
                                    # streams far faster than Kiki speaks, so
                                    # updating the screen here raced the audio.
                                    tts_streamer.add_sentence(clean_text)
                                full_response += data + " "

                            elif evt == "tool_calls":
                                pending_tool_calls = data
                                direct_care_turn = is_direct_care_complex_call(
                                    data.get("calls", []), user_utterance)
                                _tool_names = [c['name'] for c in data['calls']]
                                print(f"[Tool] Model requested: {_tool_names}")
                                # Futuristic gear/orbit animation while the
                                # tool runs (music swaps to its own view below).
                                if _tool_names and _tool_names[0] != "play_music":
                                    oled_manager.set_state("tool", _tool_names[0])
                                
                                # Start executing tool calls in the background immediately!
                                def run_tools_bg():
                                    nonlocal tool_result_data
                                    try:
                                        res = execute_tool_calls(data.get("calls", []))
                                        tool_result_data = res
                                    except Exception as ex:
                                        print(f"[Tool] Error executing tool: {ex}")
                                        tool_result_data = f"(error running tools: {ex})"
                                    finally:
                                        tool_result_ready.set()
                                
                                threading.Thread(target=run_tools_bg, daemon=True).start()

                                # If nothing has been said yet (e.g. model called tool immediately),
                                # speak a canned filler to avoid dead air.
                                if first_sentence and TOOL_FILLER_ENABLED and TOOL_FILLERS:
                                    first_sentence = False
                                    filler = random.choice(
                                        pick_tool_fillers(data.get("calls", [])))
                                    print(f"[Tool] Speaking filler: {filler!r}")
                                    tts_streamer.add_sentence(filler)
                                    full_response += filler + " "

                            elif evt == "done":
                                if first_sentence:
                                    sfx.stop()
                                # Keep the model's output VERBATIM (with <neck> and
                                # <tool_call> tags). This is exactly what the box
                                # generated and now holds in its KV cache, so resending
                                # it next turn is a byte-identical cache hit (KV-cache
                                # rule 2). The spoken/clean copy strips tags separately.
                                raw_first = data
                                # Clean any XML tags from full_content for saving
                                clean_data = re.sub(r'<tool_call>.*?</tool_call>', '', data, flags=re.DOTALL).strip()
                                full_response = clean_data
                    except Exception as e:
                        print(f"[Main] LLM/TTS error: {e}")
                        sfx.stop()

                # Follow-up: streams the tool-result answer into the SAME streamer
                # so the filler and the answer play back-to-back (no second turn,
                # no gap). The tool + this generation run WHILE the filler/initial
                # answer is still playing, so the search latency stays hidden.
                def followup_llm_and_tts():
                    nonlocal full_response, raw_followup, pending_neck
                    nonlocal followup_tool_calls
                    followup_tool_calls = None
                    raw_followup = ""
                    try:
                        # verify_prefill=False: this turn legitimately prefills
                        # the fresh tool-result note, so don't trip the
                        # prompt-reprocess diagnostic on it.
                        for evt, data in stream_response(message_history, verify_prefill=False,
                                                         abort_event=turn_abort_event):
                            if evt == "sentence":
                                g = extract_neck_tags(data)
                                if g:
                                    pending_neck.extend(g)
                                clean = strip_neck_tags(data)
                                if clean:
                                    tts_streamer.add_sentence(clean)
                                full_response += " " + data
                            elif evt == "tool_calls":
                                # A tool call while answering a tool RESULT is
                                # the model correcting itself — on 2026-07-29 it
                                # re-sent adjust_volume with an integer after the
                                # schema gate rejected the quoted "60". This
                                # branch did not exist, so the retry was parsed,
                                # logged and dropped: Kiki said "let me try that
                                # again properly" and the volume never moved.
                                # Executed by the bounded loop below, never here.
                                followup_tool_calls = data
                            elif evt == "done":
                                # Verbatim follow-up text for history (matches the box's
                                # KV cache). full_response keeps the spoken filler+answer.
                                raw_followup = data
                    except Exception as e:
                        print(f"[Main] Follow-up LLM/TTS error: {e}")

                def start_tool_execution(calls):
                    """Run `calls` on a daemon thread, mirroring the primary
                    path. Shared by every follow-up round so each one gets the
                    same bounded, abort-aware wait the first tool call gets."""
                    nonlocal tool_result_data
                    tool_result_data = None
                    tool_result_ready.clear()

                    def _run():
                        nonlocal tool_result_data
                        try:
                            tool_result_data = execute_tool_calls(calls)
                        except Exception as ex:
                            print(f"[Tool] Error executing tool: {ex}")
                            tool_result_data = f"(error running tools: {ex})"
                        finally:
                            tool_result_ready.set()

                    threading.Thread(target=_run, daemon=True).start()

                def merge_adopted_spec():
                    """Wait out the (already streaming) speculative run and take
                    its results in place of llm_and_tts. Tool calls recorded by
                    the spec are started HERE (never speculatively — tools have
                    side effects), mirroring llm_and_tts's tool block."""
                    nonlocal full_response, raw_first, pending_neck, pending_tool_calls
                    nonlocal tool_result_data, direct_care_turn
                    # Abort-aware wait: an IR barge-in kills the (still running)
                    # speculative stream too — its abort_event closes the socket
                    # and _run's finally sets done, so this never hangs.
                    while not adopted_spec.done.wait(timeout=0.25):
                        if turn_abort_event.is_set():
                            adopted_spec.abort_event.set()
                    res = adopted_spec.result
                    full_response = res["full"]
                    raw_first = res["raw"]
                    pending_neck.extend(res["neck"])
                    pending_tool_calls = res["tool_calls"]
                    if pending_tool_calls:
                        data = pending_tool_calls
                        direct_care_turn = is_direct_care_complex_call(
                            data.get("calls", []), user_utterance)
                        _tool_names = [c['name'] for c in data['calls']]
                        print(f"[Tool] Model requested (speculative turn): {_tool_names}")
                        if _tool_names and _tool_names[0] != "play_music":
                            oled_manager.set_state("tool", _tool_names[0])

                        def run_tools_bg():
                            nonlocal tool_result_data
                            try:
                                tool_result_data = execute_tool_calls(data.get("calls", []))
                            except Exception as ex:
                                print(f"[Tool] Error executing tool: {ex}")
                                tool_result_data = f"(error running tools: {ex})"
                            finally:
                                tool_result_ready.set()

                        threading.Thread(target=run_tools_bg, daemon=True).start()
                        if not full_response.strip() and TOOL_FILLER_ENABLED and TOOL_FILLERS:
                            filler = random.choice(
                                pick_tool_fillers(data.get("calls", [])))
                            print(f"[Tool] Speaking filler: {filler!r}")
                            tts_streamer.add_sentence(filler)
                            full_response += filler + " "

                # Run LLM streaming + tool + answer, then wait for playback. Wrapped
                # so wake-word detection is ALWAYS re-enabled, even on error.
                try:
                    # First call: a normal reply, OR a silent tool call (which queues
                    # a canned filler). Returns fast for tool calls. An adopted
                    # speculative turn already IS this call — just merge it.
                    if guided_care_turn:
                        from core.senior.care_voice_agent import run_care_voice_turn
                        care_reply = await run_care_voice_turn(
                            user_utterance, stop_event=turn_abort_event)
                        marker = "CARE_ACTION_FAILED:"
                        care_reply = str(care_reply or "").strip()
                        if care_reply.startswith(marker):
                            care_reply = care_reply[len(marker):].strip()
                        gestures = extract_neck_tags(care_reply)
                        if gestures:
                            pending_neck.extend(gestures)
                        spoken = strip_neck_tags(care_reply)
                        if spoken:
                            queue_voice_ready_text(tts_streamer, spoken)
                        full_response = care_reply
                        raw_first = care_reply
                        synthetic_tool_followup = True
                        print("[Care] Conversational care-agent turn queued directly to TTS.")
                    elif adopted_spec is not None:
                        await loop.run_in_executor(None, merge_adopted_spec)
                        if (not adopted_spec.result["ok"] and not pending_tool_calls
                                and not full_response.strip()):
                            # Spec stream died before producing anything usable
                            # (e.g. box went down mid-request) — run the normal
                            # path on the same context; nothing was played/queued.
                            print("[Spec] adopted stream was empty — falling back to normal path")
                            await loop.run_in_executor(None, llm_and_tts)
                    else:
                        await loop.run_in_executor(None, llm_and_tts)

                    # Tool rounds. Round 0 is the call the PRIMARY generation
                    # made; later rounds are calls a follow-up generation made
                    # while answering the previous result — most often the model
                    # fixing its own bad arguments. Bounded by
                    # llm.tool_calling.max_followup_tool_rounds so a model that
                    # keeps re-calling can never hold the turn (and the wake
                    # word) hostage.
                    tool_round = 0
                    while pending_tool_calls:
                        # Wait for the background tool call execution thread to finish.
                        # BOUNDED wait: a tool with a hung network call (no timeout of
                        # its own) must never deadlock the turn — if it did, the finally
                        # below never runs and wake-word detection stays dead, leaving
                        # Kiki unwakeable. On timeout we just speak without the result.
                        _exec_timeout = tool_exec_timeout(
                            pending_tool_calls.get("calls", []))
                        print(f"[Main] Waiting for background tool execution to "
                              f"complete (up to {_exec_timeout:.0f}s)...")
                        def _wait_tool_or_abort():
                            # Slice the wait so an IR barge-in ends the turn at
                            # once instead of blocking on a slow tool.
                            deadline = time.time() + _exec_timeout
                            while time.time() < deadline:
                                if tool_result_ready.wait(timeout=0.25):
                                    return True
                                if turn_abort_event.is_set():
                                    return False
                            return False
                        tool_done = await loop.run_in_executor(None, _wait_tool_or_abort)
                        if not tool_done:
                            # Distinguish the two exits. Reporting a 1.7s
                            # open-palm abort as "timed out after 15.0s" sent a
                            # log reader hunting a slow tool that never existed.
                            if turn_abort_event.is_set():
                                print("[Main] ⏹ Tool execution aborted — continuing without result.")
                            else:
                                print(f"[Main] ⚠️ Tool execution timed out after {_exec_timeout}s — continuing without result.")

                        tool_result = tool_result_data
                        # Observability: log the speaking-path tool call(s) + result
                        # into this turn's session.
                        if _turn_sid:
                            try:
                                for _c in (pending_tool_calls.get("calls", []) or []):
                                    get_recorder().log_step(
                                        _turn_sid, "tool", tool=_c.get("name", "?"),
                                        args=json.dumps(_c.get("arguments", {}), default=str)[:1000],
                                        result=str(tool_result)[:4000])
                            except Exception:
                                pass
                        if not tool_result:
                            break

                        # Put the generation that REQUESTED the tool (with its
                        # verbatim <tool_call> tag) into history BEFORE the result
                        # note. The box's KV cache already holds: prompt + gen +
                        # note + follow-up; omitting the generation diverged the
                        # cache right after the user message and forced a full
                        # re-prefill on both the follow-up request and the next
                        # turn. On a later round that generation is raw_followup.
                        requesting_raw = raw_first if tool_round == 0 else raw_followup
                        if requesting_raw.strip():
                            message_history.append({
                                "role": "assistant",
                                "content": requesting_raw,
                            })
                        tool_turn = True
                        message_history.append({
                            "role": "system",
                            "content": tool_result_note(
                                pending_tool_calls.get("calls", []),
                                tool_result),
                        })
                        # Queue a short bridge connector before the answer so
                        # the follow-up's generation latency doesn't surface as
                        # dead air after the lead-in drains. It plays after the
                        # already-queued initial sentences and before the answer.
                        # TTS-only (not appended to message_history → KV cache
                        # stays byte-identical for the next turn). Only when the
                        # model actually spoke a lead-in (raw_first) — a silent
                        # tool call already gets a TOOL_FILLER above. Round 0
                        # only: a second bridge mid-answer sounds like a stutter.
                        if (not direct_care_turn and tool_round == 0
                                and TOOL_BRIDGE_ENABLED and TOOL_BRIDGES
                                and raw_first.strip()):
                            bridge = random.choice(TOOL_BRIDGES)
                            print(f"[Tool] Bridging to answer: {bridge!r}")
                            tts_streamer.add_sentence(bridge)

                        # A failed care-plan mutation must not get another chance
                        # to be turned into a false success by the model. Speak a
                        # deterministic truthful result; successful writes keep
                        # the normal in-character follow-up generation.
                        direct_reply = (direct_complex_reply(
                            pending_tool_calls.get("calls", []), tool_result)
                            if direct_care_turn else "")
                        failure_reply = deterministic_care_plan_failure_reply(
                            pending_tool_calls.get("calls", []), tool_result)
                        if direct_reply:
                            gestures = extract_neck_tags(direct_reply)
                            if gestures:
                                pending_neck.extend(gestures)
                            spoken = strip_neck_tags(direct_reply)
                            if spoken:
                                queue_voice_ready_text(tts_streamer, spoken)
                            full_response += " " + direct_reply
                            raw_followup = direct_reply
                            followup_tool_calls = None
                            # The local box did not generate this assistant row;
                            # register_history must perform a real rewarm.
                            synthetic_tool_followup = True
                            print("[Care] Direct complex-agent reply queued to TTS; "
                                  "local follow-up LLM bypassed.")
                        elif failure_reply:
                            tts_streamer.add_sentence(failure_reply)
                            full_response += " " + failure_reply
                            raw_followup = failure_reply
                            followup_tool_calls = None
                            synthetic_tool_followup = True
                        else:
                            # Stream the answer onto the SAME streamer (box is free now).
                            # Clears followup_tool_calls, then sets it if this
                            # generation asked for another tool.
                            await loop.run_in_executor(None, followup_llm_and_tts)

                        tool_round += 1
                        if not followup_tool_calls:
                            break
                        if turn_abort_event.is_set():
                            break
                        if tool_round > MAX_FOLLOWUP_TOOL_ROUNDS:
                            print(f"[Tool] Follow-up tool round limit "
                                  f"({MAX_FOLLOWUP_TOOL_ROUNDS}) reached — "
                                  f"not running {[c['name'] for c in followup_tool_calls.get('calls', [])]}.")
                            break
                        pending_tool_calls = followup_tool_calls
                        _names = [c["name"] for c in pending_tool_calls.get("calls", [])]
                        print(f"[Tool] Model requested (follow-up round {tool_round}): {_names}")
                        if _names and _names[0] != "play_music":
                            oled_manager.set_state("tool", _names[0])
                        start_tool_execution(pending_tool_calls.get("calls", []))

                    # Signal no more sentences and wait for full playback.
                    await loop.run_in_executor(None, tts_streamer.finish)
                    # If the mute gesture interrupted an already-speaking
                    # streamer, generation was intentionally allowed to finish.
                    # Show that completed answer once on the LCD using the same
                    # silent streamer future muted turns use.
                    if (mute_interrupted_turn.is_set()
                            and not turn_abort_event.is_set()
                            and full_response.strip()
                            and not getattr(tts_streamer, "is_silent", False)):
                        lcd_streamer = LCDOnlyStreamer()
                        active_tts_streamer = lcd_streamer
                        lcd_streamer.start()
                        lcd_streamer.add_sentence(
                            strip_neck_tags(full_response.strip()))
                        await loop.run_in_executor(None, lcd_streamer.finish)
                    if thinking_start_time is not None:
                        qa_time = time.time() - thinking_start_time
                        lcd_manager.update_status("Done speaking", f"QA Time: {qa_time:.1f}s")
                finally:
                    active_tts_streamer = None
                    pending_tool_calls = None
                    # AI finished streaming audio → ask to re-enable wake-word
                    # detection, but the recognizer waits for the room to actually
                    # go quiet (real end-of-speech) before it listens again.
                    if recognizer:
                        recognizer.request_resume()

                # Clean (tag-stripped) copy of the spoken reply — used for TTS-side
                # bookkeeping, observability, note delivery and logging. The
                # copy STORED in message_history stays verbatim (see step 8 below).
                clean_response = strip_oled_tags(strip_neck_tags(full_response.strip()))

                # Observability: close out the turn with QA time + the spoken reply.
                try:
                    _qa_ms = (int((time.time() - thinking_start_time) * 1000)
                              if thinking_start_time is not None else None)
                    get_recorder().record("turn", name="reply", phase="end",
                                          duration_ms=_qa_ms,
                                          response=clean_response[:1200])
                    if _turn_sid:
                        get_recorder().log_step(_turn_sid, "reply",
                                                content=clean_response)
                        get_recorder().end_session(_turn_sid, status="done",
                                                   response=clean_response[:800])
                except Exception:
                    pass

                # 8. Add the assistant reply to history VERBATIM (with <neck>/<tool_call>
                # tags) so it is byte-identical to what the box generated and cached.
                # For a tool turn the pre-tool generation + result note were already
                # appended above, so here we add the follow-up generation; otherwise the
                # single first generation. clean_response (tags stripped) is still used
                # for TTS/observability/notes below. Fall back to clean text only if the
                # verbatim capture is somehow empty.
                final_assistant_raw = (raw_followup if tool_turn else raw_first).strip()
                message_history.append({
                    "role": "assistant",
                    "content": final_assistant_raw or clean_response
                })

                # A switch_mode tool updates shared runtime state while this turn
                # is speaking. Replace the prompt only after audio completes, then
                # explicitly rewarm it; the server cannot consider a replaced
                # system prompt warm via --prefill-after-response.
                mode_changed = sync_mode_prompt(rewarm=False)

                # Register THIS exact history as the speaking prefix. Must happen
                # after the append so the prefix is byte-identical to what the
                # next turn sends (cache hit). after_speaking=True: when the box
                # runs --prefill-after-response, it already prefilled the reply
                # server-side, so we skip the duplicate client rewarm and just
                # mark the prefix warm (see rewarm_after_speaking config).
                # EXCEPTION: the reply came from Groq (instant-vision, or the Groq
                # speaking provider) — the box never generated or prefilled it, so
                # we must do a REAL rewarm rather than marking a stale prefix warm.
                # This is what keeps the idle box a fast hot standby for the
                # moment every Groq key lands in 429 cooldown.
                groq_turn = last_turn_used_instant_vision() or not speaking_is_local()
                register_history(message_history,
                                 after_speaking=(not mode_changed) and not groq_turn
                                 and not synthetic_tool_followup)
                turn_active = False

                print(f"\n[Assistant] {clean_response}")
                print(f"{'─' * 40}\n")

                # 9. Apply pending neck gestures (in background thread)
                if pending_neck:
                    neck_thread = threading.Thread(
                        target=apply_neck,
                        args=(pending_neck,),
                        daemon=True
                    )
                    neck_thread.start()
                    pending_neck = []

                # 10. Buffer the exchange for the next unified cloud session.
                if idle_mgr is not None and user_utterance:
                    idle_mgr.note_turn(user_utterance, clean_response)

                if idle_mgr is not None and clean_response:
                    idle_mgr.mark_next_turn_note_used(clean_response)

                # If Kiki named a remembered person, flash their face on the OLED
                # ("show our clones / the person we're talking about"). Best-effort.
                if clean_response:
                    try:
                        from core.brain.knowledge_base import get_knowledge_base
                        _people = get_knowledge_base().data.get("people", {})
                        _low = clean_response.lower()
                        for _pname, _pinfo in _people.items():
                            if _pname == "Kiki" or len(_pname) < 3:
                                continue
                            _thumb = _pinfo.get("face_thumb")
                            if _thumb and _pname.lower() in _low:
                                oled_manager.show_face(_pname, image=_thumb,
                                                       subtitle="", hold_seconds=5.0)
                                break
                    except Exception:
                        pass

                # 10b. Bake ALL pending background context into the warm prefix
                # NOW, during the idle gap after the reply — then re-warm ONCE so
                # the NEXT user turn is a pure cache hit instead of prefilling this
                # text on the speaking path while the user waits.
                #
                # Everything that grows the prompt is funnelled through here (or
                # through hot_inject, used by face events / live findings, which
                # already appends + rewarms off the speaking path):
                #   - vision/images : completed mid-turn → fell back to pending
                #   - one model-chosen next-turn note
                # time anchor is pre-baked by idle_mgr's idle thread; summary
                # rebuilds + rewarms in the summarization path. Append-only
                # (KV-cache rule 1); one register_history re-warms the whole prefix.
                baked_ctx = False
                if vision_handler.pending_vision_context:
                    if context_enabled("vision"):
                        message_history.append({
                            "role": "system",
                            "content": vision_handler.pending_vision_context
                        })
                        print("[Context] Baked pending vision into warm prefix (idle)")
                        baked_ctx = True
                    vision_handler.pending_vision_context = None
                if idle_mgr is not None and context_enabled("idle_notes"):
                    note_bake = idle_mgr.get_pending_injection()
                    if note_bake:
                        message_history.append({
                            "role": "system", "content": note_bake})
                        print(
                            "[Context] Baked Unified Idle Mind next-turn note "
                            "into warm prefix")
                        baked_ctx = True
                if baked_ctx:
                    # Single coalesced rewarm covering vision + the note.
                    register_history(message_history)

                # 11. Trigger Vision Update (background async task)
                # Only trigger if another vision task isn't already running.
                # We trigger background vision update if:
                # - vision_every_message is True (every turn capture)
                # - main_vision_enabled is True and it's time for vision injection (every N turns)
                should_trigger_vision = (
                    vision_every_message
                    or (main_vision_enabled
                        and turn_counter % main_vision_every_n == 0))
                if should_trigger_vision:
                    if not vision_task_ref or vision_task_ref.done():
                        vision_task_ref = asyncio.create_task(
                            vision_handler.run_vision_update(force_trigger=True)
                        )

                # 11b. Fire after_response workers
                asyncio.create_task(worker_manager.fire_event("after_response"))

                # 12. Token counting and Auto-summarization
                current_tokens = token_counter.count_tokens(message_history, "gpt-5") #standard for tokenising.
                token_limit = agent_config.get("token_limit", 6000)
                print(f"[Chat Context] Token count: {current_tokens}/{token_limit}")

                if current_tokens > token_limit:
                    # Cache-preserving: this only KICKS OFF a background cloud
                    # summary (no-op if one is already running). It never trims
                    # the live prefix on the speaking path — the summary task
                    # swaps in the shorter history once it's ready. See
                    # trigger_background_summary() above.
                    trigger_background_summary(f"{current_tokens} > {token_limit} tokens")

                # 13. Wait for follow-up query or return to hotword mode immediately
                if ir_idle_requested.is_set():
                    print("[IR] Double-tap idle — keeping the query window closed.")
                    cancel_mute_timer()
                    return_to_background_listening()
                    try:
                        oled_manager.set_state("idle")
                    except Exception:
                        pass
                elif should_skip_followup() or not followups_enabled():
                    hotword_only = not followups_enabled()
                    reason = ("follow-ups disabled" if hotword_only
                              else "music/dance active")
                    print(f"[Main] {reason} — closing the query window immediately.")
                    cancel_mute_timer()
                    return_to_background_listening(hotword_only=hotword_only)
                else:
                    unmute_delay = agent_config.get("post_speech_unmute_delay_seconds", 0.35)
                    print(f"[Main] Speech ended. Waiting {unmute_delay}s before unmuting STT to ensure no loopback...")
                    await asyncio.sleep(unmute_delay)

                    # --- Guided exercise: beat out the hold before listening ---
                    # Still muted here, which is the point: the beeps are loud
                    # and close to the microphone, and a countdown Kiki
                    # transcribed as a reply would derail the routine.
                    care_hold_ran = False
                    if guided_care_turn:
                        try:
                            from core.senior.care_voice_agent import (
                                get_last_care_directive, _exercise_cfg)
                            _ex_cfg = _exercise_cfg()
                            _directive = get_last_care_directive()
                            _hold = int(_directive.get("hold_seconds", 0))
                            if _hold > 0 and _ex_cfg.get("beep_countdown", True):
                                lcd_manager.update_status("Hold", f"{_hold}s")
                                from core.senior.exercise_cadence import play_countdown
                                from core.vision.instant_vision import (
                                    capture_best_frame_b64)
                                # Photograph the position WHILE it is held. The
                                # next turn judges form from these rather than
                                # from a frame taken once the beeps stopped and
                                # the person had already relaxed.
                                _shots = (int(_ex_cfg.get("hold_captures", 3))
                                          if _ex_cfg.get("capture_during_hold", True)
                                          else 0)
                                await loop.run_in_executor(
                                    None,
                                    lambda: play_countdown(
                                        _hold, turn_abort_event,
                                        capture_fn=(capture_best_frame_b64
                                                    if _shots else None),
                                        max_captures=_shots))
                                care_hold_ran = True
                        except Exception as _hold_err:
                            print(f"[Cadence] hold skipped: {_hold_err}")

                    # Clear any stale/late-arriving events in the queue before unmuting
                    while not stt_queue.empty():
                        try:
                            stt_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    print("[Main] Unmuting STT. Listening for follow-up...")
                    lcd_manager.update_status("Listening", "follow-up...")
                    query_listening.set()
                    stt.set_capture_mode("query")
                    stt.unmute()
                    open_listen_window()

                    # --- Guided exercise: a short window, then carry on alone ---
                    # An exercising person answers with their body, not their
                    # voice. Waiting indefinitely for speech is what turned a
                    # neck routine into a question-and-answer session where
                    # every single step needed "okay" before it would advance.
                    if guided_care_turn and not turn_abort_event.is_set():
                        try:
                            from core.senior.care_voice_agent import (
                                get_last_care_directive, _exercise_cfg)
                            _ex_cfg = _exercise_cfg()
                            _directive = get_last_care_directive()
                            _max_auto = int(_ex_cfg.get("max_auto_continues", 8))
                            if (_ex_cfg.get("enabled", True)
                                    and not _directive.get("expect_reply", True)
                                    and care_auto_continues < _max_auto):
                                _listen_s = float(
                                    _ex_cfg.get("reply_listen_seconds", 3.5))

                                async def _continue_if_silent(
                                        listen_s=_listen_s,
                                        after_hold=care_hold_ran):
                                    """Give them a moment to speak; otherwise go on.

                                    Anything the person says lands on stt_queue
                                    (an interim heartbeat the instant the VAD
                                    hears them, well before the transcript), so
                                    a non-empty queue means "they are talking"
                                    and this must not interrupt.
                                    """
                                    await asyncio.sleep(listen_s)
                                    if not stt_queue.empty() or turn_active:
                                        return
                                    if not query_listening.is_set():
                                        return          # window already closed
                                    try:
                                        from core.senior.care_plan import (
                                            get_care_plan_store)
                                        if (get_care_plan_store().care_session_state()
                                                .get("status") != "active"):
                                            return      # session ended meanwhile
                                    except Exception:
                                        return
                                    stt_queue.put_nowait(("care_continue", ""))

                                care_auto_continues += 1
                                asyncio.create_task(_continue_if_silent())
                                print(f"[Care] listening {_listen_s:.1f}s for a reply, "
                                      f"then continuing on my own "
                                      f"({care_auto_continues}/{_max_auto})")
                            elif care_auto_continues >= _max_auto:
                                print(f"[Care] auto-continue budget spent "
                                      f"({_max_auto}); waiting for the person now.")
                        except Exception as _cont_err:
                            print(f"[Care] auto-continue skipped: {_cont_err}")
    except KeyboardInterrupt:
        pass
    finally:
        cancel_mute_timer()
        sfx.stop()

        # Cancel background tasks first to release any resources or locks
        if face_task and not face_task.done():
            face_task.cancel()
        if periodic_q_task and not periodic_q_task.done():
            periodic_q_task.cancel()
        if peeping_task and not peeping_task.done():
            peeping_task.cancel()
        if idle_mgr is not None:
            idle_mgr.stop()
        if whatsapp_mcp_mgr is not None:
            whatsapp_mcp_mgr.stop()
        await ambient_listener.stop()
        try:
            ir_controls.stop()
        except Exception:
            pass

        # Preempt any running background LLM requests
        try:
            local_llm.preempt_background()
        except Exception:
            pass

        # Fire shutdown workers (best-effort)
        try:
            await worker_manager.fire_event("shutdown")
        except Exception as e:
            print(f"[Main] Error firing shutdown workers: {e}")

        # Stop worker scheduler
        worker_manager.stop_scheduler()

        # Save conversation on exit
        convo_text = ""
        try:
            convo_lines = []
            for m in message_history[1:]:
                content = m.get("content")
                if not content:
                    continue
                if m.get("role") == "system":
                    # Keep time-injection markers so session summaries (and the
                    # combined past-summary built from them) carry timestamps.
                    if "right now it's" in str(content).lower():
                        convo_lines.append(f"[TIME] {content}")
                    continue
                convo_lines.append(f"{m['role'].upper()}: {content if isinstance(content, str) else '[image]'}")
            convo_text = "\n".join(convo_lines)
            if convo_text.strip():
                print("[Main] Generating final conversation summary... (Press Ctrl+C again to skip)")
                sum_prompt = prompts_config.get("summarization_prompt", "Summarize this: {conversation}")
                prompt = sum_prompt.format(conversation=convo_text)
                from core.brain.generate_llm_resp import generate
                summary = generate(prompt, purpose="summary")
                if summary:
                    print(f"[Summarization] Generated summary:\n{summary[:200]}...")
                    save_summary_to_conversations_folder(summary)
                    save_summary(summary)
                    print("[Main] Conversation summarized and saved.")
                else:
                    save_summary_to_conversations_folder(convo_text)
                    print("[Main] Summarization failed, raw conversation saved.")
        except KeyboardInterrupt:
            print("\n[Main] Summarization skipped by user.")
            try:
                save_summary_to_conversations_folder(convo_text)
                print("[Main] Raw conversation saved.")
            except Exception:
                pass
        except Exception as e:
            print(f"[Main] Error saving conversation: {e}")

        stt.stop()
        lcd_manager.update_status("Goodbye! 👋", "Powering down")
        # Play the OLED power-down flourish, then blank the panel.
        try:
            oled_manager.stop()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
