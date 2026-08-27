#!/usr/bin/env python3
"""
film_mode.py — Scripted film-shoot controller for Kiki.

Plays back a fully hardcoded sequence of:
  • TTS speech  (via the local omnivoice TTS streamer)
  • Neck turns  (via quick_command → Hailo controller)
  • LCD text    (16×2 character LCD — status words or live speech streaming)
  • OLED state  (128×64 procedural animation surface)

No LLM, no STT, no hotword — just precise, timed playback so you can film
Kiki delivering scripted lines with scripted movements.

Audio pre-caching
-----------------
Run with ``--precache`` to pre-synthesize every speech line via the TTS server
and save the raw PCM files into ``film_cache/``. On subsequent runs the cached
audio is piped straight to aplay — zero network latency, instant playback.

Usage:
    source /srv/kikifast/.venv/bin/activate
    python3 film_mode.py --precache      # download all audio first
    python3 film_mode.py                 # run the full script (uses cache)
    python3 film_mode.py --dry-run       # print the timeline without executing
    python3 film_mode.py --from-cue 5    # start from cue #5 (skip earlier cues)
    python3 film_mode.py --no-cache      # ignore cache, synthesize live

Press Ctrl+C at any time to abort cleanly.
"""

import array
import asyncio
import hashlib
import os
import shutil
import subprocess
import sys
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# ─── Cache directory ───────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "film_cache"

# Audio constants (must match omnivoice tts-server output format).
SAMPLE_RATE = 24000
BYTES_PER_MS = SAMPLE_RATE * 2 // 1000   # 16-bit mono → 48 bytes/ms
GAP_MS = 120                              # inter-sentence silence (matches tts.py)
SILENCE_THR = 300                         # edge-trim amplitude threshold
EDGE_KEEP_MS = 50                         # keep this much at trimmed edges


# ─── Subsystem imports (graceful) ──────────────────────────────────────────

def _import_subsystems():
    """Import Kiki subsystems. Returns a namespace dict."""
    ns = {}
    try:
        from core.tts import TTSStreamer, speak_sentence, sanitize_for_local_tts
        ns["TTSStreamer"] = TTSStreamer
        ns["speak_sentence"] = speak_sentence
        ns["sanitize"] = sanitize_for_local_tts
    except Exception as e:
        print(f"[Film] ⚠ TTS import failed ({e}); speech will be skipped.")
        ns["TTSStreamer"] = None
        ns["speak_sentence"] = None
        ns["sanitize"] = None

    try:
        from core.lcd_display import lcd_manager
        ns["lcd"] = lcd_manager
    except Exception as e:
        print(f"[Film] ⚠ LCD import failed ({e}); LCD will be skipped.")
        ns["lcd"] = None

    try:
        from core.oled_display import oled_manager
        ns["oled"] = oled_manager
    except Exception as e:
        print(f"[Film] ⚠ OLED import failed ({e}); OLED will be skipped.")
        ns["oled"] = None

    try:
        from kiki_control_client import quick_command
        ns["quick_command"] = quick_command
    except Exception as e:
        print(f"[Film] ⚠ Neck control import failed ({e}); neck will be skipped.")
        ns["quick_command"] = None

    try:
        from tools_and_config.config_loader import get_full_config
        ns["host"] = get_full_config().get("controller", {}).get("host", "192.0.2.20")
    except Exception:
        ns["host"] = "192.0.2.20"

    return ns


# ─── PCM helpers (mirrors core/tts.py logic) ──────────────────────────────

def _trim_edge_silence(pcm: bytes, thr: int = SILENCE_THR,
                       keep_ms: int = EDGE_KEEP_MS) -> bytes:
    """Strip model-baked leading/trailing silence from raw 16-bit PCM."""
    a = array.array("h")
    a.frombytes(pcm[: len(pcm) // 2 * 2])
    n = len(a)
    i = 0
    while i < n and abs(a[i]) < thr:
        i += 1
    j = n
    while j > i and abs(a[j - 1]) < thr:
        j -= 1
    keep = SAMPLE_RATE * keep_ms // 1000
    i = max(0, i - keep)
    j = min(n, j + keep)
    return a[i:j].tobytes()


def _cache_key(text: str) -> str:
    """Deterministic hash of the speech text → cache filename."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _cache_path(text: str) -> Path:
    return CACHE_DIR / f"{_cache_key(text)}.pcm"


def _meta_path(text: str) -> Path:
    """Human-readable companion file so you can see what each .pcm contains."""
    return CACHE_DIR / f"{_cache_key(text)}.txt"


# ─── Cue types ─────────────────────────────────────────────────────────────

class CueType(Enum):
    SPEAK       = "speak"        # Kiki speaks a line (TTS)
    NECK        = "neck"         # Neck turn: left / right / center
    LCD         = "lcd"          # Set LCD text (2 lines)
    LCD_STREAM  = "lcd_stream"   # Stream text onto LCD (typing effect)
    OLED        = "oled"         # Set OLED animation state
    WAIT        = "wait"         # Pure delay (seconds)
    MARKER      = "marker"       # Visual marker printed to terminal (no action)


@dataclass
class Cue:
    """One beat in the film script."""
    cue_type: CueType
    # Delay BEFORE this cue executes (seconds). Use this to pace the script.
    pre_delay: float = 0.0

    # --- SPEAK fields ---
    speech_text: str = ""
    # If True, LCD auto-streams the spoken text while TTS plays.
    lcd_mirror_speech: bool = True

    # --- NECK fields ---
    neck_direction: str = ""      # "left", "right", "center"
    neck_steps: Optional[int] = None

    # --- LCD fields ---
    lcd_line1: str = ""
    lcd_line2: str = ""

    # --- OLED fields ---
    oled_state: str = ""          # "idle", "speaking", "thinking", "listening", etc.
    oled_detail: str = ""

    # --- WAIT ---
    wait_seconds: float = 0.0

    # --- MARKER ---
    marker_text: str = ""

    # Human-readable label for the timeline printout.
    label: str = ""


# ─── Helper constructors for readability ───────────────────────────────────

def speak(text, pre_delay=0.0, lcd_mirror=True, label=""):
    return Cue(CueType.SPEAK, pre_delay=pre_delay,
               speech_text=text, lcd_mirror_speech=lcd_mirror,
               label=label or f'SPEAK: "{text[:50]}..."' if len(text) > 50 else label or f'SPEAK: "{text}"')

def neck(direction, steps=None, pre_delay=0.0, label=""):
    return Cue(CueType.NECK, pre_delay=pre_delay,
               neck_direction=direction, neck_steps=steps,
               label=label or f"NECK → {direction}" + (f" ({steps} steps)" if steps else ""))

def lcd(line1, line2="", pre_delay=0.0, label=""):
    return Cue(CueType.LCD, pre_delay=pre_delay,
               lcd_line1=line1, lcd_line2=line2,
               label=label or f'LCD: "{line1}" / "{line2}"')

def lcd_stream(text, pre_delay=0.0, label=""):
    return Cue(CueType.LCD_STREAM, pre_delay=pre_delay,
               speech_text=text,
               label=label or f'LCD↝ "{text[:40]}..."')

def oled(state, detail="", pre_delay=0.0, label=""):
    return Cue(CueType.OLED, pre_delay=pre_delay,
               oled_state=state, oled_detail=detail,
               label=label or f"OLED → {state}" + (f" ({detail})" if detail else ""))

def wait(seconds, pre_delay=0.0, label=""):
    return Cue(CueType.WAIT, pre_delay=pre_delay,
               wait_seconds=seconds,
               label=label or f"WAIT {seconds}s")

def marker(text, pre_delay=0.0):
    return Cue(CueType.MARKER, pre_delay=pre_delay,
               marker_text=text, label=f"🎬 {text}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE SCRIPT — edit this list to change what Kiki does on camera.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCRIPT: list[Cue] = [

    # ── SCENE 1: Cold open — dark room, LCD glowing ──────────────────────
    marker("SCENE 1 — Cold open. Dark room, Kiki's LCD glowing."),

    # # Kiki's LCD shows a moody idle state; OLED is alive with the boot anim.
    oled("pondering..", pre_delay=0.0, label="OLED boot animation"),
    # lcd("Kiki", "Online", pre_delay=0.5),

    # Hold for a beat, then Kiki's head turns toward camera (servo whir).
    # wait(3.0, label="Hold — let the viewer take in the scene"),
    # oled("idle", pre_delay=0.0),
    # # lcd("Kiki is Idle", "Scanning...", pre_delay=0.0),
    neck("right", steps=100, pre_delay=0.5, label="Head turns toward camera"),

    # # Short beat after the turn.
    # wait(2.0, label="Beat after head turn"),

    # ── SCENE 2: First Kiki line ────────────────────────────────────────
    marker("SCENE 2 — Kiki's first line.", pre_delay=0.5),

    oled("speaking", pre_delay=0.0),
    speak(
        text= "[sigh] [question-oh] Alex, seriously? Look at the clock!"
        "[surprise-ah] It's already 10:15 PM. You've been staring at that screen for hours!"
        "Go on, close the laptop. Your brain needs a reboot more than my system does right now.",
        pre_delay=0,
        label='KIKI: "Oh good, you\'re home..."',
    ),

    # After speaking, go back to idle.
    oled("idle", pre_delay=0),
    lcd("Idle", "", pre_delay=0.0),
    wait(0, label="Beat after Kiki's first line"),
    speak(
    "[dissatisfaction-hnn] oh alex! For the sake of your sanity and my peace of mind, can we please just call it a night? ",
    pre_delay=0,
    label='KIKI: "Oh good, you\'re home..."'),
    # ── SCENE 3: VO + Kiki interruptions ─────────────────────────────────
    marker("SCENE 3 — VO introduces Kiki. Kiki interjects.", pre_delay=0),

    # VO: "This is Kiki. He sees me, he hears me..."
    # (VO is recorded separately — Kiki just waits and shows "Listening")
    oled("listening", pre_delay=0.0),
    lcd("Listening...", "", pre_delay=0.0),
    wait(9.0, label="VO: 'This is Kiki. He sees me...' (record separately)"),

    # Kiki: "I prefer 'observant.'"
    oled("speaking", pre_delay=0),
    neck("right", steps=15, pre_delay=0.0, label="Slight head tilt for sass"),
    speak(
        "[dissatisfaction-hnn] I prefer 'observant.'",
        pre_delay=0.3,
        label='KIKI: "I prefer observant."',
    ),
    oled("listening", pre_delay=0.3),
    lcd("Listening...", "", pre_delay=0.0),
    # neck("center", pre_delay=0.5, label="Head back to center"),

    # VO: "I spent a year building him —"
    wait(2.0, label="VO: 'I spent a year building him —' (record separately)"),

    # Kiki: "— he means a year ignoring his degree —"
    oled("speaking", pre_delay=0.2),
    # neck("left", steps=10, pre_delay=0.0, label="Small turn for comedic timing"),
    speak(
        "[laughter] he means a year ignoring his degree.",
        pre_delay=0.2,
        label='KIKI: "— he means a year ignoring his degree —"',
    ),
    oled("listening", pre_delay=0.3),
    lcd("Listening...", "", pre_delay=0.0),

    # VO: "— four robot bodies, a 3 a.m. Linux install,
    #      and one Google Cloud bill I refuse to discuss."
    wait(8.0, label="VO: '— dead pipelines, out-of-memory errors, four tokens a second, and long response times."),

    # Kiki: "Worth it."
    oled("speaking", pre_delay=0.3),
    # neck("center", pre_delay=0.0),
    speak(
        "[confirmation-en] 20 seconds, back then. Tragic. We don't talk about it.",
        pre_delay=0.3,
        label='KIKI: "Worth it."',
    ),
    # Satisfied nod-like small movement.
    # neck("center", pre_delay=0.5, label="Settle back to center"),

    # VO: "This is the whole stupid, beautiful story.
    #      And how you can build your own."
    oled("listening", pre_delay=0.3),
    lcd("Listening...", "", pre_delay=0.0),
    wait(7.0, label="VO: 'This is the whole stupid, beautiful story. And how you can build your own running completely locally"),

    # # ── SCENE 4: Title slam ──────────────────────────────────────────────
    # marker("SCENE 4 — Title slam: KIKI", pre_delay=1.0),

    # # Big moment: LCD shows the title, OLED does a dramatic boot animation.
    # oled("boot", pre_delay=0.0, label="OLED boot anim for title slam"),
    # lcd("    K I K I", "", pre_delay=0.0, label="Title on LCD"),
    # wait(3.0, label="Hold on title"),

    # # ── SCENE 5: Hard cut to workshop mode ───────────────────────────────
    # marker("SCENE 5 — Hard cut: workshop mode, lights up.", pre_delay=1.0),

    # oled("idle", pre_delay=0.0),
    # lcd("Workshop Mode", "Lights up!", pre_delay=0.0),
    # neck("center", pre_delay=0.5),
    # wait(2.0, label="Settle into workshop mode"),

    # # Show a friendly idle state while you do the workshop VO.
    # lcd("Kiki is Idle", "Ready", pre_delay=0.0),

    # # ── END ─────────────────────────────────────────────
    # #─────────────────
    # marker("END — Script complete.", pre_delay=2.0),
    # oled("idle", pre_delay=0.0),
    # more nonsense goes here - frustrated !!!  
    # just one more edit  
    # lcd("Film Mode", "Complete!"),
    # wait(2.0),
    # hello there .. just one more edit 
    # hello there..

]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Audio pre-caching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_tts_url():
    
    """Read the TTS server URL from config (same source as core/tts.py)."""
    try:
        from tools_and_config.config_loader import get_full_config
        cfg = get_full_config().get("tts", {})
        return cfg.get("local_url", "http://127.0.0.1:8082").rstrip("/")
    except Exception:
        return "http://192.0.2.10:8082"


def _get_sanitizer():
    """Get the TTS text sanitizer function."""
    try:
        from core.tts import sanitize_for_local_tts
        return sanitize_for_local_tts
    except ImportError:
        # Minimal fallback if core.tts can't be imported (e.g. off-robot).
        # fixes here .  hello just         
        import re
        def _fallback_sanitize(text):
            text = re.sub(r'<[^>]{1,80}>', '', text)
            text = re.sub(r'[\U0001F000-\U0001FAFF☀-➿️]', '', text)
            text = text.replace("*", "").replace("`", "").replace("#", "")
            return re.sub(r'\s+', ' ', text).strip()
        return _fallback_sanitize


def _collect_speech_texts(cues: list) -> list[str]:
    """Extract all unique speech texts from the script."""
    seen = set()
    texts = []
    for cue in cues:
        if cue.cue_type == CueType.SPEAK and cue.speech_text:
            if cue.speech_text not in seen:
                seen.add(cue.speech_text)
                texts.append(cue.speech_text)
    return texts


def precache_audio(cues: list):
    """Synthesize all speech lines and save as cached PCM files.

    Downloads from the TTS server, applies the same edge-trim as
    LocalTTSStreamer, and writes raw S16_LE 24kHz mono PCM to film_cache/.
    """
    import requests

    texts = _collect_speech_texts(cues)
    if not texts:
        print("  No speech cues found in the script.")
        return

    CACHE_DIR.mkdir(exist_ok=True)
    sanitize = _get_sanitizer()
    tts_url = _get_tts_url()

    print(f"\n{'═' * 60}")
    print(f"  🎤  PRE-CACHING {len(texts)} speech line(s)")
    print(f"  📂  Cache dir: {CACHE_DIR}")
    print(f"  🌐  TTS server: {tts_url}")
    print(f"{'═' * 60}\n")

    session = requests.Session()
    cached = 0
    downloaded = 0
    errors = 0

    for idx, raw_text in enumerate(texts, 1):
        pcm_file = _cache_path(raw_text)
        meta_file = _meta_path(raw_text)
        speakable = sanitize(raw_text)

        # Already cached?
        if pcm_file.exists() and pcm_file.stat().st_size > 0:
            dur_s = pcm_file.stat().st_size / (SAMPLE_RATE * 2)
            print(f"  [{idx}/{len(texts)}] ✅ CACHED ({dur_s:.1f}s): {speakable[:60]}...")
            cached += 1
            continue

        print(f"  [{idx}/{len(texts)}] ⬇  Downloading: {speakable[:60]}...")
        t0 = time.time()

        try:
            r = session.post(
                f"{tts_url}/v1/audio/speech",
                json={"input": speakable, "response_format": "pcm"},
                stream=True,
                timeout=(5, 120),
            )
            if r.status_code != 200:
                print(f"           ❌ HTTP {r.status_code}: {r.text[:200]}")
                errors += 1
                continue

            buf = b""
            with r:
                for chunk in r.iter_content(chunk_size=65536):
                    buf += chunk

            # Apply the same edge-trim as LocalTTSStreamer.
            buf = _trim_edge_silence(buf)

            elapsed = time.time() - t0
            dur_s = len(buf) / (SAMPLE_RATE * 2)

            # Write the PCM file.
            pcm_file.write_bytes(buf)

            # Write a human-readable companion file.
            meta_file.write_text(
                f"text: {raw_text}\n"
                f"sanitized: {speakable}\n"
                f"duration: {dur_s:.2f}s\n"
                f"size: {len(buf)} bytes\n"
                f"synth_time: {elapsed:.2f}s\n"
            )

            print(f"           ✅ Done in {elapsed:.1f}s → {dur_s:.1f}s audio ({len(buf)} bytes)")
            downloaded += 1

        except Exception as e:
            print(f"           ❌ Error: {e}")
            errors += 1

    print(f"\n{'═' * 60}")
    print(f"  📊  Results: {downloaded} downloaded, {cached} already cached, {errors} errors")
    total_bytes = sum(f.stat().st_size for f in CACHE_DIR.glob("*.pcm"))
    total_dur = total_bytes / (SAMPLE_RATE * 2)
    print(f"  💾  Total cache: {total_bytes / 1024:.0f} KB ({total_dur:.1f}s of audio)")
    print(f"{'═' * 60}\n")

    if errors:
        print(f"  ⚠  {errors} line(s) failed. Re-run --precache to retry.\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Execution engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FilmRunner:
    """Plays back a list of Cues in sequence."""

    def __init__(self, cues: list[Cue], dry_run: bool = False,
                 from_cue: int = 0, use_cache: bool = True):
        self.cues = cues
        self.dry_run = dry_run
        self.from_cue = from_cue
        self.use_cache = use_cache
        self.subs = _import_subsystems() if not dry_run else {}
        self._abort = threading.Event()

    def run(self):
        total = len(self.cues)
        print(f"\n{'═' * 60}")
        print(f"  🎬  KIKI FILM MODE — {total} cues")
        if self.dry_run:
            print("  📋  DRY RUN — no hardware will be touched")
        if self.from_cue > 0:
            print(f"  ⏩  Starting from cue #{self.from_cue}")
        if self.use_cache and not self.dry_run:
            n_cached = sum(1 for c in self.cues
                          if c.cue_type == CueType.SPEAK
                          and _cache_path(c.speech_text).exists())
            n_speak = sum(1 for c in self.cues if c.cue_type == CueType.SPEAK)
            if n_cached == n_speak and n_speak > 0:
                print(f"  🔊  All {n_speak} speech lines cached — zero TTS latency!")
            elif n_cached > 0:
                print(f"  🔊  {n_cached}/{n_speak} speech lines cached"
                      f" ({n_speak - n_cached} will synthesize live)")
            else:
                print(f"  ⚠  No cached audio. Run with --precache first for instant playback.")
        print(f"{'═' * 60}\n")

        # Print the full timeline first.
        self._print_timeline()

        if self.dry_run:
            return

        input("\n  Press ENTER to start the script (Ctrl+C to abort)...\n")

        t0 = time.time()
        for idx, cue in enumerate(self.cues):
            if self._abort.is_set():
                break
            if idx < self.from_cue:
                continue
            elapsed = time.time() - t0
            self._execute_cue(idx, cue, elapsed)

        elapsed = time.time() - t0
        print(f"\n{'═' * 60}")
        print(f"  ✅  Script finished in {elapsed:.1f}s")
        print(f"{'═' * 60}\n")

    def _print_timeline(self):
        """Print the full cue sheet with cumulative timing."""
        cum_time = 0.0
        print("  #   TIME     TYPE           LABEL")
        print("  " + "─" * 56)
        for idx, cue in enumerate(self.cues):
            cum_time += cue.pre_delay
            tag = cue.cue_type.value.upper().ljust(12)
            lbl = cue.label or "(no label)"
            skip = "  ⏩" if idx < self.from_cue else ""
            # Show cache status for SPEAK cues.
            cache_tag = ""
            if cue.cue_type == CueType.SPEAK and self.use_cache:
                cf = _cache_path(cue.speech_text)
                if cf.exists():
                    dur = cf.stat().st_size / (SAMPLE_RATE * 2)
                    cache_tag = f"  [cached {dur:.1f}s]"
                else:
                    cache_tag = "  [LIVE]"
            print(f"  {idx:>2}  {cum_time:>6.1f}s  {tag}  {lbl}{cache_tag}{skip}")
            # Account for blocking durations.
            if cue.cue_type == CueType.WAIT:
                cum_time += cue.wait_seconds
        print()

    def _execute_cue(self, idx: int, cue: Cue, elapsed: float):
        lbl = cue.label or cue.cue_type.value
        print(f"  [{elapsed:>6.1f}s]  cue {idx:>2}: {lbl}")

        # Pre-delay.
        if cue.pre_delay > 0:
            self._sleep(cue.pre_delay)

        if cue.cue_type == CueType.MARKER:
            print(f"\n  {'─' * 50}")
            print(f"  🎬  {cue.marker_text}")
            print(f"  {'─' * 50}\n")

        elif cue.cue_type == CueType.WAIT:
            self._sleep(cue.wait_seconds)

        elif cue.cue_type == CueType.LCD:
            self._set_lcd(cue.lcd_line1, cue.lcd_line2)

        elif cue.cue_type == CueType.LCD_STREAM:
            self._stream_lcd(cue.speech_text)

        elif cue.cue_type == CueType.OLED:
            self._set_oled(cue.oled_state, cue.oled_detail)

        elif cue.cue_type == CueType.NECK:
            self._move_neck(cue.neck_direction, cue.neck_steps)

        elif cue.cue_type == CueType.SPEAK:
            self._speak(cue.speech_text, cue.lcd_mirror_speech)

    # ── Hardware drivers ─────────────────────────────────────────────────

    def _sleep(self, seconds: float):
        """Interruptible sleep."""
        end = time.time() + seconds
        while time.time() < end and not self._abort.is_set():
            time.sleep(min(0.1, end - time.time()))

    def _set_lcd(self, line1: str, line2: str = ""):
        lcd = self.subs.get("lcd")
        if lcd:
            try:
                lcd.write(line1, line2)
            except Exception as e:
                print(f"    [LCD error] {e}")

    def _stream_lcd(self, text: str):
        lcd = self.subs.get("lcd")
        if lcd:
            try:
                lcd.display_stream("Speaking:", text)
            except Exception as e:
                print(f"    [LCD stream error] {e}")

    def _set_oled(self, state: str, detail: str = ""):
        oled = self.subs.get("oled")
        if oled:
            try:
                oled.set_state(state, detail)
            except Exception as e:
                print(f"    [OLED error] {e}")

    def _move_neck(self, direction: str, steps: Optional[int] = None):
        qc = self.subs.get("quick_command")
        host = self.subs.get("host", "192.0.2.20")
        if qc:
            def _do():
                try:
                    asyncio.run(qc(host=host, look=direction, look_steps=steps))
                except Exception as e:
                    print(f"    [Neck error] {e}")
            # Fire-and-forget in a daemon thread (same pattern as robot/neck.py).
            threading.Thread(target=_do, daemon=True, name="film_neck").start()

    @staticmethod
    def _clean_display_text(text: str) -> str:
        """Strip bracket tags and normalize whitespace for the LCD display.

        Mirrors what LocalTTSStreamer does: the TTS engine gets the tagged
        version ([laughter], [sigh], etc.) but the LCD shows only the
        actual spoken words.
        """
        import re
        # Remove bracket tags like [laughter], [sigh], [confirmation-en]
        text = re.sub(r'\[[^\[\]]{1,40}\]', '', text)
        # Remove angle/XML tags
        text = re.sub(r'<[^>]{1,80}>', '', text)
        # Remove emojis
        text = re.sub(r'[\U0001F000-\U0001FAFF☀-➿️]', '', text)
        # Remove markdown/misc
        text = text.replace("*", "").replace("`", "").replace("#", "")
        return re.sub(r'\s+', ' ', text).strip()

    def _scroll_lcd_synced(self, lcd, display_text: str, duration: float):
        """Reveal `display_text` word-by-word over `duration` seconds.

        Mirrors LocalTTSStreamer._scroll_sentence exactly: each word's
        reveal time is weighted by its length so longer words linger
        proportionally, matching natural speech pacing.
        """
        words = display_text.split()
        if not words:
            return
        weights = [len(w) + 1 for w in words]
        total = sum(weights) or 1
        start = time.time()
        cum = 0
        revealed = []
        for word, weight in zip(words, weights):
            if self._abort.is_set():
                return
            target = start + duration * (cum / total)
            wait = target - time.time()
            if wait > 0:
                self._sleep(wait)
            if self._abort.is_set():
                return
            revealed.append(word)
            try:
                lcd.display_stream("Speaking:", " ".join(revealed))
            except Exception:
                pass
            cum += weight

    def _play_cached_pcm(self, pcm_file: Path, text: str, lcd_mirror: bool):
        """Play a pre-cached PCM file directly via aplay (zero TTS latency).

        LCD text is revealed word-by-word in sync with the audio playback
        (same karaoke-style pacing as the production TTS display worker).
        """
        lcd = self.subs.get("lcd")
        oled = self.subs.get("oled")

        # Show speaking state on OLED.
        if oled:
            try:
                oled.set_state("speaking")
            except Exception:
                pass

        if not shutil.which("aplay"):
            print("    [Cache] ⚠ aplay not found — cannot play cached audio")
            return

        pcm_data = pcm_file.read_bytes()
        dur_s = len(pcm_data) / (SAMPLE_RATE * 2)
        print(f"    [Cache] ▶ Playing {dur_s:.1f}s from {pcm_file.name}")

        # Prepare clean display text (strip [tags] so only spoken words show).
        display_text = self._clean_display_text(text)

        # Start the LCD word-reveal in a background thread so it runs
        # concurrently with the aplay pipe (which blocks on ALSA drain).
        lcd_thread = None
        if lcd and lcd_mirror and display_text:
            lcd_thread = threading.Thread(
                target=self._scroll_lcd_synced,
                args=(lcd, display_text, dur_s),
                daemon=True,
                name="film_lcd_sync",
            )
            lcd_thread.start()

        try:
            proc = subprocess.Popen(
                ["aplay", "-q", "-f", "S16_LE", "-r", str(SAMPLE_RATE),
                 "-c", "1", "-t", "raw", "-"],
                stdin=subprocess.PIPE,
            )
            proc.stdin.write(pcm_data)
            proc.stdin.close()
            # Wait for playback to complete (aplay blocks until ALSA drains).
            proc.wait(timeout=dur_s + 10)
        except subprocess.TimeoutExpired:
            print("    [Cache] ⚠ aplay timed out — killing")
            try:
                proc.kill()
            except Exception:
                pass
        except Exception as e:
            print(f"    [Cache] Playback error: {e}")

        # Let the LCD thread finish its last word reveal.
        if lcd_thread is not None:
            lcd_thread.join(timeout=1.0)

    def _speak(self, text: str, lcd_mirror: bool = True):
        """Speak a line — from cache if available, otherwise live TTS."""
        # Try cached audio first.
        if self.use_cache:
            pcm_file = _cache_path(text)
            if pcm_file.exists() and pcm_file.stat().st_size > 0:
                self._play_cached_pcm(pcm_file, text, lcd_mirror)
                return

        # Fall back to live TTS synthesis.
        TTSStreamer = self.subs.get("TTSStreamer")
        lcd = self.subs.get("lcd")
        oled = self.subs.get("oled")

        # Show speaking state on displays.
        if oled:
            try:
                oled.set_state("speaking")
            except Exception:
                pass

        if lcd and lcd_mirror:
            try:
                lcd.display_stream("Speaking:", text)
            except Exception:
                pass

        if TTSStreamer:
            try:
                streamer = TTSStreamer()
                streamer.start()
                streamer.add_sentence(text)
                streamer.finish()  # Blocks until audio is done.
            except Exception as e:
                print(f"    [TTS error] {e}")
                # Fallback: just wait an estimated duration.
                est_duration = len(text.split()) * 0.4
                print(f"    [TTS fallback] waiting {est_duration:.1f}s")
                self._sleep(est_duration)
        else:
            # No TTS available: wait an estimated duration so timing works.
            est_duration = len(text.split()) * 0.4
            print(f"    [No TTS] simulating {est_duration:.1f}s of speech")
            self._sleep(est_duration)

    def abort(self):
        self._abort.set()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    dry_run = "--dry-run" in sys.argv
    no_cache = "--no-cache" in sys.argv
    do_precache = "--precache" in sys.argv
    from_cue = 0
    for arg in sys.argv[1:]:
        if arg.startswith("--from-cue"):
            if "=" in arg:
                from_cue = int(arg.split("=")[1])
            else:
                idx = sys.argv.index(arg)
                if idx + 1 < len(sys.argv):
                    from_cue = int(sys.argv[idx + 1])

    # Pre-cache mode: synthesize all audio and exit.
    if do_precache:
        precache_audio(SCRIPT)
        return

    runner = FilmRunner(SCRIPT, dry_run=dry_run, from_cue=from_cue,
                        use_cache=not no_cache)
    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n\n  ⛔  Ctrl+C — aborting film mode.\n")
        runner.abort()
        # Clean up displays.
        if not dry_run:
            subs = runner.subs
            if subs.get("lcd"):
                try:
                    subs["lcd"].write("Film Aborted", "")
                except Exception:
                    pass
            if subs.get("oled"):
                try:
                    subs["oled"].set_state("idle")
                except Exception:
                    pass


if __name__ == "__main__":
    main()