"""
core/oled_display.py — 128x64 OLED "Clawd crab" character for Kiki.

A singleton `oled_manager` that owns an SSD1306 128x64 I2C OLED (addr 0x3c) and
runs a continuous render-loop thread drawing a little pixel-art crab character
whose pose reflects whatever Kiki is doing right now: breathing/blinking while
idle, an attentive lean with incoming sound waves while listening, a thought
bubble while thinking, an animated mouth while speaking, a dancing crab over a
live spectrum strip while music plays, a magnifying glass during deep research,
burrowing into the ground on shutdown, and so on.

The character geometry is the canonical "Clawd" block crab (see the reference in
`test2.py`): a 15x16 logical-pixel sprite (11x7 torso, 2x2 arms, four 1x2 legs,
cut-out eyes) drawn at a native 4x logical-pixel scale with cheap PIL rectangles
— no bitmap upscaling or per-pixel Python loops, so it stays crisp and holds
frame rate on a Pi over I2C.

Design notes
------------
* Mirrors `core/lcd_display.py`'s singleton + graceful-degrade philosophy: if
  `board`/`busio`/`adafruit_ssd1306` (or the panel) are missing it runs in
  emulated mode and never throws, so the rest of Kiki is unaffected off-robot.
* The OLED needs MOTION, so a background thread renders frames continuously at a
  per-state target FPS. State changes are a cheap flag flip under a lock — the
  render thread picks them up.
* The OLED shares the I2C bus (port 1) with the 16x2 char LCD (0x27). BOTH use
  the process-wide `core.i2c_bus.I2C_LOCK` around every hardware transaction so
  their start/stop conditions can't interleave and corrupt each other. Frame
  rates are kept modest (6-16 FPS for most states) so the OLED never starves the
  LCD's writes or hogs the bus.

Integration: `core/lcd_display.py update_status()` forwards here, so every
existing status call drives the OLED automatically. `main.py` adds the
`speaking`/`tool` states and `tools.py play_music` drives `music_on/off`.
"""

import base64
import io
import math
import os
import random
import threading
import time
from collections import deque

from core.i2c_bus import I2C_LOCK

# How long a background-activity item keeps showing in the ambient feed before
# it's considered stale (the resting OLED's tiny top ticker rotates through
# everything newer than this). Deliberately long — 20 minutes.
ACTIVITY_TTL = 1200.0

# --- Optional hardware imports (graceful degrade off-robot) -----------------
try:
    import board
    import busio
    import adafruit_ssd1306
    from PIL import Image, ImageDraw, ImageFont
    _OLED_LIBS = True
except Exception as _e:  # pragma: no cover - hardware/lib specific
    _OLED_LIBS = False
    _IMPORT_ERR = _e

OLED_ADDR = 0x3c
WIDTH = 128
HEIGHT = 64

WHITE = 255
BLACK = 0

# Native logical-pixel geometry for the crab.  Four OLED pixels per sprite
# pixel makes Clawd 33% larger than the old 3x drawing while retaining exact,
# hard 1-bit edges: the sprite is still drawn directly into the panel's
# 128x64 framebuffer, never resized from a lower-resolution bitmap.
CRAB_SCALE = 4
CRAB_OX = (WIDTH - 15 * CRAB_SCALE) // 2
CRAB_OY = 0

# Canonical state names the render loop knows how to draw (each has a matching
# `_draw_<state>` method). The first block mirrors Kiki's runtime; the second
# block are the extra Clawd expressions available for optional wiring.
VALID_STATES = {
    "boot", "idle", "wake", "listening", "thinking", "speaking", "music",
    "tool", "idle_mind", "summarizing",
    "warming", "vision", "goodbye", "workers",
    # Optional / situational expressions.
    "disconnected", "confused", "dizzy", "happy", "sad", "sleeping",
    # Expressions the speaking model can select inline with <oled:name> tags.
    "love", "shy", "giggle", "wink", "excited", "curious", "proud", "sulk",
    "surprised", "sleepy", "idea", "mischief", "scared", "awe",
    # People: a captured face card.
    "face",
}

# The subset a reply may select with an inline `<oled:name>` tag (see
# robot/oled_tags.py). Everything here is a *mood*, never a system state —
# the model can colour how it looks while speaking, but it can't claim to be
# listening, running a tool, or playing music.
#
# This is the single source of truth for tag validity AND for the prompt note,
# so the vocabulary the model is taught can never drift from what can be drawn.
EXPRESSION_STATES = frozenset({
    "love", "shy", "giggle", "wink", "excited", "curious", "proud", "sulk",
    "surprised", "sleepy", "idea", "mischief", "scared", "awe",
    # Pre-existing expressions that had no caller until tags arrived.
    "happy", "sad", "dizzy", "confused", "sleeping",
})

# States an expression tag is allowed to override. Kiki is by definition
# speaking when a tag fires, so this is really a guard against a *late* tag
# landing after the runtime moved on to something that matters more —
# listening, thinking, music, a face card, worker progress, boot/goodbye.
_EXPRESSION_OK_FROM = frozenset({"speaking", "tool"}) | EXPRESSION_STATES

# Per-state target frames-per-second. Kept modest so we don't hog the shared
# I2C bus; near-static/calm states render slowest. These roughly match the
# reference crab's frame-delay timing (idle/sleeping ~6, alert/happy ~10, the
# rest ~8) so the frame-keyed pose animations look right — while music stays a
# touch higher so the real-audio spectrum bars move smoothly.
_STATE_FPS = {
    "boot": 10, "idle": 6, "wake": 10, "listening": 8, "thinking": 8,
    "speaking": 8, "music": 16, "tool": 8, "idle_mind": 8,
    "summarizing": 8, "warming": 8,
    "vision": 8, "goodbye": 10, "workers": 8,
    "disconnected": 6, "confused": 10, "dizzy": 10, "happy": 12, "sad": 10,
    "sleeping": 8,
    # Tag-selectable expressions. Bouncy/startled moods run a touch faster so
    # the motion reads as lively rather than laggy; bashful/sleepy ones slower.
    "love": 12, "shy": 10, "giggle": 14, "wink": 12, "excited": 14,
    "curious": 10, "proud": 10, "sulk": 8, "surprised": 14, "sleepy": 8,
    "idea": 12, "mischief": 10, "scared": 14, "awe": 12,
    "face": 6,
}

# States that play once and then auto-revert to whatever came before them (used
# for momentary flourishes like a happy face-greeting). frame count = duration.
# NOTE: when an expression is set by a <oled:> tag it is held for the rest of
# the turn instead (see `set_expression` / `_expr_hold`), so these durations
# apply only to `play_oneshot` callers like the face greeting.
_ONE_SHOT_FRAMES = {"happy": 48, "sad": 56}

# Maps raw tool names to short human-readable phrases for the OLED status line
# ("now searching the web" reads better than "calling search_web").
_TOOL_PHRASES = {
    "search_web": "searching the web",
    "execute_shell_command": "running a command",
    "execute_python_code": "running some code",
    "get_current_time": "checking the time",
    "recall_memory": "searching memory",
    "update_knowledge": "updating memory",
    "play_music": "queuing music",
    "set_timer": "setting a timer",
    "remember_me": "memorizing a face",
    "track_person": "tracking someone",
    "follow_me": "following along",
    "schedule_worker": "scheduling a worker",
    "cancel_worker": "cancelling a worker",
    "list_workers": "checking its workers",
}


def friendly_tool(name: str) -> str:
    """Human-readable phrase for a tool name, for the OLED status line."""
    if not name:
        return "working"
    if name in _TOOL_PHRASES:
        return _TOOL_PHRASES[name]
    if name.startswith("self_extend"):
        return "extending itself"
    return name.replace("_", " ")


def _font():
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        return ImageFont.load_default()


def _font_big():
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        return ImageFont.load_default()


def _font_tiny():
    """A very small font for the unobtrusive top activity/status ticker."""
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
    except Exception:
        return ImageFont.load_default()


class OLEDDisplayManager:
    _instance = None
    _singleton_lock = threading.Lock()

    def __new__(cls, *a, **k):
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.disp = None
        self.enabled = False
        self.font = None
        self.font_big = None
        self.font_tiny = None

        self._lock = threading.Lock()
        self._state = "boot"
        self._detail = ""
        self._state_started = time.time()
        self._frame = 0
        self._stopped = False
        # Where a one-shot flourish (e.g. "happy") returns to when it finishes.
        self._oneshot_fallback = "idle"
        # True while the current expression came from a <oled:> tag. Such an
        # expression is a mood held for the rest of the spoken turn, so the
        # _ONE_SHOT_FRAMES auto-revert must not cut it short. Any real
        # set_state() clears it, so system events always win.
        self._expr_hold = False

        # Music marquee + animation memory.
        self._music_title = ""
        # Persistent spectrum-analyzer bar levels (smoothed frame to frame).
        self._bars = [0.0] * 24
        # Real audio band levels (0..1) pushed by the MusicVisualizer thread,
        # plus the time they arrived — the music view uses them when fresh and
        # falls back to the procedural animation when they go stale.
        self._audio_levels = None
        self._audio_ts = 0.0
        self._visualizer = None
        # A longer descriptive status line ("now searching the web", a truncated
        # summary, ...) shown as an unobtrusive top ticker beneath the crab's
        # working poses. Separate from `_detail` so it can change without
        # resetting the running animation.
        self._status_line = ""
        self._status_ts = 0.0
        # Rolling feed of what Kiki has been doing in the background (summaries,
        # idle thoughts, reflections, worker results, ...). When resting, the
        # crab's idle view shows a tiny one-line ticker rotating through these so
        # the display still reads as "things are happening". Each item is
        # {ts, kind, text}; newest surface first and linger ACTIVITY_TTL.
        self._activity = deque(maxlen=16)

        # People view: a pre-dithered 1-bit face bitmap + name/subtitle for the
        # "face" card.
        self._face_img = None      # PIL "1" image (~56x56) or None
        self._face_name = ""
        self._face_subtitle = ""
        # How long the face card stays up before auto-reverting to idle.
        self._people_until = 0.0

        if _OLED_LIBS:
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                with I2C_LOCK:
                    self.disp = adafruit_ssd1306.SSD1306_I2C(
                        WIDTH, HEIGHT, i2c, addr=OLED_ADDR)
                    self.disp.fill(0)
                    self.disp.show()
                self.font = _font()
                self.font_big = _font_big()
                self.font_tiny = _font_tiny()
                self.enabled = True
                print("[OLED] Initialized SSD1306 128x64 successfully.")
            except Exception as e:
                print(f"[OLED] Warning: init failed ({e}). Emulated mode.")
        else:
            print(f"[OLED] Libraries unavailable ({_IMPORT_ERR}). Emulated mode.")

        # Image buffers reused every frame.
        if _OLED_LIBS:
            self._image = Image.new("1", (WIDTH, HEIGHT))
            self._draw = ImageDraw.Draw(self._image)
        else:
            self._image = None
            self._draw = None

        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ API
    def set_state(self, state: str, detail: str = ""):
        """Switch the live animation. `state` should be a VALID_STATES name.

        The frame counter / start time reset ONLY when the *state* itself
        changes — updating just the `detail` text (granular progress narration)
        keeps the underlying animation running smoothly instead of stuttering
        back to frame 0 on every progress tick.
        """
        state = (state or "idle").lower()
        if state not in VALID_STATES:
            state = "idle"
        with self._lock:
            # `speaking` is the BASELINE face for a turn, and an expression is a
            # refinement of it — not something it should overwrite. main.py sets
            # "speaking" from the first-play callback at almost exactly the
            # moment the player fires a tag on sentence 1, so without this the
            # two race and the expression loses. Every other state still wins
            # outright, and the turn's closing set_state("idle") releases the
            # hold.
            if state == "speaking" and self._expr_hold and self._state in EXPRESSION_STATES:
                self._detail = detail or ""
                return
            # An explicit system state always outranks a held tag expression.
            self._expr_hold = False
            if state != self._state:
                # Remember where a one-shot flourish should return to.
                if state in _ONE_SHOT_FRAMES and self._state not in _ONE_SHOT_FRAMES:
                    self._oneshot_fallback = self._state
                self._state = state
                self._state_started = time.time()
                self._frame = 0
                # A new animation drops any leftover narration from the previous
                # one (e.g. a finished worker line shouldn't bleed into the next
                # speaking turn) — progress updates that stay in the same state
                # keep their line; set_progress re-pushes after this.
                self._status_line = ""
            self._detail = detail or ""

    def play_oneshot(self, state: str, fallback: str = None):
        """Play a momentary flourish (e.g. `happy`) that auto-reverts. If
        `fallback` is given it's used as the return state, else the state that
        was showing when this was called."""
        if fallback is not None:
            with self._lock:
                self._oneshot_fallback = fallback
        self.set_state(state)

    def set_expression(self, name: str) -> bool:
        """Set a mood requested by an inline `<oled:name>` tag in Kiki's reply.

        Called from the TTS player on the first audible chunk of the sentence
        that carried the tag, so the face changes exactly when Kiki speaks those
        words. Returns True if it was applied.

        Two rules make this safe to call from the audio path:

        * Only expressions may be selected (`EXPRESSION_STATES`) — a reply can
          colour how Kiki looks, but can't claim to be listening or running a
          tool.
        * It only overrides `_EXPRESSION_OK_FROM`. If the runtime has already
          moved the display on to something that matters more (a face card,
          music, worker progress, shutting down), a late-arriving tag is
          dropped rather than fighting it.

        The mood is then *held* — it stays up for the rest of the turn instead
        of auto-reverting after `_ONE_SHOT_FRAMES` — until the next tag or the
        next real `set_state()`.
        """
        name = (name or "").lower()
        if name not in EXPRESSION_STATES:
            return False
        with self._lock:
            if self._state not in _EXPRESSION_OK_FROM:
                return False
            if name != self._state:
                # Return to `speaking` rather than to the previous expression,
                # so a chain of tagged sentences can't build a revert stack.
                if not self._expr_hold:
                    self._oneshot_fallback = "speaking"
                self._state = name
                self._state_started = time.time()
                self._frame = 0
                self._status_line = ""
            self._expr_hold = True
            self._detail = ""
        return True

    def _prepare_face_bitmap(self, image, size=56):
        """Turn a face crop (file path, base64 JPEG string, raw bytes, or PIL
        Image) into a square 1-bit dithered bitmap for the OLED. Returns a PIL
        "1" Image or None. Safe to call off the render thread."""
        if not _OLED_LIBS or image is None:
            return None
        try:
            img = None
            if isinstance(image, Image.Image):
                img = image
            elif isinstance(image, (bytes, bytearray)):
                img = Image.open(io.BytesIO(bytes(image)))
            elif isinstance(image, str):
                if os.path.exists(image):
                    img = Image.open(image)
                else:
                    # Assume a base64 JPEG/PNG string.
                    img = Image.open(io.BytesIO(base64.b64decode(image)))
            if img is None:
                return None
            img = img.convert("L").resize((size, size))
            return img.convert("1")  # Floyd-Steinberg dither
        except Exception as e:
            print(f"[OLED] face bitmap prep failed: {e}")
            return None

    def show_face(self, name: str, image=None, subtitle: str = "", hold_seconds: float = 8.0):
        """Show a person's face + name on the OLED (used when meeting a new
        stranger and when Kiki mentions/sees a remembered person). `image` may
        be a file path, base64 string, raw bytes, or PIL Image. Auto-reverts to
        idle after `hold_seconds`."""
        bitmap = self._prepare_face_bitmap(image)
        with self._lock:
            self._face_img = bitmap
            self._face_name = (name or "").strip()
            self._face_subtitle = (subtitle or "").strip()
            self._people_until = time.time() + max(1.0, hold_seconds)
        self.set_state("face")

    def set_progress(self, state: str, status: str = "", detail: str = ""):
        """Drive a long-running animation (idle_mind / workers / tool /
        summarizing) and its status line in one
        call. `status` is the descriptive narration ("now searching the web",
        a truncated summary, ...)."""
        # State first (clears any stale line on a real change), then the line.
        self.set_state(state, detail)
        self.push_status(status)

    def push_status(self, status: str):
        """Update only the descriptive status line (the top ticker) without
        touching the current animation/state."""
        with self._lock:
            self._status_line = (status or "").strip()
            self._status_ts = time.time()

    def log_activity(self, kind: str, text: str):
        """Record something Kiki just did in the background (a summary, an idle
        thought, a worker result, ...) into the rolling feed the
        resting crab's ticker rotates through. `kind` is a short ALL-CAPS tag
        shown as a heading ("IDLE MIND", "SUMMARY", ...)."""
        text = " ".join((text or "").split())
        if not text:
            return
        if len(text) > 180:
            text = text[:177] + "..."
        kind = (kind or "NOTE").strip().upper()[:16]
        with self._lock:
            # Drop an immediate exact-duplicate of the newest entry (cycles can
            # re-narrate the same outcome) so the feed doesn't show it twice.
            if self._activity and self._activity[-1].get("text") == text:
                self._activity[-1]["ts"] = time.time()
            else:
                self._activity.append(
                    {"ts": time.time(), "kind": kind, "text": text})

    def _recent_activity(self):
        """Recent feed items (newer than ACTIVITY_TTL), newest first."""
        cut = time.time() - ACTIVITY_TTL
        with self._lock:
            return [dict(it) for it in reversed(self._activity)
                    if it["ts"] >= cut]

    @staticmethod
    def _rel_age(ts):
        s = max(0, int(time.time() - ts))
        if s < 45:
            return "just now"
        if s < 3600:
            return f"{s // 60}m ago"
        return f"{s // 3600}h ago"

    def set_audio_levels(self, levels):
        """Called by the MusicVisualizer thread with real spectrum band levels
        (an iterable of floats in 0..1). Stamped so the music view knows they're
        fresh; stale levels fall back to the procedural animation."""
        if levels is None:
            return
        with self._lock:
            self._audio_levels = list(levels)
            self._audio_ts = time.time()

    def update_status(self, action: str, details: str = None):
        """Map an LCD-style status string to an OLED state. Called by
        `lcd_display.update_status` so every existing status drives the OLED."""
        a = (action or "").lower()
        detail = details or ""
        if "wake" in a:
            self.set_state("wake")
        elif "listening" in a:
            self.set_state("listening", detail)
        elif "idle mind" in a:
            self.set_state("idle_mind", detail)
        elif "thinking" in a:
            self.set_state("thinking", detail)
        elif "speaking" in a and "done" not in a:
            self.set_state("speaking", detail)
        elif "done speaking" in a:
            self.set_state("idle")
        elif "working" in a:
            # Previously fell through to the catch-all and showed `idle` — i.e.
            # the display went to sleep exactly while Kiki was busy.
            self.set_state("thinking", detail)
        elif "summar" in a:
            self.set_state("summarizing", detail)
        elif "warm" in a:
            self.set_state("warming", detail)
        elif "vision" in a:
            self.set_state("vision", detail)
        elif "disconnect" in a or "offline" in a:
            self.set_state("disconnected", detail)
        elif "confus" in a or "error" in a:
            self.set_state("confused", detail)
        elif "dizzy" in a:
            self.set_state("dizzy")
        elif "sleep" in a:
            self.set_state("sleeping")
        elif "happy" in a or "greet" in a:
            self.set_state("happy")
        elif "sad" in a or "hurt" in a:
            self.set_state("sad")
        elif "startup" in a:
            self.set_state("boot")
        elif "goodbye" in a or "shutdown" in a or "powering" in a:
            self.set_state("goodbye")
        elif "idle" in a:
            self.set_state("idle")
        else:
            self.set_state("idle", detail)

    def music_on(self, title: str = ""):
        self._music_title = (title or "").strip()
        self.set_state("music", self._music_title)
        # Start tapping the real audio so the spectrum is in sync with the song.
        try:
            if self._visualizer is None or not self._visualizer.is_alive():
                self._visualizer = MusicVisualizer(self)
                self._visualizer.start()
            else:
                self._visualizer.keep_running()
        except Exception as e:
            print(f"[OLED] Visualizer start failed ({e}); using procedural spectrum.")

    def music_off(self):
        """Leave music view only if we're still showing it (don't clobber a
        conversation that started while the song was ending)."""
        if self._visualizer is not None:
            try:
                self._visualizer.stop()
            except Exception:
                pass
            self._visualizer = None
        with self._lock:
            self._audio_levels = None
            if self._state == "music":
                self._state = "idle"
                self._detail = ""
                self._state_started = time.time()
                self._frame = 0

    def clear(self):
        if self.enabled and self.disp:
            try:
                with I2C_LOCK:
                    self.disp.fill(0)
                    self.disp.show()
            except Exception:
                pass

    def stop(self):
        self._stopped = True
        if self._visualizer is not None:
            try:
                self._visualizer.stop()
            except Exception:
                pass
            self._visualizer = None
        # Final power-down flourish then blank.
        self.set_state("goodbye")
        time.sleep(0.6)
        self.clear()

    # ---------------------------------------------------------- render loop
    def _render_loop(self):
        while not self._stopped:
            with self._lock:
                state = self._state
                detail = self._detail
                t = time.time() - self._state_started
                frame = self._frame
                self._frame += 1
            fps = _STATE_FPS.get(state, 12)
            try:
                self._render_frame(state, detail, t, frame)
            except Exception as e:
                # Never let a draw bug kill the loop.
                print(f"[OLED] render error in {state}: {e}")
            # One-shot flourishes auto-revert to the previous state once done —
            # unless the expression is being held for a spoken turn by a
            # <oled:> tag, which ends when the turn does.
            limit = _ONE_SHOT_FRAMES.get(state)
            if limit is not None and frame >= limit and not self._expr_hold:
                self.set_state(self._oneshot_fallback)
            # The face card holds for a few seconds then reverts.
            if state == "face" and time.time() > self._people_until:
                self.set_state("idle")
            time.sleep(max(0.0, 1.0 / fps))

    def _render_frame(self, state, detail, t, frame):
        if not (_OLED_LIBS and self._image is not None):
            return  # emulated: nothing to draw
        d = self._draw
        d.rectangle((0, 0, WIDTH, HEIGHT), outline=0, fill=0)

        fn = getattr(self, f"_draw_{state}", None)
        if fn is None:
            fn = self._draw_idle
        fn(d, t, frame, detail)

        if self.enabled and self.disp:
            # Hold the shared I2C lock so the LCD's writes on the same bus
            # can't interleave with this full-buffer refresh (Errno 121).
            with I2C_LOCK:
                self.disp.image(self._image)
                self.disp.show()

    # ------------------------------------------------------- text helpers
    def _label(self, d, text, y=HEIGHT - 11, fill=255):
        if not text:
            return
        text = text[:21]
        try:
            w = d.textlength(text, font=self.font)
        except Exception:
            w = len(text) * 6
        d.text(((WIDTH - w) // 2, y), text, font=self.font, fill=fill)

    def _marquee(self, d, text, y, t, fill=255):
        """Horizontally scroll text that overflows 128px; center if it fits."""
        if not text:
            return
        try:
            w = d.textlength(text, font=self.font)
        except Exception:
            w = len(text) * 6
        if w <= WIDTH - 4:
            d.text(((WIDTH - w) // 2, y), text, font=self.font, fill=fill)
            return
        gap = 24
        span = w + gap
        off = int((t * 26) % span)
        x = 2 - off
        d.text((x, y), text, font=self.font, fill=fill)
        d.text((x + span, y), text, font=self.font, fill=fill)

    def _top_ticker(self, d, t, fallback=""):
        """Unobtrusive one-line ticker along the very top (tiny font): rotates
        through the recent background-activity feed while resting, else shows
        `fallback`. Sits above the crab (which lives in the lower half), so it
        never overlaps the character."""
        if self.font_tiny is None:
            return
        items = self._recent_activity()
        if items:
            idx = int(t / 5.0) % len(items)
            it = items[idx]
            line = f"{it['kind']}: {it['text']}"
        else:
            line = fallback
        if not line:
            return
        d.text((1, 0), line[:26], font=self.font_tiny, fill=255)

    def _status_top(self, d, t, fallback=""):
        """Top ticker showing the live `_status_line` (granular 'now searching
        the web' / truncated-summary narration) when fresh, else the short
        `fallback` label. Used by the crab's 'working' poses whose props occupy
        the bottom of the screen."""
        if self.font_tiny is None:
            return
        with self._lock:
            status = self._status_line
            fresh = status and (time.time() - self._status_ts) < 30.0
        line = status if fresh else fallback
        if not line:
            return
        d.text((1, 0), line[:26], font=self.font_tiny, fill=255)

    # =================================================================
    # Clawd crab character (canonical 15x16 block sprite + expression props)
    # Ported from the reference in test2.py; every helper draws on the passed
    # ImageDraw `d`. WHITE=255 / BLACK=0 on the mode-"1" buffer; eyes are cut
    # out in black from the white shell.
    # =================================================================
    def _prect(self, d, x, y, w, h, fill=255):
        if w <= 0 or h <= 0:
            return
        d.rectangle((int(x), int(y), int(x + w - 1), int(y + h - 1)), fill=fill)

    def _pline(self, d, xy, width=1, fill=255):
        d.line(tuple(int(v) for v in xy), fill=fill, width=width)

    def _urect(self, d, ox, oy, scale, x, y, w, h, fill=255):
        """Draw a rectangle in Clawd's logical sprite pixel coordinate system."""
        self._prect(d, ox + x * scale, oy + y * scale, w * scale, h * scale, fill)

    def _pstar(self, d, x, y, size=2):
        self._prect(d, x + size, y, size, size)
        self._prect(d, x, y + size, size * 3, size)
        self._prect(d, x + size, y + size * 2, size, size)

    def _pquestion(self, d, x, y, scale=2):
        for px, py in ((1, 0), (2, 0), (0, 1), (3, 1), (3, 2),
                       (2, 3), (1, 4), (1, 6)):
            self._prect(d, x + px * scale, y + py * scale, scale, scale)

    def _pexclaim(self, d, x, y, scale=2):
        self._prect(d, x, y, scale * 2, scale * 4)
        self._prect(d, x, y + scale * 5, scale * 2, scale * 2)

    def _pz(self, d, x, y, scale=2, small=False):
        if small:
            pts = ((0, 0), (1, 0), (1, 1), (0, 2), (1, 2))
        else:
            pts = ((0, 0), (1, 0), (2, 0), (3, 0),
                   (2, 1), (1, 2),
                   (0, 3), (1, 3), (2, 3), (3, 3))
        for px, py in pts:
            self._prect(d, x + px * scale, y + py * scale, scale, scale)

    def _pheart(self, d, x, y, scale=2):
        for px, py in ((0, 0), (1, 0), (3, 0), (4, 0),
                       (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
                       (1, 2), (2, 2), (3, 2),
                       (2, 3)):
            self._prect(d, x + px * scale, y + py * scale, scale, scale)

    def _pbulb(self, d, x, y, scale=2):
        # Little idea bulb: round-ish glass over a two-row screw base.
        for px, py in ((1, 0), (2, 0), (3, 0),
                       (0, 1), (4, 1), (0, 2), (4, 2),
                       (1, 3), (2, 3), (3, 3),
                       (1, 4), (3, 4), (1, 5), (2, 5), (3, 5)):
            self._prect(d, x + px * scale, y + py * scale, scale, scale)

    def _psweat(self, d, x, y, scale=2):
        # Classic anime sweat bead: narrow at the top, round at the bottom.
        for px, py in ((1, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2),
                       (1, 3)):
            self._prect(d, x + px * scale, y + py * scale, scale, scale)

    def _pblush(self, d, ox, oy, scale, x, y):
        """Two-row blush hatch in the crab's sprite coordinates (drawn BLACK so
        it reads as shading cut into the white shell)."""
        for px, py in ((0, 0), (2, 0), (1, 1), (3, 1)):
            self._urect(d, ox, oy, scale, x + px, y + py, 1, 1, BLACK)

    def _ppuff(self, d, x, y, scale=2):
        # Little huff of steam.
        for px, py in ((1, 0), (2, 0), (0, 1), (3, 1), (1, 2), (2, 2)):
            self._prect(d, x + px * scale, y + py * scale, scale, scale)

    def _pspiral(self, d, cx, cy, t, size=7):
        """A slow swirl of dots — dazed, dreamy, off-balance."""
        for i in range(6):
            a = t * 2.2 + i * (math.tau / 6)
            r = 2 + i * (size - 2) / 6.0
            self._prect(d, cx + int(math.cos(a) * r), cy + int(math.sin(a) * r), 2, 2)

    def _pbt(self, d, x, y, scale=2):
        # Compact pixel Bluetooth rune, intentionally outline-free.
        pts = [
            (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
            (4, 1), (5, 2), (4, 3),
            (4, 5), (5, 4),
            (1, 2), (2, 3), (1, 4),
        ]
        for px, py in pts:
            self._prect(d, x + px * scale, y + py * scale, scale, scale)

    _ARM_POSITIONS = {
        "normal": ((0, 9), (13, 9)),
        "up": ((0, 5), (13, 5)),
        "high": ((1, 3), (12, 3)),
        "wave_left": ((0, 5), (13, 9)),
        "wave_right": ((0, 9), (13, 5)),
        "chin": ((0, 10), (11, 10)),
        "typing_a": ((2, 11), (11, 12)),
        "typing_b": ((2, 12), (11, 11)),
        "head_scratch": ((-1, 6), (13, 9)),
        "conduct_a": ((0, 4), (13, 8)),
        "conduct_b": ((0, 8), (13, 4)),
        "sweep": ((1, 9), (12, 10)),
        "debug": ((0, 10), (13, 10)),
        # Expression poses (tag-selectable states only). NOTE: the torso spans
        # logical x=2..12, and arms are drawn *before* the shell's eye cut-outs,
        # so any pose inside that span renders white-on-white and vanishes.
        # Every pose here stays at x=0 or x=13 to remain visible.
        "cover": ((0, 4), (13, 4)),      # claws raised beside the face — shy / scared
        "cheeks": ((0, 8), (13, 8)),     # claws at eye level — swooning / awe
        "hips": ((0, 11), (13, 11)),     # akimbo — pleased with itself
        "reach": ((-1, 4), (14, 4)),     # both claws flung wide
    }

    def _crab(self, d, ox=CRAB_OX, oy=CRAB_OY, scale=CRAB_SCALE,
              body_dx=0, body_dy=0,
              eyes="normal", look_x=0, look_y=0, arms="normal", leg_phase=0,
              mouth=False):
        """Draw the canonical 15x16 block-body Clawd crab."""
        ox += body_dx
        oy += body_dy

        # Legs. Walking alternates the middle/outer pairs by one logical pixel.
        leg_y = [13, 13, 13, 13]
        if leg_phase == 1:
            leg_y = [12, 14, 14, 12]
        elif leg_phase == 2:
            leg_y = [14, 12, 12, 14]
        for x, y in zip((3, 5, 9, 11), leg_y):
            self._urect(d, ox, oy, scale, x, y, 1, 2)

        # Torso: canonical 11x7 rectangle at (2,6).
        self._urect(d, ox, oy, scale, 2, 6, 11, 7)

        # Arms: 2x2 blocks; only their placement changes by pose.
        left_arm, right_arm = self._ARM_POSITIONS.get(
            arms, self._ARM_POSITIONS["normal"])
        self._urect(d, ox, oy, scale, left_arm[0], left_arm[1], 2, 2)
        self._urect(d, ox, oy, scale, right_arm[0], right_arm[1], 2, 2)

        # Eyes are cut out in black from the white shell.
        if eyes == "closed":
            self._urect(d, ox, oy, scale, 4, 9, 2, 1, BLACK)
            self._urect(d, ox, oy, scale, 9, 9, 2, 1, BLACK)
        elif eyes == "sad":
            # Inner corners slope down, giving Clawd a visibly hurt look.
            self._urect(d, ox, oy, scale, 4, 8, 1, 1, BLACK)
            self._urect(d, ox, oy, scale, 5, 9, 1, 1, BLACK)
            self._urect(d, ox, oy, scale, 9, 9, 1, 1, BLACK)
            self._urect(d, ox, oy, scale, 10, 8, 1, 1, BLACK)
        elif eyes == "squint":
            self._urect(d, ox, oy, scale, 4 + look_x, 9 + look_y, 1, 1, BLACK)
            self._urect(d, ox, oy, scale, 10 + look_x, 9 + look_y, 1, 1, BLACK)
        elif eyes == "x":
            for cx in (4, 10):
                for px, py in ((-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)):
                    self._urect(d, ox, oy, scale, cx + px, 8 + py, 1, 1, BLACK)
        elif eyes == "wide":
            self._urect(d, ox, oy, scale, 4 + look_x, 8 + look_y, 2, 2, BLACK)
            self._urect(d, ox, oy, scale, 9 + look_x, 8 + look_y, 2, 2, BLACK)
        elif eyes == "curve":
            # Happy "^ ^" arcs — the single cutest thing the shell can do.
            for cx in (4, 9):
                self._urect(d, ox, oy, scale, cx, 9, 1, 1, BLACK)
                self._urect(d, ox, oy, scale, cx + 1, 8, 1, 1, BLACK)
                self._urect(d, ox, oy, scale, cx + 2, 9, 1, 1, BLACK)
        elif eyes == "heart":
            for cx in (4, 9):
                for px, py in ((0, 8), (2, 8), (0, 9), (1, 9), (2, 9), (1, 10)):
                    self._urect(d, ox, oy, scale, cx + px, py, 1, 1, BLACK)
        elif eyes == "star":
            for cx in (4, 9):
                for px, py in ((1, 7), (0, 8), (1, 8), (2, 8), (0, 9), (2, 9)):
                    self._urect(d, ox, oy, scale, cx + px, py, 1, 1, BLACK)
        elif eyes == "droopy":
            # Half-lidded: a thin slit low in the socket, so it reads as barely
            # awake rather than as another pair of wide eyes.
            for cx in (4, 9):
                self._urect(d, ox, oy, scale, cx, 9, 2, 1, BLACK)
        elif eyes in ("wink_l", "wink_r"):
            closed, open_ = ((4, 9) if eyes == "wink_l" else (9, 4))
            self._urect(d, ox, oy, scale, closed, 9, 2, 1, BLACK)
            self._urect(d, ox, oy, scale, open_ + 1, 8 + look_y, 1, 2, BLACK)
        else:
            self._urect(d, ox, oy, scale, 4 + look_x, 8 + look_y, 1, 2, BLACK)
            self._urect(d, ox, oy, scale, 10 + look_x, 8 + look_y, 1, 2, BLACK)

        # `mouth` stays truthy-compatible: True (and "open") draw exactly the
        # block mouth every existing caller expects.
        if mouth is True or mouth == "open":
            self._urect(d, ox, oy, scale, 6, 10, 3, 2, BLACK)
        elif mouth == "smile":
            self._urect(d, ox, oy, scale, 6, 11, 3, 1, BLACK)
            self._urect(d, ox, oy, scale, 5, 10, 1, 1, BLACK)
            self._urect(d, ox, oy, scale, 9, 10, 1, 1, BLACK)
        elif mouth == "yawn":
            self._urect(d, ox, oy, scale, 6, 10, 3, 3, BLACK)
            self._urect(d, ox, oy, scale, 7, 10, 1, 1, WHITE)
        elif mouth == "pout":
            self._urect(d, ox, oy, scale, 6, 11, 3, 1, BLACK)
            self._urect(d, ox, oy, scale, 5, 12, 1, 1, BLACK)
            self._urect(d, ox, oy, scale, 9, 12, 1, 1, BLACK)
        elif mouth == "smirk":
            self._urect(d, ox, oy, scale, 6, 11, 3, 1, BLACK)
            self._urect(d, ox, oy, scale, 9, 10, 1, 1, BLACK)

    def _sleeping_crab(self, d, ox=CRAB_OX, oy=CRAB_OY,
                       scale=CRAB_SCALE, breathe=0):
        # Splooted sleeping geometry.
        oy += breathe
        for x in (3, 5, 9, 11):
            self._urect(d, ox, oy, scale, x, 9, 1, 1)
        self._urect(d, ox, oy, scale, 1, 10, 13, 5)
        self._urect(d, ox, oy, scale, -1, 13, 2, 2)
        self._urect(d, ox, oy, scale, 14, 13, 2, 2)
        self._urect(d, ox, oy, scale, 4, 12, 2, 1, BLACK)
        self._urect(d, ox, oy, scale, 9, 12, 2, 1, BLACK)

    @staticmethod
    def _resample(levels, n):
        """Resample an arbitrary-length band-level list to exactly n bars
        (nearest-neighbour — cheap and plenty for a 128px display)."""
        if not levels:
            return None
        m = len(levels)
        if m == n:
            return [max(0.0, min(1.0, float(v))) for v in levels]
        out = []
        for i in range(n):
            src = levels[min(m - 1, int(i * m / n))]
            out.append(max(0.0, min(1.0, float(src))))
        return out

    def _mini_bars(self, d, t):
        """A short real-audio spectrum strip along the very bottom, under the
        dancing crab. Prefers fresh MusicVisualizer levels, falls back to a
        procedural animation so it never freezes."""
        n = len(self._bars)
        with self._lock:
            levels = self._audio_levels
            fresh = levels is not None and (time.time() - self._audio_ts) < 0.5
        real = self._resample(levels, n) if fresh else None
        bw, gap = 4, 1
        total = n * bw + (n - 1) * gap
        x0 = (WIDTH - total) // 2
        baseY = HEIGHT - 1
        beat = 0.5 + 0.5 * math.sin(t * 4.0)
        for i in range(n):
            if real is not None:
                target = real[i]
            else:
                tilt = 1.0 - 0.5 * (i / n)
                target = (0.5 + 0.5 * math.sin(t * 7 + i * 0.7)
                          + 0.4 * math.sin(t * 11 + i * 1.3)) * tilt
                target = max(0.0, min(1.0, target * (0.6 + 0.6 * beat)))
            cur = self._bars[i]
            cur += (target - cur) * (0.6 if target > cur else 0.25)
            self._bars[i] = max(0.0, min(1.0, cur))
            h = int(1 + self._bars[i] * 11)
            x = x0 + i * (bw + gap)
            d.rectangle((x, baseY - h, x + bw - 1, baseY), fill=255)

    def _debugger_scene(self, d, frame):
        """Crouching crab sweeping a pixel magnifying glass over the floor —
        shared by background research and web-search tool calls."""
        f = frame % 32
        crouch = 3 + (1 if (f // 4) % 2 else 0)
        self._crab(d, ox=32, body_dy=crouch, eyes="squint", look_x=1,
                   arms="debug", leg_phase=1 if (f // 4) % 2 else 2)
        cx = 88 + int(math.sin(f / 32.0 * math.tau) * 10)
        cy = 40
        d.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), outline=WHITE, width=3)
        self._pline(d, (cx + 7, cy + 7, cx + 15, cy + 15), width=3)
        self._pline(d, (cx - 6, cy, cx + 6, cy), width=1)

    # =================================================================
    # Per-state animations
    # =================================================================
    def _draw_boot(self, d, t, frame, detail):
        # Clawd walks in from the left, settles centre, then breathes. "KIKI"
        # sits along the top.
        p = min(1.0, t / 2.2)
        x = int(-60 + (CRAB_OX + 60) * p)
        if p < 1.0:
            bob = -2 if frame % 8 in (2, 3, 4) else 0
            phase = 1 if (frame // 4) % 2 == 0 else 2
            arms = "wave_left" if phase == 1 else "wave_right"
            self._crab(d, ox=x, body_dy=bob, leg_phase=phase, arms=arms)
        else:
            breath = 1 if (frame % 12) in (5, 6, 7) else 0
            self._crab(d, ox=CRAB_OX, body_dy=breath,
                       eyes="closed" if frame % 20 < 2 else "normal")
        if self.font_big is not None:
            title = "KIKI"
            try:
                w = d.textlength(title, font=self.font_big)
            except Exception:
                w = 40
            d.text(((WIDTH - w) // 2, 0), title, font=self.font_big, fill=255)

    def _draw_idle(self, d, t, frame, detail):
        # Resting Clawd: gentle breathing, occasional blink / look / head
        # scratch, and a little arms-up "happy" stretch beat — with a tiny
        # top ticker rotating through recent background activity.
        f = frame % 96
        breath = 1 if (f % 19) in range(8, 13) else 0
        look_x = 1 if 11 <= f <= 21 else (-1 if 40 <= f <= 49 else 0)
        blink = f in {4, 5, 19, 20, 44, 45, 84, 85}
        arms = "normal"
        mouth = False
        if 29 <= f <= 35:
            arms = "head_scratch"
        elif 59 <= f <= 72:                      # a happy little stretch
            arms = "high" if f < 67 else "up"
            blink = True
            mouth = True
            breath = -2 if f < 67 else 1
        self._crab(d, body_dy=breath, eyes="closed" if blink else "normal",
                   look_x=look_x, arms=arms, mouth=mouth)
        self._top_ticker(d, t)

    def _draw_wake(self, d, t, frame, detail):
        # A startled "!" then an eager little jump.
        f = frame % 40
        if f < 10:
            self._pexclaim(d, 91, 7, 2)
            self._crab(d, body_dx=-4, look_x=1)
            return
        jp = (f - 10) % 5
        jump = -8 if jp in (1, 2) else 1
        arms = "up" if jump < 0 else "high"
        self._crab(d, body_dy=jump, arms=arms)

    def _draw_listening(self, d, t, frame, detail):
        # Attentive lean, wide eyes looking up, with sound waves sweeping IN
        # from both sides toward the crab.
        f = frame % 24
        look = (-1, 0, 1, 0)[(f // 6) % 4]
        blink = f in (10, 11)
        self._crab(d, eyes="closed" if blink else "wide",
                   look_x=look, look_y=-1, arms="up")
        phase = (frame // 2) % 3
        for i in range(3):
            if (2 - i) != phase:
                continue
            r = 6 + i * 7
            d.arc((26 - r, 24 - r, 26 + r, 24 + r), 300, 60, fill=255)      # left
            d.arc((102 - r, 24 - r, 102 + r, 24 + r), 120, 240, fill=255)   # right

    def _draw_thinking(self, d, t, frame, detail):
        # Chin-in-hand pondering with a thought bubble whose loading dots fill.
        f = frame % 32
        sway = -2 if f < 8 else (2 if 16 <= f < 24 else 0)
        blink = f in {15, 16}
        self._crab(d, body_dx=sway, eyes="closed" if blink else "normal",
                   look_x=1, look_y=-1, arms="chin")
        self._prect(d, 77, 3, 34, 15)
        self._prect(d, 74, 6, 40, 9)
        self._prect(d, 82, 18, 5, 4)
        self._prect(d, 78, 22, 3, 3)
        shown = 1 + (f // 5) % 4
        for i in range(min(shown, 3)):
            self._prect(d, 84 + i * 8, 9, 3, 3, BLACK)

    def _draw_speaking(self, d, t, frame, detail):
        # Animated mouth + sound waves radiating OUT from the head.
        f = frame % 16
        mouth = (f % 4) < 2
        look = (0, 1, 0, -1)[(f // 4) % 4]
        self._crab(d, eyes="normal", look_x=look, mouth=mouth)
        phase = (frame // 2) % 3
        for i in range(3):
            if i != phase:
                continue
            r = 6 + i * 7
            d.arc((90 - r, 40 - r, 90 + r, 40 + r), 300, 60, fill=255)

    def _draw_music(self, d, t, frame, detail):
        # Clawd dances/conducts above a live real-audio spectrum strip.
        f = frame % 32
        bob = -3 if (f % 8) in (2, 3, 4) else 0
        dx = -2 if f < 16 else 2
        arms = "conduct_a" if (f // 4) % 2 == 0 else "conduct_b"
        self._crab(d, oy=-12, body_dx=dx, body_dy=bob, eyes="closed",
                   arms=arms, mouth=(f % 8 < 4))
        self._mini_bars(d, t)
        title = detail or self._music_title
        if title and self.font_tiny is not None:
            d.text((1, 0), title[:24], font=self.font_tiny, fill=255)

    def _draw_tool(self, d, t, frame, detail):
        name = detail or ""
        # Web search gets the magnifying-glass scene; other tools "type".
        if "search" in name:
            self._debugger_scene(d, frame)
            self._status_top(d, t, friendly_tool(name))
            return
        f = frame % 16
        jitter = 1 if f % 2 else 0
        arms = "typing_a" if f % 2 else "typing_b"
        look = (-1, 0, 1, 0)[(f // 4) % 4]
        self._crab(d, body_dy=jitter, eyes="squint", look_x=look, arms=arms)
        # Laptop in front of the crab.
        self._prect(d, 47, 43, 34, 14)
        self._prect(d, 44, 57, 40, 3)
        self._prect(d, 61, 48, 4, 4, BLACK)
        for i, (x, y) in enumerate(((38, 31), (48, 24), (76, 27), (88, 34))):
            yy = y - ((frame * 2 + i * 7) % 15)
            if 5 < yy < 39:
                self._prect(d, x, yy, 2, 2)
        self._status_top(d, t, friendly_tool(name))

    def _draw_idle_mind(self, d, t, frame, detail):
        # Wizard Clawd daydreaming, wand sparkles orbiting.
        f = frame % 32
        bob = -3 if 8 <= f < 16 else 0
        self._crab(d, ox=40, body_dy=bob, eyes="closed", arms="wave_right")
        self._prect(d, 50, 22 + bob, 30, 3)
        self._prect(d, 56, 16 + bob, 18, 6)
        self._prect(d, 61, 10 + bob, 10, 6)
        self._prect(d, 68, 7 + bob, 5, 4)
        self._prect(d, 61, 15 + bob, 3, 3, BLACK)
        self._pline(d, (83, 42 + bob, 102, 22 + bob), width=2)
        for i in range(4):
            a = f / 32.0 * math.tau + i * math.tau / 4
            x = 101 + int(math.cos(a) * 16)
            y = 21 + int(math.sin(a) * 12)
            self._pstar(d, x, y, 1)
        self._status_top(d, t, detail or "pondering")

    def _draw_summarizing(self, d, t, frame, detail):
        # Clawd sweeps the day's thoughts into memory.
        f = frame % 32
        self._crab(d, ox=29, body_dx=(f // 8), arms="sweep", eyes="squint")
        sweep = f % 8
        broom_x = 67 + sweep * 3
        self._pline(d, (61, 40, broom_x, 56), width=3)
        self._prect(d, broom_x - 2, 54, 24, 4)
        for i in range(4):
            dx = 90 + ((frame * 4 + i * 9) % 34)
            dy = 50 - ((frame + i * 3) % 7)
            self._prect(d, dx, dy, 2, 2)
        self._status_top(d, t, detail or "saving memory")

    def _draw_warming(self, d, t, frame, detail):
        # A slow wake-up stretch/yawn while the model prefills.
        f = frame % 32
        if f < 8:
            arms = "up"
        elif f < 16:
            arms = "high"
        elif f < 24:
            arms = "up"
        else:
            arms = "normal"
        breath = -1 if 4 <= f < 20 else 0
        self._crab(d, body_dy=breath, eyes="squint", arms=arms,
                   mouth=(8 <= f < 16))
        dots = int((t * 2) % 4)
        self._status_top(d, t, (detail or "warming up") + "." * dots)

    def _draw_vision(self, d, t, frame, detail):
        # Antenna scanning/sensing outward (beacon pose) — Kiki looking around.
        f = frame % 32
        self._crab(d)
        self._pline(d, (64, 31, 64, 18), width=2)
        self._prect(d, 61, 14, 7, 5)
        phase = (f // 4) % 4
        for r in range(1, phase + 1):
            radius = 5 + r * 6
            d.arc((64 - radius, 16 - radius // 2,
                   64 + radius, 16 + radius // 2 + 8),
                  start=195, end=345, fill=WHITE, width=1)
        self._status_top(d, t, detail or "looking around")

    def _draw_workers(self, d, t, frame, detail):
        # Clawd juggling several task packets at once.
        f = frame % 32
        self._crab(d, body_dx=-2 if f < 16 else 2,
                   look_x=(-1, 0, 1, 0)[(f // 4) % 4], arms="up")
        for i in range(3):
            a = (f / 32.0) * math.tau + i * math.tau / 3
            x = 64 + int(math.cos(a) * 28)
            y = 22 - int(abs(math.sin(a)) * 15)
            self._prect(d, x - 2, y - 2, 5, 5)
        self._status_top(d, t, detail or "working")

    def _draw_goodbye(self, d, t, frame, detail):
        # Clawd burrows down into the ground and vanishes.
        f = min(frame, 27)
        sink = int((f / 27.0) * 42)
        self._crab(d, body_dy=sink, eyes="closed")
        ground_y = 59
        d.rectangle((0, ground_y, WIDTH, HEIGHT), fill=0)
        for i in range(6):
            x = 36 + ((i * 13 + f * 4) % 57)
            y = 56 - ((f + i * 2) % 8)
            self._prect(d, x, y, 2, 2)
        self._label(d, "goodbye")

    # ---------------------------------------------- optional expressions
    def _draw_disconnected(self, d, t, frame, detail):
        # Worried Clawd; a Bluetooth rune with a slash through it.
        f = frame % 24
        look = 1 if f < 8 or f >= 18 else -1
        self._crab(d, body_dx=1 if 5 <= f <= 12 else 0, look_x=look, look_y=-1)
        self._pexclaim(d, 27, 30, 1)
        self._pbt(d, 88, 8, 2)
        self._pline(d, (88, 27, 104, 8), width=2)
        self._status_top(d, t, detail or "reconnecting")

    def _draw_confused(self, d, t, frame, detail):
        # Repeated lean/scratch/settle beats keep the uncertainty readable.
        f = frame % 64
        if 8 <= f < 24:
            dx = -min(5, (f - 8) // 2)
        elif 24 <= f < 32:
            dx = -max(0, 5 - (f - 24))
        elif 36 <= f < 52:
            dx = min(5, (f - 36) // 2)
        elif 52 <= f < 60:
            dx = max(0, 5 - (f - 52))
        else:
            dx = 0
        look = -1 if dx < 0 else (1 if dx > 0 else 0)
        scratch = (f // 3) % 2
        self._crab(d, body_dx=dx, body_dy=scratch if dx else 0,
                   look_x=look, arms="head_scratch",
                   leg_phase=(f // 4) % 3 if dx else 0)
        if 7 <= f < 30:
            self._pquestion(d, 17 + ((f - 7) // 8), 12 - ((f - 7) % 8) // 2, 2)
        if 34 <= f < 59:
            self._pquestion(d, 98 - ((f - 34) // 8), 12 - ((f - 34) % 8) // 2, 2)

    def _draw_dizzy(self, d, t, frame, detail):
        # X eyes, an uneven body wobble, and stars on independent orbits.
        f = frame % 48
        dx = int(math.sin(f / 48.0 * math.tau * 2) * 5)
        dy = (0, -1, -2, -1, 0, 1)[(f // 2) % 6]
        self._crab(d, body_dx=dx, body_dy=dy, eyes="x",
                   arms="up" if (f // 8) % 2 else "normal",
                   leg_phase=1 if dx < 0 else 2)
        for i in range(3):
            a = f / 48.0 * math.tau * 1.5 + i * math.tau / 3
            radius = 23 + ((f + i * 5) % 10)
            x = 64 + int(math.cos(a) * radius)
            y = 17 + int(math.sin(a) * (5 + i))
            self._pstar(d, x, y, 2 if (f + i * 4) % 16 < 4 else 1)

    def _draw_happy(self, d, t, frame, detail):
        # Two springy jumps per loop with arm/leg follow-through.
        f = frame % 48
        jump_arc = (0, -1, -3, -6, -9, -11, -10, -8,
                    -5, -2, 0, 1, 2, 1, 0, 0)
        jump = jump_arc[f % len(jump_arc)]
        airborne = jump < -2
        self._crab(d, body_dy=jump, eyes="curve",
                   arms="reach" if airborne else "high", mouth="open",
                   leg_phase=1 if (f // 2) % 2 else 2)
        radius = 4 + (f % 16) * 2
        for ang in (210, 245, 295, 330):
            rad = math.radians(ang)
            x = 64 + int(math.cos(rad) * radius)
            y = 31 + int(math.sin(rad) * radius)
            self._pstar(d, x, y, 2 if f % 16 < 4 else 1)

    def _draw_sad(self, d, t, frame, detail):
        # Slow slump, tiny sniff, and alternating tears. This is also used as a
        # short one-shot when the user dismisses Kiki.
        f = frame % 56
        if f < 22:
            sink = min(4, f // 5)
        elif f < 42:
            sink = 4
        else:
            sink = max(2, 4 - (f - 42) // 5)
        sniff = -1 if f in (30, 31, 36, 37) else 0
        self._crab(d, body_dx=sniff, body_dy=sink, eyes="sad",
                   arms="normal", mouth="pout")
        tear_phase = f % 18
        if tear_phase < 14:
            self._prect(d, 96 + (tear_phase // 5), 31 + tear_phase * 2, 2, 3)
        if 28 <= f < 45:
            left_tear = (f - 28) % 14
            self._prect(d, 29 - (left_tear // 6), 31 + left_tear * 2, 2, 3)

    def _draw_sleeping(self, d, t, frame, detail):
        # Splooted with a full inhale/exhale and a continuous stream of Z's.
        f = frame % 64
        breathe = (0, 0, -1, -2, -2, -1, 0, 1)[(f // 4) % 8]
        self._sleeping_crab(d, breathe=breathe)
        for i in range(3):
            z = (f + i * 18) % 54
            if z < 36:
                self._pz(d, 80 + z // 6 + i * 2, 29 - z // 2, 1,
                         small=z < 14)

    # ---- Tag-selectable expressions (see robot/oled_tags.py) ---------------
    # All of these reuse the canonical `_crab()` sprite exactly as-is; they only
    # vary pose, eye cut-outs and the props drawn around it.

    def _draw_love(self, d, t, frame, detail):
        # Heart eyes, claws clasped to cheeks, hearts drifting up and away.
        f = frame % 48
        sway = (-2, -1, 0, 1, 2, 1, 0, -1)[(f // 3) % 8]
        pulse = (0, -1, -2, -1)[(f // 2) % 4]
        self._crab(d, body_dx=sway, body_dy=pulse,
                   eyes="heart", arms="cheeks", mouth="smile")
        for i, (bx, phase) in enumerate(((22, 0), (98, 12), (14, 24), (108, 36))):
            p = (f + phase) % 48
            self._pheart(d, bx + int(math.sin((p + i) / 5) * 3),
                         51 - p, 2 if p < 10 else 1)

    def _draw_shy(self, d, t, frame, detail):
        # Claws up over the face, peeking out, cheeks lit up.
        f = frame % 56
        sway = (-2, -2, -1, 0, 1, 2, 2, 1, 0, -1)[(f // 3) % 10]
        peek = 9 <= f < 19 or 34 <= f < 47
        bob = (0, -1, -2, -1)[(f // 2) % 4] if peek else (
            1 if f % 14 in (0, 1) else 0)
        self._crab(d, body_dx=sway, body_dy=bob,
                   eyes=("normal" if peek and f % 8 < 3 else
                         ("curve" if peek else "closed")),
                   look_x=-1 if f < 28 else 1, arms="cover",
                   leg_phase=1 if sway < 0 else (2 if sway > 0 else 0))
        self._pblush(d, CRAB_OX + sway, CRAB_OY, CRAB_SCALE, 2, 10)
        self._pblush(d, CRAB_OX + sway, CRAB_OY, CRAB_SCALE, 9, 10)

    def _draw_giggle(self, d, t, frame, detail):
        # Laugh in two bursts, with a short breath between them.
        f = frame % 24
        active = f < 9 or 13 <= f < 22
        jiggle = (0, 2, -1, 1, -2, 1)[f % 6] if active else 0
        self._crab(d, body_dx=jiggle, body_dy=-2 if active and f % 3 else 0,
                   eyes="curve", arms="cheeks" if active else "normal",
                   mouth="open" if active else "smile",
                   leg_phase=1 if jiggle < 0 else 2)
        for i, (bx, by) in enumerate(((26, 20), (96, 16), (20, 34))):
            if active and (f + i * 3) % 9 < 6:
                self._pstar(d, bx, by - (f % 4), 1)

    def _draw_wink(self, d, t, frame, detail):
        # Anticipate, wink, wave the raised claw, then visibly reset.
        f = frame % 40
        wink = 7 <= f < 27
        wave = "wave_right" if 10 <= f < 16 or 21 <= f < 27 else "up"
        self._crab(d, body_dx=1 if wink else 0,
                   body_dy=-2 if 7 <= f < 12 else 0,
                   eyes="wink_r" if wink else "normal",
                   arms=wave if wink else "normal",
                   mouth="smirk" if wink else "smile")
        if 9 <= f < 25:
            self._pstar(d, 96 + (f % 3), 16 - ((f - 9) // 5), 2 if f < 14 else 1)

    def _draw_excited(self, d, t, frame, detail):
        # Fast hopping, claws flung up, sparkles blowing outward.
        f = frame % 32
        jump = (0, -1, -3, -6, -9, -11, -10, -8,
                -5, -2, 0, 2, 1, 0, -1, 0)[f % 16]
        self._crab(d, body_dy=jump, eyes="wide",
                   arms="reach" if jump < -5 else "high", mouth="open",
                   leg_phase=1 if (f // 2) % 2 else 2)
        self._pexclaim(d, 26, 14, 2)
        self._pexclaim(d, 98, 14, 2)
        radius = 5 + (f % 16) * 2
        for ang in (200, 250, 290, 340):
            rad = math.radians(ang)
            self._pstar(d, 64 + int(math.cos(rad) * radius),
                        30 + int(math.sin(rad) * radius), 2 if f % 16 < 4 else 1)

    def _draw_curious(self, d, t, frame, detail):
        # Leaning in, eyes up, one bobbing question mark. Interested, not lost —
        # this is the friendly cousin of `confused`.
        f = frame % 48
        lean = (0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 0)[f // 4]
        look_x = -1 if f < 12 else (1 if 28 <= f < 40 else 0)
        self._crab(d, body_dx=lean, body_dy=-1 if 16 <= f < 32 else 0,
                   eyes="wide", look_x=look_x, look_y=-1,
                   arms="up" if (f // 8) % 2 else "chin")
        self._pquestion(d, 22 + (f // 12), 14 - (f % 12) // 3,
                        2 if f % 24 < 5 else 1)

    def _draw_proud(self, d, t, frame, detail):
        # Visible inhale, chest-out hold, then a jaunty pleased bounce.
        f = frame % 48
        puff = (0, -1, -2, -3, -3, -2, -1, 0)[(f // 3) % 8]
        dx = 1 if 24 <= f < 36 else 0
        self._crab(d, body_dx=dx, body_dy=puff, eyes="curve",
                   arms="hips" if f % 16 < 12 else "high", mouth="smile",
                   leg_phase=1 if 24 <= f < 30 else (2 if 30 <= f < 36 else 0))
        if 8 <= f < 38:
            p = f - 8
            self._pstar(d, 96 + p // 10, 25 - p // 3,
                        2 if p < 8 else 1)

    def _draw_sulk(self, d, t, frame, detail):
        # Turned away in a huff, with an impatient foot-tap and repeated puff.
        f = frame % 56
        dx = -min(4, f // 3) if f < 14 else (-4 if f < 45 else -max(0, 4 - (f - 45) // 2))
        tap = (f // 3) % 2
        self._crab(d, body_dx=dx, eyes="closed", arms="normal",
                   body_dy=tap, mouth="pout",
                   leg_phase=1 if tap else 2)
        for phase in (0, 24):
            p = (f - phase) % 56
            if p < 16:
                self._ppuff(d, 94 + p, 29 - p, 1)

    def _draw_surprised(self, d, t, frame, detail):
        # A big jolt, nervous settle, then a smaller aftershock.
        f = frame % 36
        jolt_seq = (-9, -9, -7, -5, -3, -1, 1, 0)
        jolt = jolt_seq[f] if f < 8 else (
            (-5, -3, -1, 0)[f - 20] if 20 <= f < 24 else 0)
        shake = (-1, 1, 0, 1, -1, 0)[f % 6] if 8 <= f < 32 else 0
        self._crab(d, body_dx=shake, body_dy=jolt, eyes="wide",
                   arms="reach" if f < 5 else "high", mouth="open",
                   leg_phase=1 if f % 4 < 2 else 2)
        punctuation_bob = (-2, -1, 0, 1)[(f // 2) % 4]
        self._pexclaim(d, 24, 12 + punctuation_bob, 2)
        self._pexclaim(d, 100, 12 - punctuation_bob, 2)
        if f < 7 or 20 <= f < 24:
            for x in (30, 94):
                self._pline(d, (x, 40, x + (6 if x < 64 else -6), 46), width=2)

    def _draw_sleepy(self, d, t, frame, detail):
        # Heavy lids and a big yawn — drowsy, but still sitting up (unlike the
        # fully splooted `sleeping`).
        f = frame % 64
        yawning = 16 <= f < 37
        nod = (0, 1, 2, 3, 3, 2, 1, 0)[(f // 4) % 8]
        self._crab(d, body_dx=-1 if 42 <= f < 50 else 0,
                   body_dy=nod + (1 if yawning else 0),
                   eyes="closed" if yawning else "droopy",
                   arms="up" if yawning else "normal",
                   mouth="yawn" if yawning else False)
        for phase in (0, 30):
            z = (f - phase) % 64
            if z < 24:
                self._pz(d, 92 + z // 5, 28 - z, 1, small=z < 10)

    def _draw_idea(self, d, t, frame, detail):
        # Think, notice, then celebrate while the bulb visibly pulses.
        f = frame % 48
        on = 12 <= f < 43
        pop = 12 <= f < 19
        self._crab(d, body_dy=-4 if pop else (-1 if on else 0),
                   eyes="wide" if on else "squint",
                   arms="high" if pop else ("up" if on else "chin"),
                   mouth="smile" if on else False,
                   leg_phase=(f // 3) % 3 if pop else 0)
        bulb_scale = 2 if not on or f % 12 < 9 else 1
        bulb_bob = (0, -1, -2, -1)[f % 4] if on else 0
        self._pbulb(d, 22 + (2 - bulb_scale) * 2,
                    14 + (2 - bulb_scale) * 3 + bulb_bob, bulb_scale)
        if on:
            ray_len = 13 + ((f - 12) % 12)
            for ang in (200, 235, 270, 305, 340):
                rad = math.radians(ang)
                x0, y0 = 30 + int(math.cos(rad) * 12), 20 + int(math.sin(rad) * 12)
                x1, y1 = 30 + int(math.cos(rad) * ray_len), 20 + int(math.sin(rad) * ray_len)
                self._pline(d, (x0, y0, x1, y1), width=2)

    def _draw_mischief(self, d, t, frame, detail):
        # Sneak out and back on tiptoe with glances over each shoulder.
        f = frame % 48
        step = f if f < 24 else 47 - f
        creep = -5 + step // 2
        self._crab(d, body_dx=creep, body_dy=-2 if (f // 2) % 2 else 0,
                   eyes="squint", look_x=1, arms="chin", mouth="smirk",
                   leg_phase=1 if (f // 3) % 2 else 2)
        if f % 16 < 10:
            self._pstar(d, 98 + (f % 3), 20 - (f % 10) // 2,
                        2 if f % 16 < 3 else 1)

    def _draw_scared(self, d, t, frame, detail):
        # Irregular trembling, ducking, claws over the face, and a falling bead.
        f = frame % 32
        tremble = (-2, 1, -1, 2, -2, 2, -1, 1)[f % 8]
        crouch = (0, 1, 2, 3, 3, 2, 1, 0)[(f // 2) % 8]
        self._crab(d, body_dx=tremble, body_dy=crouch,
                   eyes="wide", arms="cover",
                   leg_phase=1 if tremble < 0 else 2)
        self._psweat(d, 92 + (f // 8), 13 + (f % 16) * 2, 2 if f % 16 < 8 else 1)
        self._pexclaim(d, 26, 16, 1)

    def _draw_awe(self, d, t, frame, detail):
        # Star-struck: slow levitation, pulsing eyes, and uneven star orbits.
        f = frame % 56
        float_y = (0, -1, -2, -3, -2, -1, 0)[(f // 4) % 7]
        self._crab(d, body_dy=float_y,
                   eyes="star" if f % 28 < 23 else "wide",
                   arms="cheeks" if f % 14 < 10 else "reach", mouth="open")
        for i in range(4):
            a = f / 56.0 * math.tau * 1.5 + i * (math.tau / 4)
            self._pstar(d, 64 + int(math.cos(a) * (36 + i * 2)),
                        25 + int(math.sin(a) * (9 + i)),
                        2 if (f + i * 5) % 18 < 4 else 1)

    def _draw_face(self, d, t, frame, detail):
        """A person's face card: the dithered face crop on the left, their name
        (big) + subtitle on the right. Falls back to a placeholder box if no
        bitmap is available (emulated / decode failure)."""
        img = self._face_img
        name = self._face_name or "?"
        sub = self._face_subtitle
        fx, fy = 4, 4
        if img is not None:
            try:
                self._image.paste(img, (fx, fy))
            except Exception:
                img = None
        if img is None:
            d.rectangle((fx, fy, fx + 55, fy + 55), outline=255, fill=0)
            d.text((fx + 20, fy + 24), "?", font=self.font_big or self.font, fill=255)
        # Text column to the right of the 56px face.
        tx = 66
        font_name = self.font_big or self.font
        d.text((tx, 8), name[:9], font=font_name, fill=255)
        if sub:
            d.text((tx, 34), sub[:11], font=self.font, fill=255)


class MusicVisualizer(threading.Thread):
    """Taps the actual audio that's playing and feeds real spectrum band levels
    to the OLED so the music animation is in sync with the song.

    It reads raw PCM from the PulseAudio playback *monitor* with `parec`
    (the same source `cava`/`pavucontrol` use), runs a small numpy FFT per
    frame, buckets it into log-spaced bands and pushes them via
    `oled_manager.set_audio_levels()`. Everything is best-effort: if numpy or
    parec or PulseAudio isn't available the thread simply exits and the OLED
    falls back to its procedural spectrum, so off-robot/headless runs are fine.
    """

    RATE = 22050
    CHUNK = 1024          # samples per FFT frame (~21 updates/sec)
    BANDS = 24            # spectrum buckets sent to the display

    def __init__(self, manager):
        super().__init__(daemon=True)
        self._manager = manager
        self._stop = threading.Event()
        self._proc = None
        self._agc = 1e-3      # auto-gain running peak (keeps quiet songs visible)

    def keep_running(self):
        """Called if music restarts while the thread is still alive — nothing
        to do, the tap is already live."""
        self._stop.clear()

    def stop(self):
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def _monitor_source(self):
        """Best-effort name of the monitor source to capture."""
        import subprocess
        try:
            sink = subprocess.run(
                ["pactl", "get-default-sink"], capture_output=True, text=True,
                timeout=2).stdout.strip()
            if sink:
                return sink + ".monitor"
        except Exception:
            pass
        return "@DEFAULT_MONITOR@"

    def _band_edges(self, nbins):
        # Log-spaced bin edges from ~40 Hz up to Nyquist across BANDS buckets.
        import numpy as np
        lo, hi = 40.0, self.RATE / 2.0
        freqs = np.logspace(math.log10(lo), math.log10(hi), self.BANDS + 1)
        edges = (freqs / (self.RATE / 2.0) * (nbins - 1)).astype(int)
        return np.clip(edges, 1, nbins - 1)

    def run(self):
        import subprocess
        try:
            import numpy as np
        except Exception as e:
            print(f"[OLED] MusicVisualizer: numpy unavailable ({e}).")
            return

        source = self._monitor_source()
        cmd = ["parec", "--format=s16le", f"--rate={self.RATE}",
               "--channels=1", "--latency-msec=40", "-d", source]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("[OLED] MusicVisualizer: `parec` not found; procedural spectrum.")
            return
        except Exception as e:
            print(f"[OLED] MusicVisualizer: capture failed ({e}).")
            return

        window = np.hanning(self.CHUNK)
        nbytes = self.CHUNK * 2  # int16
        nbins = self.CHUNK // 2 + 1
        edges = self._band_edges(nbins)

        try:
            while not self._stop.is_set():
                raw = self._proc.stdout.read(nbytes)
                if not raw or len(raw) < nbytes:
                    break
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                samples /= 32768.0
                spectrum = np.abs(np.fft.rfft(samples * window))
                bands = np.empty(self.BANDS, dtype=np.float32)
                for i in range(self.BANDS):
                    a, b = edges[i], max(edges[i] + 1, edges[i + 1])
                    bands[i] = spectrum[a:b].mean() if b > a else spectrum[a]
                # Auto-gain: track a decaying peak so soft tracks still fill the
                # screen and loud ones don't clip permanently.
                peak = float(bands.max())
                self._agc = max(peak, self._agc * 0.995, 1e-3)
                norm = np.clip(bands / self._agc, 0.0, 1.0) ** 0.55
                self._manager.set_audio_levels(norm.tolist())
        except Exception as e:
            print(f"[OLED] MusicVisualizer loop ended: {e}")
        finally:
            try:
                if self._proc is not None:
                    self._proc.terminate()
            except Exception:
                pass


def get_oled_tag_prompt_note() -> str:
    """Extra system-prompt text teaching the model the `<oled:name>` vocabulary.

    Generated from `EXPRESSION_STATES` (sorted, so it is byte-stable across
    restarts) rather than hand-written into config.json. Two reasons:

    * The taught vocabulary can never drift from what the display can draw.
    * Persona modes REPLACE `llm.system_prompt` wholesale
      (`tools_and_config/config.json` `assistant_modes`), so anything written
      into that prompt is invisible in every mode — which is exactly why the
      neck tags were never available outside the default persona. Appending it
      here, next to `get_tts_system_prompt_note()`, reaches every mode.

    KV-cache note: this is a *static* suffix appended once when the system
    prompt is built. It becomes part of the warmed prefix, so turns still
    prefill nothing but the user's new message.
    """
    names = ", ".join(f"`<oled:{n}>`" for n in sorted(EXPRESSION_STATES))
    return (
        "\n\n## YOUR FACE (silent OLED expression tags)\n"
        "You have a small screen showing your face. You can set your expression by "
        f"dropping a silent tag inline in your reply. Available: {names}.\n"
        "- The tag is NOT spoken and NOT part of your words — it only changes your face.\n"
        "- **NEVER start a reply or a sentence with one.** Put it after the first few "
        "words, or at the end of a sentence. A tag before your first word delays your "
        "voice, and that matters more than the face does.\n"
        "- At most one per sentence, and only when your mood actually changes — the "
        "expression stays on your face until you change it or stop talking.\n"
        "- Match it to what you are saying: "
        "`Oh! <oled:surprised> I did not expect that.` / "
        "`That is... really sweet <oled:shy> anyway, moving on.`"
    )


def make_progress_fn(base_state: str, prefix: str = ""):
    """Build a `progress_fn(phase, detail)` for `run_agent_loop` that narrates
    the agent's work on the OLED under the given `base_state` animation
    (idle_mind / workers / tool). `phase` is
    "thinking" | "tool" | "done"; the matching status line is shown beneath the
    animation. `prefix` is an optional lead-in (e.g. a worker name)."""
    head = f"{prefix}: " if prefix else ""

    def _fn(phase, detail=""):
        try:
            if phase == "tool":
                status = f"{head}now {friendly_tool(detail)}"
            elif phase == "done":
                summary = " ".join((detail or "").split())
                if len(summary) > 80:
                    summary = summary[:77] + "..."
                status = f"{head}{summary}" if summary else f"{head}done"
            else:  # thinking
                status = f"{head}{detail or 'thinking'}"
            oled_manager.set_progress(base_state, status)
        except Exception:
            pass

    return _fn


# Global singleton (constructed at import, like lcd_manager).
oled_manager = OLEDDisplayManager()
