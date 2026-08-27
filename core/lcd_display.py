import threading
import queue
import time
import re
import random
import unicodedata
from dataclasses import dataclass

# Safely try importing RPLCD
try:
    from RPLCD.i2c import CharLCD
    RPLCD_AVAILABLE = True
except ImportError:
    RPLCD_AVAILABLE = False

# --- LCD Configuration ---
I2C_ADDR = 0x27  
CHIP = 'PCF8574' 

# The HD44780 character ROM differs between LCD variants. Use only ordinary
# periods for the animation; slash/backslash spinners and emoji are unreliable.
SPINNER_FRAMES = (".", "..", "...")
SPINNER_FRAME_SECONDS = 0.35
SPINNER_VERB_SECONDS = 2.2

SPINNER_VERBS = [
  'Accomplishing',
  'Actioning',
  'Actualizing',
  'Architecting',
  'Baking',
  'Beaming',
  "Beboppin'",
  'Befuddling',
  'Billowing',
  'Blanching',
  'Bloviating',
  'Boogieing',
  'Boondoggling',
  'Booping',
  'Bootstrapping',
  'Brewing',
  'Bunning',
  'Burrowing',
  'Calculating',
  'Canoodling',
  'Caramelizing',
  'Cascading',
  'Catapulting',
  'Cerebrating',
  'Channeling',
  'Channelling',
  'Choreographing',
  'Churning',
  'Clauding',
  'Coalescing',
  'Cogitating',
  'Combobulating',
  'Composing',
  'Computing',
  'Concocting',
  'Considering',
  'Contemplating',
  'Cooking',
  'Crafting',
  'Creating',
  'Crunching',
  'Crystallizing',
  'Cultivating',
  'Deciphering',
  'Deliberating',
  'Determining',
  'Dilly-dallying',
  'Discombobulating',
  'Doing',
  'Doodling',
  'Drizzling',
  'Ebbing',
  'Effecting',
  'Elucidating',
  'Embellishing',
  'Enchanting',
  'Envisioning',
  'Evaporating',
  'Fermenting',
  'Fiddle-faddling',
  'Finagling',
  'Flambéing',
  'Flibbertigibbeting',
  'Flowing',
  'Flummoxing',
  'Fluttering',
  'Forging',
  'Forming',
  'Frolicking',
  'Frosting',
  'Gallivanting',
  'Galloping',
  'Garnishing',
  'Generating',
  'Gesticulating',
  'Germinating',
  'Gitifying',
  'Grooving',
  'Gusting',
  'Harmonizing',
  'Hashing',
  'Hatching',
  'Herding',
  'Honking',
  'Hullaballooing',
  'Hyperspacing',
  'Ideating',
  'Imagining',
  'Improvising',
  'Incubating',
  'Inferring',
  'Infusing',
  'Ionizing',
  'Jitterbugging',
  'Julienning',
  'Kneading',
  'Leavening',
  'Levitating',
  'Lollygagging',
  'Manifesting',
  'Marinating',
  'Meandering',
  'Metamorphosing',
  'Misting',
  'Moonwalking',
  'Moseying',
  'Mulling',
  'Mustering',
  'Musing',
  'Nebulizing',
  'Nesting',
  'Newspapering',
  'Noodling',
  'Nucleating',
  'Orbiting',
  'Orchestrating',
  'Osmosing',
  'Perambulating',
  'Percolating',
  'Perusing',
  'Philosophising',
  'Photosynthesizing',
  'Pollinating',
  'Pondering',
  'Pontificating',
  'Pouncing',
  'Precipitating',
  'Prestidigitating',
  'Processing',
  'Proofing',
  'Propagating',
  'Puttering',
  'Puzzling',
  'Quantumizing',
  'Razzle-dazzling',
  'Razzmatazzing',
  'Recombobulating',
  'Reticulating',
  'Roosting',
  'Ruminating',
  'Sautéing',
  'Scampering',
  'Schlepping',
  'Scurrying',
  'Seasoning',
  'Shenaniganing',
  'Shimmying',
  'Simmering',
  'Skedaddling',
  'Sketching',
  'Slithering',
  'Smooshing',
  'Sock-hopping',
  'Spelunking',
  'Spinning',
  'Sprouting',
  'Stewing',
  'Sublimating',
  'Swirling',
  'Swooping',
  'Symbioting',
  'Synthesizing',
  'Tempering',
  'Thinking',
  'Thundering',
  'Tinkering',
  'Tomfoolering',
  'Topsy-turvying',
  'Transfiguring',
  'Transmuting',
  'Twisting',
  'Undulating',
  'Unfurling',
  'Unravelling',
  'Vibing',
  'Waddling',
  'Wandering',
  'Warping',
  'Whatchamacalliting',
  'Whirlpooling',
  'Whirring',
  'Whisking',
  'Wibbling',
  'Working',
  'Wrangling',
  'Zesting',
  'Zigzagging',
]

@dataclass
class _LCDWrite:
    line1: str
    line2: str
    stream_id: int | None = None
    queued_at: float = 0.0

class LCDDisplayManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LCDDisplayManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.lcd = None
        self.enabled = False
        self._queue = queue.Queue()
        self._state_lock = threading.Lock()
        self._stream_serial = 0
        self._active_stream_id = None
        self._last_commit = None
        self._commit_observer = None
        self._spinner_lock = threading.Lock()
        self._spinner_generation = 0
        self._spinner_thread = None
        
        # Keep track of last displayed lines to avoid redundant writes
        self.last_line1 = ""
        self.last_line2 = ""
        
        if RPLCD_AVAILABLE:
            try:
                self.lcd = CharLCD(
                    i2c_expander=CHIP, 
                    address=I2C_ADDR, 
                    port=1, 
                    cols=16, 
                    rows=2, 
                    dotsize=8
                )
                self.lcd.clear()
                self.enabled = True
                print("[LCD] Initialized physical 16x2 LCD successfully.")
            except Exception as e:
                print(f"[LCD] Warning: Failed to initialize physical LCD ({e}). Running in emulated mode.")
        else:
            print("[LCD] RPLCD library not found. Running in emulated mode.")
            
        # Start background update worker thread for asynchronous writes
        self._worker_thread = threading.Thread(target=self._lcd_worker, daemon=True)
        self._worker_thread.start()
        
        self._initialized = True

    def _lcd_worker(self):
        """Background thread worker to process LCD writes asynchronously."""
        while True:
            try:
                item = self._queue.get()
                if item is None:
                    break
                candidates = [item]
                
                # If there are newer updates in the queue, skip intermediate ones
                # to prevent lagging and backlog on streaming text.
                while self._queue.qsize() > 0:
                    try:
                        next_item = self._queue.get_nowait()
                        self._queue.task_done()
                        if next_item is None:
                            return
                        candidates.append(next_item)
                    except queue.Empty:
                        break

                # Pick the newest still-valid frame. A stale speech frame must
                # not cause an earlier normal status in the same drained batch
                # to be discarded.
                write = None
                with self._state_lock:
                    active_stream_id = self._active_stream_id
                for candidate in reversed(candidates):
                    if (candidate.stream_id is None
                            or candidate.stream_id == active_stream_id):
                        write = candidate
                        break
                if write is None:
                    self._queue.task_done()
                    continue

                # Speech writes belong to one playback session. Once that
                # session is cancelled/finished, never let a queued stale word
                # overwrite the next Listening/Thinking status.
                if write.stream_id is not None:
                    with self._state_lock:
                        if write.stream_id != self._active_stream_id:
                            self._queue.task_done()
                            continue

                # Write to the physical LCD. Hold the shared I2C lock so OLED
                # transactions on the same bus can't interleave (Errno 121).
                if self.enabled and self.lcd:
                    try:
                        from core.i2c_bus import I2C_LOCK
                        with I2C_LOCK:
                            self.lcd.cursor_pos = (0, 0)
                            self.lcd.write_string(write.line1.ljust(16)[:16])
                            self.lcd.cursor_pos = (1, 0)
                            self.lcd.write_string(write.line2.ljust(16)[:16])
                    except Exception as e:
                        print(f"[LCD] Write error: {e}")

                committed_at = time.monotonic()
                commit = {
                    "stream_id": write.stream_id,
                    "line1": write.line1,
                    "line2": write.line2,
                    "queued_at": write.queued_at,
                    "committed_at": committed_at,
                }
                with self._state_lock:
                    self._last_commit = commit
                    observer = self._commit_observer
                if observer is not None:
                    try:
                        observer(dict(commit))
                    except Exception:
                        pass
                
                self._queue.task_done()
            except Exception as e:
                print(f"[LCD Worker] Error: {e}")

    def write(self, line1: str, line2: str = "", stream_id=None):
        """Queue two lines to be written asynchronously to the 16x2 LCD."""
        self._stop_spinner()
        # Clean up text
        line1 = self._clean_text(line1)[:16]
        line2 = self._clean_text(line2)[:16]

        self._queue_lines(line1, line2, stream_id=stream_id)

    def write_raw(self, line1: str, line2: str = ""):
        """Write literal printable characters without markdown/tag stripping.

        Used for user-entered secrets that must be inspectable exactly as entered.
        Control characters are replaced with spaces so they cannot alter the panel.
        """
        self._stop_spinner()
        line1 = "".join(char if char.isprintable() else " " for char in str(line1))[:16]
        line2 = "".join(char if char.isprintable() else " " for char in str(line2))[:16]
        self._queue_lines(line1, line2)

    def _queue_lines(self, line1: str, line2: str, stream_id=None):
        """Queue two already-normalized LCD rows."""

        with self._state_lock:
            if stream_id is not None:
                if stream_id != self._active_stream_id:
                    return
            # Pad to 16 characters to overwrite previous characters
            line1_padded = line1.ljust(16)
            line2_padded = line2.ljust(16)

            # Only queue if text has changed to prevent redundant work.
            if line1_padded == self.last_line1 and line2_padded == self.last_line2:
                return
            self.last_line1 = line1_padded
            self.last_line2 = line2_padded
        self._queue.put(_LCDWrite(line1, line2, stream_id, time.monotonic()))

    def clear(self):
        """Queue a clear operation for the LCD."""
        self._stop_spinner()
        self.last_line1 = ""
        self.last_line2 = ""
        self._queue.put(_LCDWrite("", "", None, time.monotonic()))

    def begin_stream_session(self):
        """Start a cancellable speech-display session and return its id."""
        # TTS may create its stream before the first word is ready.  Retire the
        # thinking animation now so it cannot overwrite that first speech frame.
        self._stop_spinner()
        with self._state_lock:
            self._stream_serial += 1
            self._active_stream_id = self._stream_serial
            self.last_line1 = ""
            self.last_line2 = ""
            return self._active_stream_id

    def end_stream_session(self, stream_id):
        """Invalidate pending speech frames from ``stream_id`` immediately."""
        with self._state_lock:
            if self._active_stream_id == stream_id:
                self._active_stream_id = None
                self.last_line1 = ""
                self.last_line2 = ""

    def set_commit_observer(self, callback):
        """Install an optional calibration/test observer for physical commits."""
        with self._state_lock:
            self._commit_observer = callback

    def get_last_commit(self):
        with self._state_lock:
            return dict(self._last_commit) if self._last_commit else None

    def _stop_spinner(self):
        """Invalidate the active spinner without blocking on its daemon thread."""
        with self._spinner_lock:
            self._spinner_generation += 1
            self._spinner_thread = None

    def _start_spinner(self):
        """Start an LCD-safe thinking animation, replacing any older one."""
        with self._spinner_lock:
            self._spinner_generation += 1
            generation = self._spinner_generation
            thread = threading.Thread(
                target=self._spinner_loop,
                args=(generation,),
                name="lcd-thinking-spinner",
                daemon=True,
            )
            self._spinner_thread = thread
            thread.start()

    def _spinner_loop(self, generation):
        """Animate trailing dots after a verb, keeping the second row empty."""
        verb = self._spinner_verb()
        verb_changed_at = time.monotonic()
        frame_index = 0

        while True:
            now = time.monotonic()
            if now - verb_changed_at >= SPINNER_VERB_SECONDS:
                verb = self._spinner_verb(excluding=verb)
                verb_changed_at = now

            # Serialize the validity check and enqueue with _stop_spinner().
            # Thus an expired animation can never enqueue after a newer status.
            with self._spinner_lock:
                if generation != self._spinner_generation:
                    return
                self._queue_lines(
                    verb + SPINNER_FRAMES[frame_index],
                    "",
                    stream_id=None,
                )

            frame_index = (frame_index + 1) % len(SPINNER_FRAMES)
            time.sleep(SPINNER_FRAME_SECONDS)

    @staticmethod
    def _spinner_verb(excluding=None):
        """Return an ASCII-only verb with room for three trailing dots."""
        candidates = SPINNER_VERBS
        if excluding:
            alternatives = [verb for verb in candidates
                            if LCDDisplayManager._format_spinner_verb(verb) != excluding]
            if alternatives:
                candidates = alternatives
        return LCDDisplayManager._format_spinner_verb(random.choice(candidates))

    @staticmethod
    def _format_spinner_verb(verb):
        # Transliterate accented entries such as "Sauteing"; unknown glyphs are
        # dropped rather than being sent as multi-byte characters to RPLCD.
        safe = unicodedata.normalize("NFKD", str(verb)).encode("ascii", "ignore").decode()
        safe = " ".join(safe.split())
        if len(safe) > 13:
            safe = safe[:13].rstrip()
        return safe

    def update_status(self, action: str, details: str = None):
        """Always print what Kiki is doing to the terminal and show on LCD."""
        status_msg = f"[Kiki] 🤖 {action}"
        if details:
            status_msg += f": {details}"
        print(status_msg)

        # Mirror the status onto the futuristic OLED animation surface. Lazy
        # import + best-effort so the LCD never depends on the OLED subsystem.
        try:
            from core.oled_display import oled_manager
            oled_manager.update_status(action, details)
        except Exception:
            pass

        # Map typical states to clean LCD layouts
        action_lower = action.lower()
        if "thinking" in action_lower and action_lower.strip() == "thinking":
            self._start_spinner()
            return

        self._stop_spinner()
        if "wake word" in action_lower:
            self.write("Wake Word", "Detected! 🔊")
        elif "listening" in action_lower:
            self.write("Listening...", details if details else "")
        elif "thinking" in action_lower:
            # Long-running background thinking keeps a static verb to avoid
            # needlessly saturating the shared I2C bus for minutes at a time.
            self.write(self._spinner_verb() + "...",
                       details if details else "")
        elif "speaking" in action_lower:
            # Let the streaming sentence updates handle the screen
            pass
        elif "idle" in action_lower:
            self.write("Kiki is Idle", "Waiting for 'heyy'")
        elif "summarizing" in action_lower:
            self.write("Summarizing...", "💾 Saving memo")
        elif "idle mind" in action_lower:
            self.write("Idle mind...", "cloud session")
        elif "warm" in action_lower:
            self.write("Warming up model", "Please wait...")
        elif "autonomous vision" in action_lower:
            self.write("Vision Event", "Looking around 📷")
        elif "startup" in action_lower:
            self.write("Kiki Robot", "Starting up...")
        elif "goodbye" in action_lower or "shutdown" in action_lower:
            self.write("Goodbye! 👋", "Powering down")
        else:
            # Default fallback write
            self.write(action, details if details else "")

    def display_stream(self, prefix: str, text: str, stream_id=None):
        """Wraps and displays streaming text (like STT or TTS)."""
        line1, line2 = self.wrap_text_16x2(text)
        if not line2:
            self.write(prefix, line1, stream_id=stream_id)
        else:
            self.write(line1, line2, stream_id=stream_id)

    @staticmethod
    def wrap_text_16x2(text: str):
        """
        Wraps word tokens into lines of max 16 chars.
        Returns the last 2 lines for sliding-window vertical scroll.
        """
        words = text.split()
        lines = []
        current_line = []
        current_len = 0
        for word in words:
            # Plus 1 for space if it's not the first word in the line
            space_len = 1 if current_line else 0
            if current_len + len(word) + space_len <= 16:
                current_line.append(word)
                current_len += len(word) + space_len
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_len = len(word)
        if current_line:
            lines.append(" ".join(current_line))
        
        if not lines:
            return "", ""
        if len(lines) == 1:
            return lines[0], ""
        
        return lines[-2], lines[-1]

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip markdown and voice/bracket/XML tags."""
        if not text:
            return ""
        # Remove markdown symbols
        text = re.sub(r'[*_`#\-\[\]]', '', text)
        # Remove bracket tags like [laughter], [gasp]
        text = re.sub(r'\[.*?\]', '', text)
        # Remove angle/XML tags
        text = re.sub(r'<.*?>', '', text)
        # Standardize whitespace
        text = " ".join(text.split())
        return text

# Global singleton
lcd_manager = LCDDisplayManager()
