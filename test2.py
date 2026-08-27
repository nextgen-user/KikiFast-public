"""
Clawd Tank crab animations for a 128x64 monochrome SSD1306 OLED.

This version keeps the canonical Clawd Tank character geometry:
- 15x16 logical pixel sprite
- 11x7 rectangular torso
- 2x2 side arms
- four 1x2 legs
- two 1x2 eyes

The large aquarium UI, clock, cards, and grass are intentionally omitted so the
small OLED can devote its pixels to Clawd and the prop that communicates each
state.

Dependencies:
    pip install pillow adafruit-blinka adafruit-circuitpython-ssd1306

Example:
    animator = ClawdOLED(disp)
    animator.set_state("thinking")
    while True:
        animator.tick()
        # Do microphone / AI / motor work here too.
"""

import math
import time

import board
import busio
from PIL import Image, ImageDraw
import adafruit_ssd1306


OLED_WIDTH = 128
OLED_HEIGHT = 64
OLED_ADDRESS = 0x3C

# Set True to cycle through every animation automatically.
DEMO_ALL_STATES = True
DEMO_STATE_SECONDS = 3.0

WHITE = 255
BLACK = 0

STATES = (
    "idle",
    "alert",
    "happy",
    "sleeping",
    "disconnected",
    "thinking",
    "typing",
    "juggling",
    "building",
    "confused",
    "dizzy",
    "sweeping",
    "walking",
    "going_away",
    "debugger",
    "wizard",
    "conducting",
    "beacon",
)

STATE_ALIASES = {
    "working_1": "typing",
    "working_2": "juggling",
    "working_3": "building",
}

# Repository behavior: alert, happy, sweeping, and going-away are one-shots.
ONE_SHOT_LENGTHS = {
    "alert": 40,
    "happy": 30,
    "sweeping": 32,
    "going_away": 28,
}


class ClawdOLED:
    """Non-blocking Pillow renderer for the canonical Clawd pixel crab."""

    def __init__(self, display):
        self.display = display
        self.width = display.width
        self.height = display.height
        self.image = Image.new("1", (self.width, self.height), BLACK)
        self.draw = ImageDraw.Draw(self.image)

        self.state = "idle"
        self.fallback_state = "idle"
        self.frame = 0
        self.last_frame_time = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_state(self, state, fallback="idle"):
        state = self._normalise_state(state)
        fallback = self._normalise_state(fallback)
        self.state = state
        self.fallback_state = fallback
        self.frame = 0
        self.last_frame_time = 0.0
        self.render()

    def play_oneshot(self, state, fallback="idle"):
        self.set_state(state, fallback)

    def tick(self):
        """Advance one frame when due. Call repeatedly from your main loop."""
        now = time.monotonic()
        delay = self._frame_delay(self.state)
        if self.last_frame_time and now - self.last_frame_time < delay:
            return

        self.last_frame_time = now
        self.render()
        self.frame += 1

        limit = ONE_SHOT_LENGTHS.get(self.state)
        if limit is not None and self.frame >= limit:
            fallback = self.fallback_state
            self.set_state(fallback, fallback)

    def render(self):
        self.draw.rectangle((0, 0, self.width - 1, self.height - 1), fill=BLACK)
        getattr(self, f"_draw_{self.state}")(self.frame)
        self.display.image(self.image)
        self.display.show()

    @staticmethod
    def _normalise_state(state):
        name = str(state).strip().lower()
        name = STATE_ALIASES.get(name, name)
        if name not in STATES:
            raise ValueError(f"Unknown state {state!r}. Valid states: {', '.join(STATES)}")
        return name

    @staticmethod
    def _frame_delay(state):
        if state in {"idle", "sleeping", "disconnected"}:
            return 1.0 / 6.0
        if state in {"alert", "happy"}:
            return 1.0 / 10.0
        return 1.0 / 8.0

    # ------------------------------------------------------------------
    # Pixel helpers
    # ------------------------------------------------------------------

    def _rect(self, x, y, w, h, fill=WHITE):
        if w <= 0 or h <= 0:
            return
        self.draw.rectangle((int(x), int(y), int(x + w - 1), int(y + h - 1)), fill=fill)

    def _line(self, xy, width=1, fill=WHITE):
        self.draw.line(tuple(int(v) for v in xy), fill=fill, width=width)

    def _u_rect(self, ox, oy, scale, x, y, w, h, fill=WHITE):
        """Draw a rectangle in Clawd's logical SVG pixel coordinate system."""
        self._rect(ox + x * scale, oy + y * scale, w * scale, h * scale, fill)

    def _pixel_star(self, x, y, size=2):
        self._rect(x + size, y, size, size)
        self._rect(x, y + size, size * 3, size)
        self._rect(x + size, y + size * 2, size, size)

    def _question(self, x, y, scale=2):
        # Same block-style question mark used by the SVG.
        pixels = ((1, 0), (2, 0), (0, 1), (3, 1), (3, 2),
                  (2, 3), (1, 4), (1, 6))
        for px, py in pixels:
            self._rect(x + px * scale, y + py * scale, scale, scale)

    def _exclamation(self, x, y, scale=2):
        self._rect(x, y, scale * 2, scale * 4)
        self._rect(x, y + scale * 5, scale * 2, scale * 2)

    def _z(self, x, y, scale=2, small=False):
        if small:
            pixels = ((0, 0), (1, 0), (1, 1), (0, 2), (1, 2))
        else:
            pixels = ((0, 0), (1, 0), (2, 0), (3, 0),
                      (2, 1), (1, 2),
                      (0, 3), (1, 3), (2, 3), (3, 3))
        for px, py in pixels:
            self._rect(x + px * scale, y + py * scale, scale, scale)

    def _bluetooth(self, x, y, scale=2):
        # Compact pixel Bluetooth rune, intentionally outline-free.
        pts = [
            (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
            (4, 1), (5, 2), (4, 3),
            (4, 5), (5, 4),
            (1, 2), (2, 3), (1, 4),
        ]
        for px, py in pts:
            self._rect(x + px * scale, y + py * scale, scale, scale)

    # ------------------------------------------------------------------
    # Canonical Clawd character
    # ------------------------------------------------------------------

    def _draw_crab(
        self,
        ox=42,
        oy=13,
        scale=3,
        body_dx=0,
        body_dy=0,
        eyes="normal",
        look_x=0,
        look_y=0,
        arms="normal",
        leg_phase=0,
        mouth=False,
    ):
        """Draw the exact 15x16 block-body design from clawd-static-base.svg."""
        ox += body_dx
        oy += body_dy

        # Legs. Walking alternates the middle/outer pairs by one logical pixel.
        leg_y = [13, 13, 13, 13]
        if leg_phase == 1:
            leg_y = [12, 14, 14, 12]
        elif leg_phase == 2:
            leg_y = [14, 12, 12, 14]

        for x, y in zip((3, 5, 9, 11), leg_y):
            self._u_rect(ox, oy, scale, x, y, 1, 2)

        # Torso: canonical 11x7 rectangle at (2,6).
        self._u_rect(ox, oy, scale, 2, 6, 11, 7)

        # Arms are still 2x2 blocks; only their placement changes by pose.
        arm_positions = {
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
        }
        left_arm, right_arm = arm_positions.get(arms, arm_positions["normal"])
        self._u_rect(ox, oy, scale, left_arm[0], left_arm[1], 2, 2)
        self._u_rect(ox, oy, scale, right_arm[0], right_arm[1], 2, 2)

        # Eyes are cut out in black from the white shell.
        if eyes == "closed":
            self._u_rect(ox, oy, scale, 4, 9, 2, 1, BLACK)
            self._u_rect(ox, oy, scale, 9, 9, 2, 1, BLACK)
        elif eyes == "squint":
            self._u_rect(ox, oy, scale, 4 + look_x, 9 + look_y, 1, 1, BLACK)
            self._u_rect(ox, oy, scale, 10 + look_x, 9 + look_y, 1, 1, BLACK)
        elif eyes == "x":
            for cx in (4, 10):
                for px, py in ((-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)):
                    self._u_rect(ox, oy, scale, cx + px, 8 + py, 1, 1, BLACK)
        else:
            self._u_rect(ox, oy, scale, 4 + look_x, 8 + look_y, 1, 2, BLACK)
            self._u_rect(ox, oy, scale, 10 + look_x, 8 + look_y, 1, 2, BLACK)

        if mouth:
            self._u_rect(ox, oy, scale, 6, 10, 3, 2, BLACK)

    def _draw_sleeping_crab(self, ox=42, oy=13, scale=3, breathe=0):
        # Exact splooted sleeping geometry from clawd-sleeping.svg.
        oy += breathe
        for x in (3, 5, 9, 11):
            self._u_rect(ox, oy, scale, x, 9, 1, 1)
        self._u_rect(ox, oy, scale, 1, 10, 13, 5)
        self._u_rect(ox, oy, scale, -1, 13, 2, 2)
        self._u_rect(ox, oy, scale, 14, 13, 2, 2)
        self._u_rect(ox, oy, scale, 4, 12, 2, 1, BLACK)
        self._u_rect(ox, oy, scale, 9, 12, 2, 1, BLACK)

    # ------------------------------------------------------------------
    # State renderers
    # ------------------------------------------------------------------

    def _draw_idle(self, frame):
        f = frame % 96  # 16 seconds at 6 FPS, matching the source SVG.
        breath = 1 if (f % 19) in range(8, 13) else 0
        look_x = 1 if 11 <= f <= 21 else (-1 if 40 <= f <= 49 else 0)
        blink = f in {4, 5, 19, 20, 44, 45, 84, 85}

        arms = "normal"
        mouth = False
        if 29 <= f <= 35:
            arms = "head_scratch"
        elif 59 <= f <= 72:
            arms = "high" if f < 67 else "up"
            blink = True
            mouth = True
            breath = -2 if f < 67 else 1

        self._draw_crab(body_dy=breath, eyes="closed" if blink else "normal",
                        look_x=look_x, arms=arms, mouth=mouth)

    def _draw_alert(self, frame):
        f = min(frame, 39)
        if f < 10:
            self._exclamation(91, 7, 2)
            self._draw_crab(body_dx=-4, look_x=1)
            return

        jump_phase = (f - 10) % 5
        jump = -8 if jump_phase in {1, 2} else 1
        arms = "up" if jump < 0 else "high"
        self._draw_crab(body_dy=jump, arms=arms)

    def _draw_happy(self, frame):
        f = min(frame, 29)
        jump = -8 if (f % 8) in {2, 3, 4} else 0
        self._draw_crab(body_dy=jump, eyes="closed", arms="high")

        # Pixel celebration sparkles expanding away from Clawd.
        radius = min(18, 4 + f)
        for angle in (210, 245, 295, 330):
            rad = math.radians(angle)
            x = 64 + int(math.cos(rad) * radius)
            y = 31 + int(math.sin(rad) * radius)
            self._pixel_star(x, y, 1)

    def _draw_sleeping(self, frame):
        f = frame % 36
        breathe = -1 if 8 <= f <= 16 else 0
        self._draw_sleeping_crab(oy=13, breathe=breathe)

        # Three staggered floating Z particles.
        z_phase = f % 18
        if z_phase < 12:
            self._z(82 + (z_phase // 4) * 2, 25 - z_phase, 1, small=True)
        if 8 <= f < 26:
            self._z(91 - ((f - 8) // 5), 16 - ((f - 8) // 3), 1)

    def _draw_disconnected(self, frame):
        f = frame % 24
        look = 1 if f < 8 or f >= 18 else -1
        self._draw_crab(body_dx=1 if 5 <= f <= 12 else 0, look_x=look, look_y=-1)
        self._exclamation(27, 30, 1)
        self._bluetooth(88, 8, 2)
        # Slash through the symbol to communicate lost connection.
        self._line((88, 27, 104, 8), width=2)

    def _draw_thinking(self, frame):
        f = frame % 32
        sway = -2 if f < 8 else (2 if 16 <= f < 24 else 0)
        blink = f in {15, 16}
        self._draw_crab(body_dx=sway, eyes="closed" if blink else "normal",
                        look_x=1, look_y=-1, arms="chin")

        # Pixel thought bubble with sequential loading dots.
        self._rect(77, 3, 34, 15)
        self._rect(74, 6, 40, 9)
        self._rect(82, 18, 5, 4)
        self._rect(78, 22, 3, 3)
        shown = 1 + (f // 5) % 4
        for i in range(min(shown, 3)):
            self._rect(84 + i * 8, 9, 3, 3, BLACK)

    def _draw_typing(self, frame):
        f = frame % 16
        jitter = 1 if f % 2 else 0
        arms = "typing_a" if f % 2 else "typing_b"
        look = (-1, 0, 1, 0)[(f // 4) % 4]
        self._draw_crab(body_dy=jitter, eyes="squint", look_x=look, arms=arms)

        # Laptop placed in front, matching the SVG composition.
        self._rect(47, 43, 34, 14)
        self._rect(44, 57, 40, 3)
        self._rect(61, 48, 4, 4, BLACK)

        packets = ((38, 31), (48, 24), (76, 27), (88, 34))
        for i, (x, y) in enumerate(packets):
            yy = y - ((f * 2 + i * 7) % 15)
            if 5 < yy < 39:
                self._rect(x, yy, 2, 2)

    def _draw_juggling(self, frame):
        f = frame % 32
        self._draw_crab(body_dx=-2 if f < 16 else 2,
                        look_x=(-1, 0, 1, 0)[(f // 4) % 4], arms="up")

        # Three packets separated by one third of a circular path.
        for i in range(3):
            a = (f / 32.0) * math.tau + i * math.tau / 3
            x = 64 + int(math.cos(a) * 28)
            y = 22 - int(abs(math.sin(a)) * 15)
            self._rect(x - 2, y - 2, 5, 5)

    def _draw_building(self, frame):
        f = frame % 16
        impact = f in {7, 8}
        oy = 14 + (2 if impact else (-2 if 3 <= f <= 5 else 0))
        self._draw_crab(ox=31, oy=oy, eyes="closed" if impact else "normal",
                        arms="wave_right")

        # Pixel hard hat based on the repository's block hat.
        self._rect(40, oy + 6, 30, 3)
        self._rect(43, oy + 3, 24, 3)
        self._rect(49, oy, 5, 3)
        self._rect(58, oy, 5, 3)

        # Hammer alternates between raised and impact positions.
        if impact:
            self._line((77, 34, 91, 48), width=3)
            self._rect(87, 44, 13, 6)
        else:
            self._line((75, 38, 82, 17), width=3)
            self._rect(76, 13, 14, 6)

        # Anvil and impact sparks.
        self._rect(92, 51, 18, 4)
        self._rect(96, 47, 10, 5)
        if impact:
            self._pixel_star(105, 39, 1)
            self._pixel_star(113, 46, 1)

    def _draw_confused(self, frame):
        f = frame % 48
        dx = -5 if 8 <= f < 20 else (5 if 25 <= f < 37 else 0)
        look = -1 if dx < 0 else (1 if dx > 0 else 0)
        self._draw_crab(body_dx=dx, look_x=look, arms="head_scratch")
        if 10 <= f < 22:
            self._question(17, 9, 2)
        if 27 <= f < 39:
            self._question(98, 9, 2)

    def _draw_dizzy(self, frame):
        f = frame % 32
        dx = int(math.sin(f / 32.0 * math.tau) * 4)
        self._draw_crab(body_dx=dx, eyes="x")
        for i in range(3):
            a = f / 32.0 * math.tau + i * math.tau / 3
            x = 64 + int(math.cos(a) * 26)
            y = 18 + int(math.sin(a) * 5)
            self._pixel_star(x, y, 1)

    def _draw_sweeping(self, frame):
        f = min(frame, 31)
        self._draw_crab(ox=29, body_dx=(f // 8), arms="sweep", eyes="squint")

        sweep = f % 8
        broom_x = 67 + sweep * 3
        self._line((61, 40, broom_x, 56), width=3)
        self._rect(broom_x - 2, 54, 24, 4)
        for i in range(4):
            dust_x = 90 + ((f * 4 + i * 9) % 34)
            dust_y = 50 - ((f + i * 3) % 7)
            self._rect(dust_x, dust_y, 2, 2)

    def _draw_walking(self, frame):
        f = frame % 80
        x = -45 + int((self.width + 50) * f / 79)
        bob = -2 if f % 8 in {2, 3, 4} else 0
        phase = 1 if (f // 4) % 2 == 0 else 2
        self._draw_crab(ox=x, body_dy=bob, leg_phase=phase,
                        arms="wave_left" if phase == 1 else "wave_right")

    def _draw_going_away(self, frame):
        f = min(frame, 27)
        sink = int((f / 27.0) * 42)
        self._draw_crab(body_dy=sink, eyes="closed")

        # Ground/dust line hides the crab as it burrows down.
        ground_y = 59
        self._rect(0, ground_y, self.width, self.height - ground_y, BLACK)
        for i in range(6):
            x = 36 + ((i * 13 + f * 4) % 57)
            y = 56 - ((f + i * 2) % 8)
            self._rect(x, y, 2, 2)

    def _draw_debugger(self, frame):
        f = frame % 32
        crouch = 3 + (1 if (f // 4) % 2 else 0)
        self._draw_crab(ox=32, body_dy=crouch, eyes="squint",
                        look_x=1, arms="debug",
                        leg_phase=1 if (f // 4) % 2 else 2)

        # Pixel magnifying glass sweeping over the floor.
        cx = 88 + int(math.sin(f / 32.0 * math.tau) * 10)
        cy = 43
        self.draw.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), outline=WHITE, width=3)
        self._line((cx + 7, cy + 7, cx + 17, cy + 17), width=3)
        self._line((cx - 6, cy, cx + 6, cy), width=1)

    def _draw_wizard(self, frame):
        f = frame % 32
        bob = -3 if 8 <= f < 16 else 0
        self._draw_crab(ox=40, body_dy=bob, eyes="closed", arms="wave_right")

        # Crooked pixel wizard hat.
        self._rect(50, 22 + bob, 30, 3)
        self._rect(56, 16 + bob, 18, 6)
        self._rect(61, 10 + bob, 10, 6)
        self._rect(68, 7 + bob, 5, 4)
        self._rect(61, 15 + bob, 3, 3, BLACK)

        # Wand and moving sparkles.
        self._line((83, 42 + bob, 102, 22 + bob), width=2)
        for i in range(4):
            a = f / 32.0 * math.tau + i * math.tau / 4
            x = 101 + int(math.cos(a) * 16)
            y = 21 + int(math.sin(a) * 12)
            self._pixel_star(x, y, 1)

    def _draw_conducting(self, frame):
        f = frame % 32
        arms = "conduct_a" if (f // 4) % 2 == 0 else "conduct_b"
        bob = -2 if 8 <= f < 16 else 0
        self._draw_crab(body_dy=bob, eyes="closed", arms=arms)

        # Data packets flow in an arch from one hand to the other.
        for i in range(6):
            p = ((f + i * 5) % 32) / 31.0
            x = 25 + int(p * 78)
            y = 31 - int(math.sin(p * math.pi) * 22)
            self._rect(x, y, 3, 3)

    def _draw_beacon(self, frame):
        f = frame % 32
        self._draw_crab()

        # Antenna attached to the top of the canonical torso.
        self._line((64, 31, 64, 18), width=2)
        self._rect(61, 14, 7, 5)

        phase = (f // 4) % 4
        for r in range(1, phase + 1):
            radius = 5 + r * 6
            self.draw.arc((64 - radius, 16 - radius // 2,
                           64 + radius, 16 + radius // 2 + 8),
                          start=195, end=345, fill=WHITE, width=1)


def initialise_display():
    """Initialise the same I2C SSD1306 stack as the user's sample."""
    i2c = busio.I2C(board.SCL, board.SDA)
    disp = adafruit_ssd1306.SSD1306_I2C(
        OLED_WIDTH, OLED_HEIGHT, i2c, addr=OLED_ADDRESS
    )
    disp.fill(0)
    disp.show()
    return disp


def main():
    try:
        disp = initialise_display()
        animator = ClawdOLED(disp)

        if DEMO_ALL_STATES:
            while True:
                for state in STATES:
                    print(f"Clawd state: {state}")
                    # Demo mode holds one-shots instead of immediately falling back.
                    animator.state = state
                    animator.fallback_state = state
                    animator.frame = 0
                    animator.last_frame_time = 0.0
                    started = time.monotonic()
                    while time.monotonic() - started < DEMO_STATE_SECONDS:
                        animator.tick()
                        time.sleep(0.005)
        else:
            animator.set_state("idle")
            while True:
                animator.tick()
                time.sleep(0.005)

    except KeyboardInterrupt:
        print("Stopped.")
    except Exception as exc:
        print(f"Error initializing or displaying to OLED: {exc}")


if __name__ == "__main__":
    main()
