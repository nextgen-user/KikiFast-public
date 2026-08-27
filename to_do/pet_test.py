"""
pet_mode.py  🐾
================
Detects petting via ultrasonic sensors (hand waved close to front or rear)
and responds with one of 25 unique, cute, cuddly pet-like movements.

Usage (standalone):
    python pet_mode.py

Usage (inside a larger app):
    from pet_mode import start_pet_mode, stop_pet_mode
    start_pet_mode()
    ...
    stop_pet_mode()

HOW PETTING IS DETECTED
-----------------------
  • Any sensor reading below PET_DIST_CM counts as a "close" reading.
  • Two consecutive close readings = "being petted".
  • The robot performs one random cute movement, then waits MIN_MOVE_GAP
    seconds before it can react again (so petting doesn't flood movements).
  • Losing contact resets the count and prints a little message.
"""

import time
import random
import threading
import signal
import gpiod
from gpiod.line import Direction, Value

import motor_control as mc

# ─── Sensor pin config ────────────────────────────────────────────────────────
CHIP_PATH  = '/dev/gpiochip4'
FRONT_TRIG = 20
FRONT_ECHO = 21
BACK_TRIG  = 6
BACK_ECHO  = 19

# ─── Tuning knobs ─────────────────────────────────────────────────────────────
PET_DIST_CM   = 18     # hand must be within this (cm) to count as a pet
POLL_SEC      = 0.07   # sensor polling interval  — ~14 readings/sec
MIN_MOVE_GAP  = 1.2    # min seconds between cute movements while being petted
CONFIRM_HITS  = 2      # consecutive close readings needed to trigger reaction

# ─── Speed presets (keep everything gentle & small) ──────────────────────────
SPD_GENTLE  = 22       # very slow — snuggly moves
SPD_NORMAL  = 30       # normal cuddle speed
SPD_EXCITED = 44       # "can't contain myself" speed

# ─── Module state ─────────────────────────────────────────────────────────────
_running    = False
_thread     = None
_sensor_req = None


# ══════════════════════════════════════════════════════════════════════════════
#   SENSOR HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _dist(req, trig, echo):
    """Fire one ultrasonic pulse and return distance in cm (999 on timeout)."""
    try:
        req.set_value(trig, Value.INACTIVE)
        time.sleep(0.000002)
        req.set_value(trig, Value.ACTIVE)
        time.sleep(0.00001)
        req.set_value(trig, Value.INACTIVE)

        deadline = time.time() + 0.03
        t0 = time.time()
        while req.get_value(echo) == Value.INACTIVE:
            t0 = time.time()
            if t0 > deadline:
                return 999

        t1 = time.time()
        while req.get_value(echo) == Value.ACTIVE:
            t1 = time.time()
            if t1 > deadline:
                return 999

        return round((t1 - t0) * 17150, 1)
    except Exception:
        return 999


# ══════════════════════════════════════════════════════════════════════════════
#   MOVEMENT BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

def _go(fn, secs, spd=None):
    """Run movement `fn` for `secs` seconds at optional `spd`, then stop."""
    if spd is not None:
        mc.update_speed(spd)
    fn()
    time.sleep(secs)
    mc.stop()


def _pause(secs=0.10):
    """Stop and wait briefly."""
    mc.stop()
    time.sleep(secs)


# ══════════════════════════════════════════════════════════════════════════════
#   🐾  25 UNIQUE CUTE PET REACTIONS  🐾
# ══════════════════════════════════════════════════════════════════════════════

def wiggle_happy():
    """Side-to-side shimmy — the classic happy-dog wiggle."""
    mc.update_speed(SPD_NORMAL)
    for _ in range(4):
        _go(mc.strafe_left,  0.11)
        _go(mc.strafe_right, 0.11)


def butt_wiggle():
    """Rear rocks left/right while nose stays still — pure puppy energy."""
    mc.update_speed(SPD_NORMAL)
    for _ in range(4):
        _go(mc.turn_front_axis_left,  0.13)
        _go(mc.turn_front_axis_right, 0.13)


def head_tilt():
    """Tilts 'head' left, pauses, rights itself — confused and adorable."""
    _go(mc.turn_left,  0.19, SPD_GENTLE)
    _pause(0.15)
    _go(mc.turn_right, 0.13, SPD_GENTLE)


def nuzzle_forward():
    """Nudges gently toward the hand, then retreats shyly."""
    _go(mc.forward,  0.22, SPD_GENTLE)
    _pause(0.12)
    _go(mc.backward, 0.18, SPD_GENTLE)


def happy_bounce():
    """Three quick forward-back hops — over-excited puppy!"""
    mc.update_speed(SPD_NORMAL + 6)
    for _ in range(3):
        _go(mc.forward,  0.10)
        _go(mc.backward, 0.10)


def tail_chase():
    """Partial spin trying to catch its own tail."""
    _go(mc.turn_right, 0.38, SPD_EXCITED)
    _pause(0.05)
    _go(mc.turn_left,  0.12, SPD_NORMAL)


def shy_lean_back():
    """Modest backwards retreat — bashful kitty receiving affection."""
    _go(mc.backward, 0.26, SPD_GENTLE)
    _pause(0.18)
    _go(mc.forward,  0.16, SPD_GENTLE)


def happy_feet():
    """Rapid tiny shuffles — body vibrating with excitement."""
    mc.update_speed(SPD_NORMAL)
    for _ in range(6):
        _go(mc.strafe_left,  0.06)
        _go(mc.strafe_right, 0.06)


def stretch_yawn():
    """Slow deliberate forward stretch, then a lazy return — sleepy cat."""
    _go(mc.forward,  0.38, SPD_GENTLE - 5)
    _pause(0.28)
    _go(mc.backward, 0.30, SPD_GENTLE)


def boop_back():
    """Quick boop right back at the hand, then shy retreat."""
    _go(mc.forward,  0.12, SPD_EXCITED)
    _go(mc.backward, 0.20, SPD_NORMAL)


def roll_over_shimmy():
    """Diagonal wiggles — attempting (and failing) to roll over."""
    mc.update_speed(SPD_NORMAL)
    for _ in range(3):
        _go(mc.diagonal_front_left,  0.11)
        _go(mc.diagonal_front_right, 0.11)


def paw_tap():
    """Front-axis rocks like tapping a paw repeatedly on the ground."""
    mc.update_speed(SPD_GENTLE)
    for _ in range(4):
        _go(mc.turn_rear_axis_left,  0.11)
        _go(mc.turn_rear_axis_right, 0.11)


def lean_into_pet():
    """Leans diagonally toward the sensor — pressing into the scratch."""
    _go(mc.diagonal_front_left,  0.22, SPD_GENTLE)
    _pause(0.10)
    _go(mc.diagonal_back_right, 0.18, SPD_GENTLE)


def curious_step():
    """Steps toward you, tilts, peeks left, then rights itself."""
    _go(mc.forward,    0.18, SPD_GENTLE)
    _pause(0.08)
    _go(mc.turn_left,  0.16, SPD_GENTLE)
    _pause(0.06)
    _go(mc.turn_right, 0.10, SPD_GENTLE)


def playful_dart():
    """Sideways dart-and-return — kitten pounce energy."""
    _go(mc.strafe_right, 0.15, SPD_EXCITED)
    _pause(0.05)
    _go(mc.strafe_left,  0.20, SPD_EXCITED)
    _pause(0.04)
    _go(mc.strafe_right, 0.09, SPD_NORMAL)


def contentment_sway():
    """Slow, rhythmic side-to-side arc — deep purring contentment."""
    mc.update_speed(SPD_GENTLE - 5)
    for _ in range(3):
        _go(mc.turn_left,  0.22)
        _go(mc.turn_right, 0.22)


def zoomie_burst():
    """Sudden excitement burst: sprint, spin, then calm down."""
    _go(mc.forward,   0.18, SPD_EXCITED + 10)
    _pause(0.06)
    _go(mc.turn_left, 0.16, SPD_EXCITED)
    _pause(0.06)
    _go(mc.backward,  0.15, SPD_NORMAL)


def snuggle_press():
    """Creeps forward very slowly — pushing its whole body into you."""
    _go(mc.forward,  0.35, SPD_GENTLE - 8)
    _pause(0.25)
    _go(mc.backward, 0.28, SPD_GENTLE - 5)


def head_nod():
    """Tiny forward-back nods — YES, I LOVE THIS. YES. YES. YES."""
    mc.update_speed(SPD_GENTLE)
    for _ in range(4):
        _go(mc.forward,  0.09)
        _go(mc.backward, 0.09)


def excited_spin_fragment():
    """Random-direction partial spin — can't decide which way to spin!"""
    direction = random.choice([mc.turn_left, mc.turn_right])
    _go(direction, 0.30, SPD_NORMAL + 5)


def rear_march():
    """Rear wheels alternate swing — marching the booty in celebration."""
    mc.update_speed(SPD_NORMAL)
    for _ in range(5):
        _go(mc.swing_turn_back_left,  0.09)
        _go(mc.swing_turn_back_right, 0.09)


def belly_flop_shimmy():
    """Back-diagonal wiggles — the belly flop happy dance."""
    mc.update_speed(SPD_NORMAL)
    for _ in range(3):
        _go(mc.diagonal_back_left,  0.12)
        _go(mc.diagonal_back_right, 0.12)


def coy_turn_away():
    """Turns away … then turns back. Playing hard to get."""
    _go(mc.turn_right, 0.24, SPD_GENTLE)
    _pause(0.22)
    _go(mc.turn_left,  0.24, SPD_GENTLE)


def love_combo():
    """Nuzzle + side wiggle + retreat — maximum affection sequence."""
    _go(mc.forward, 0.14, SPD_GENTLE)
    _pause(0.05)
    mc.update_speed(SPD_NORMAL)
    for _ in range(2):
        _go(mc.strafe_left,  0.09)
        _go(mc.strafe_right, 0.09)
    _go(mc.backward, 0.12, SPD_GENTLE)


def freeze_and_enjoy():
    """Stops completely and just … soaks it in. Pure cat mode."""
    mc.stop()
    time.sleep(random.uniform(0.6, 1.3))
    # tiny approving nod at the end
    _go(mc.forward,  0.08, SPD_GENTLE)
    _go(mc.backward, 0.07, SPD_GENTLE)


# ── Master reaction table ─────────────────────────────────────────────────────

PET_REACTIONS = [
    (wiggle_happy,       "🐶  happy wiggle!"),
    (butt_wiggle,        "🐕  butt wiggle!"),
    (head_tilt,          "🤔  head tilt~"),
    (nuzzle_forward,     "🥰  nuzzle forward~"),
    (happy_bounce,       "🐇  bouncy bounce!"),
    (tail_chase,         "🌀  chasing tail!"),
    (shy_lean_back,      "😊  shy lean back~"),
    (happy_feet,         "🪩  happy feet!"),
    (stretch_yawn,       "😸  stretchy yawn~"),
    (boop_back,          "👉  boop back!"),
    (roll_over_shimmy,   "🙃  roll-over shimmy~"),
    (paw_tap,            "🐾  paw tap tap!"),
    (lean_into_pet,      "💕  leaning in~"),
    (curious_step,       "👀  curious peek!"),
    (playful_dart,       "⚡  playful dart!"),
    (contentment_sway,   "😌  content sway~"),
    (zoomie_burst,       "💨  mini zoomie!"),
    (snuggle_press,      "🤗  snuggle press~"),
    (head_nod,           "✅  yes yes YES!"),
    (excited_spin_fragment, "🌪️  spinny spin!"),
    (rear_march,         "🥁  booty march!"),
    (belly_flop_shimmy,  "💃  belly shimmy~"),
    (coy_turn_away,      "🙈  playing coy~"),
    (love_combo,         "💖  love combo!!"),
    (freeze_and_enjoy,   "😻  just vibing~"),
]


# ══════════════════════════════════════════════════════════════════════════════
#   SENSOR INIT
# ══════════════════════════════════════════════════════════════════════════════

def _init_sensors():
    global _sensor_req
    cfg = {
        FRONT_TRIG: gpiod.LineSettings(direction=Direction.OUTPUT,
                                       output_value=Value.INACTIVE),
        FRONT_ECHO: gpiod.LineSettings(direction=Direction.INPUT),
        BACK_TRIG:  gpiod.LineSettings(direction=Direction.OUTPUT,
                                       output_value=Value.INACTIVE),
        BACK_ECHO:  gpiod.LineSettings(direction=Direction.INPUT),
    }
    _sensor_req = gpiod.request_lines(CHIP_PATH, consumer="pet_mode", config=cfg)
    print("🐾  Sensors ready.")


# ══════════════════════════════════════════════════════════════════════════════
#   PETTING DETECTION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def _pet_loop():
    """Background thread: poll sensors, detect pets, trigger reactions."""
    last_move_time = 0
    last_reaction  = None
    hit_count      = 0
    was_petting    = False

    print("\n" + "─" * 52)
    print("  🐾  PET MODE ACTIVE")
    print("  Wave your hand near the front or rear sensor!")
    print("─" * 52 + "\n")

    while _running:
        df = _dist(_sensor_req, FRONT_TRIG, FRONT_ECHO)
        db = _dist(_sensor_req, BACK_TRIG,  BACK_ECHO)

        hand_close = (df < PET_DIST_CM) or (db < PET_DIST_CM)

        # Hysteresis counter — need CONFIRM_HITS consecutive close readings
        if hand_close:
            hit_count = min(hit_count + 1, CONFIRM_HITS + 2)
        else:
            hit_count = max(hit_count - 1, 0)

        being_petted = hit_count >= CONFIRM_HITS
        now          = time.time()

        if being_petted and (now - last_move_time) >= MIN_MOVE_GAP:
            # Pick random reaction, never repeat back-to-back
            pool = [r for r in PET_REACTIONS if r is not last_reaction]
            reaction_fn, label = random.choice(pool)
            last_reaction = (reaction_fn, label)

            sensor_tag = "FRONT" if df < PET_DIST_CM else "BACK "
            print(f"  ✋ [{sensor_tag}]  →  {label}")

            try:
                reaction_fn()
            except Exception as err:
                print(f"  ⚠️  movement error: {err}")
                mc.stop()

            last_move_time = time.time()
            was_petting    = True

        elif not hand_close and was_petting and hit_count == 0:
            print("  💤  Petting stopped … aahhh that was nice~")
            was_petting = False

        time.sleep(POLL_SEC)


# ══════════════════════════════════════════════════════════════════════════════
#   PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def start_pet_mode():
    """Initialise hardware and start the petting-detection thread."""
    global _running, _thread

    if _running:
        print("Pet Mode is already running.")
        return

    mc.init_gpio()
    _init_sensors()

    _running = True
    _thread  = threading.Thread(target=_pet_loop, daemon=True)
    _thread.start()
    print("🐾  Pet Mode started!")


def stop_pet_mode():
    """Stop the detection thread and release all hardware resources."""
    global _running, _sensor_req

    _running = False
    if _thread and _thread.is_alive():
        _thread.join(timeout=2)

    mc.stop()
    mc.release_gpio()

    if _sensor_req:
        _sensor_req.release()
        _sensor_req = None

    print("\n🐾  Pet Mode stopped. Goodnight~ 🌙")


# ══════════════════════════════════════════════════════════════════════════════
#   STANDALONE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    def _handle_signal(sig, frame):
        print("\n  Ctrl+C caught — shutting down…")
        stop_pet_mode()
        raise SystemExit(0)

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    start_pet_mode()

    # Keep the main thread alive so the daemon thread keeps running.
    while _running:
        time.sleep(0.5)