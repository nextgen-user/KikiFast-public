import re
import time
import os

# ============================================================================
# Movement Tag Parsing (from KIKI-SMART agent.py)
# ============================================================================

# Regex patterns for movement tags
_MOVEMENT_PATTERNS = [
    # <turn(angle)> or <turn_right(angle)> or <turn_left(angle)>
    (re.compile(r'<turn\((-?\d+)\)>'), 'turn'),
    (re.compile(r'<turn_right\((\d+)\)>'), 'turn_right'),
    (re.compile(r'<turn_left\((\d+)\)>'), 'turn_left'),
    # <forward(dist)> / <backward(dist)>
    (re.compile(r'<forward\((\d+)\)>'), 'forward'),
    (re.compile(r'<backward\((\d+)\)>'), 'backward'),
    # <strafe_left(dist)> / <strafe_right(dist)>
    (re.compile(r'<strafe_left\((\d+)\)>'), 'strafe_left'),
    (re.compile(r'<strafe_right\((\d+)\)>'), 'strafe_right'),
    # <diagonal_front_left(dist)> etc.
    (re.compile(r'<diagonal_front_left\((\d+)\)>'), 'diagonal_front_left'),
    (re.compile(r'<diagonal_front_right\((\d+)\)>'), 'diagonal_front_right'),
    (re.compile(r'<diagonal_back_left\((\d+)\)>'), 'diagonal_back_left'),
    (re.compile(r'<diagonal_back_right\((\d+)\)>'), 'diagonal_back_right'),
    # <move(angle, dist)>
    (re.compile(r'<move\((-?\d+),\s*(\d+)\)>'), 'move'),
]

# Pattern to match ANY movement tag for stripping
_ALL_MOVEMENT_TAG_RE = re.compile(
    r'<(?:turn|turn_right|turn_left|forward|backward|strafe_left|strafe_right|'
    r'diagonal_front_left|diagonal_front_right|diagonal_back_left|diagonal_back_right|'
    r'move)\([^)]*\)>'
)

def extract_movement_tags(text: str) -> list:
    """Extract movement commands from LLM response text."""
    movements = []
    for pattern, move_type in _MOVEMENT_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()
            if move_type == 'move':
                movements.append({'type': move_type, 'angle': int(groups[0]), 'distance': int(groups[1])})
            elif move_type in ('turn', 'turn_right', 'turn_left'):
                movements.append({'type': move_type, 'angle': int(groups[0])})
            else:
                movements.append({'type': move_type, 'distance': int(groups[0])})
    return movements

def strip_movement_tags(text: str) -> str:
    """Remove movement tags from text before sending to TTS."""
    return _ALL_MOVEMENT_TAG_RE.sub('', text).strip()

def execute_movements(movements: list):
    """Execute extracted movement commands via motor_control (runs in thread)."""
    if not movements:
        return

    try:
        import robot.motor_control as motor_control
    except ImportError:
        print("[Movement] motor_control not available, skipping movements")
        return

    try:
        motor_control.init_gpio()
        for move in movements:
            move_type = move['type']
            print(f"[Movement] Executing: {move}")

            if move_type == 'turn':
                angle = move['angle']
                if angle > 0:
                    motor_control.turn_right()
                else:
                    motor_control.turn_left()
                duration = abs(angle) / 360.0 * 2.0  # rough timing
                time.sleep(max(0.2, min(3.0, duration)))
                motor_control.stop()
            elif move_type == 'turn_right':
                motor_control.turn_right()
                duration = move['angle'] / 360.0 * 2.0
                time.sleep(max(0.2, min(3.0, duration)))
                motor_control.stop()
            elif move_type == 'turn_left':
                motor_control.turn_left()
                duration = move['angle'] / 360.0 * 2.0
                time.sleep(max(0.2, min(3.0, duration)))
                motor_control.stop()
            elif move_type == 'forward':
                motor_control.forward()
                time.sleep(max(0.2, min(5.0, move['distance'] / 100.0)))
                motor_control.stop()
            elif move_type == 'backward':
                motor_control.backward()
                time.sleep(max(0.2, min(5.0, move['distance'] / 100.0)))
                motor_control.stop()
            else:
                func = getattr(motor_control, move_type, None)
                if func:
                    func()
                    time.sleep(0.5)
                    motor_control.stop()

        motor_control.release_gpio()
    except Exception as e:
        print(f"[Movement] Error executing movements: {e}")
        try:
            motor_control.stop()
            motor_control.release_gpio()
        except:
            pass
