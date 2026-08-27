import gpiod
import time
import threading
import serial
from gpiod.line import Direction, Value

# --- Constants & Pin Configuration ---
# Reconstructed from bytecode integer literals (e.g., é = 17, é = 22)
R1_PINS = (22, 17)  # Front Right ?
R2_PINS = (24, 23)  # Back Right ?
L1_PINS = (16, 26)  # Front Left ?
L2_PINS = (5, 27)   # Back Left ?

ENA_PIN = 12        # PWM Speed Control Pin A
ENB_PIN = 13        # PWM Speed Control Pin B

SERIAL_PORT = '/dev/ttyAMA0'
BAUD_RATE = 9600
CHIP_PATH = '/dev/gpiochip4'

# Global variables
chip_request = None
pwm_r = None
pwm_l = None
ser = None
current_speed = 80  # Default speed
LEFT_TRIM = 0.7
RIGHT_TRIM = 0

class SoftPWM:
    """
    Software PWM implementation using threading.
    """
    def __init__(self, pin, frequency=50):
        self.pin = pin
        self.period = 1.0 / frequency
        self.duty_cycle = 0
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        # Initialize line request handled externally or in loop logic context
        # In this specific file context, the pin setup seems handled by init() globally,
        # but the SoftPWM logic toggles the value.

    def start(self, duty_cycle):
        self.duty_cycle = duty_cycle
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def ChangeDutyCycle(self, duty_cycle):
        with self.lock:
            self.duty_cycle = max(0, min(100, duty_cycle))

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        # Assuming a method exists to set final state, or handled via global gpiod object
        # Based on dump context:
        self.set_value(0) # INACTIVE

    def _loop(self):
        while self.running:
            dc = self.duty_cycle
            if dc == 0:
                self.set_value(0)
                time.sleep(self.period)
            elif dc == 100:
                self.set_value(1)
                time.sleep(self.period)
            else:
                on_time = self.period * (dc / 100.0)
                off_time = self.period - on_time
                self.set_value(1) # ACTIVE
                time.sleep(on_time)
                self.set_value(0) # INACTIVE
                time.sleep(off_time)
    
    def set_value(self, val):
        # This method abstracts the specific GPIO library call
        # In the context of the main file, this likely writes to the request line
        if chip_request:
            chip_request.set_value(self.pin, Value.ACTIVE if val else Value.INACTIVE)

def init():
    """
    Initializes GPIO lines and Serial connection.
    """
    global chip_request, pwm_r, pwm_l, ser
    
    try:
        chip = gpiod.Chip(CHIP_PATH)
        config = gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE)
        
        # Aggregate all pins
        all_pins = list(R1_PINS) + list(R2_PINS) + list(L1_PINS) + list(L2_PINS) + [ENA_PIN, ENB_PIN]
        
        chip_request = chip.request_lines(consumer="motor_control", config={pin: config for pin in all_pins})
        
        # Initialize PWM wrappers
        pwm_r = SoftPWM(ENA_PIN, frequency=100)
        pwm_l = SoftPWM(ENB_PIN, frequency=100)
        
        pwm_r.start(0)
        pwm_l.start(0)
        
    except Exception as e:
        print(f"Failed to request GPIO lines: {e}")

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
        ser.flush()
        print(f"Connected to HC-05 on {SERIAL_PORT}")
    except Exception as e:
        print(f"Error connecting to HC-05: {e}")

def init_gpio():
    """
    Initialize GPIO only, don't touch serial connection.
    Used for re-initialization after release.
    """
    global chip_request, pwm_r, pwm_l
    
    if chip_request:
        return

    try:
        chip = gpiod.Chip(CHIP_PATH)
        config = gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE)
        all_pins = list(R1_PINS) + list(R2_PINS) + list(L1_PINS) + list(L2_PINS) + [ENA_PIN, ENB_PIN]
        chip_request = chip.request_lines(consumer="motor_control", config={pin: config for pin in all_pins})
        
        pwm_r = SoftPWM(ENA_PIN, frequency=100)
        pwm_l = SoftPWM(ENB_PIN, frequency=100)
        pwm_r.start(0)
        pwm_l.start(0)
        print("[MotorControl] GPIO re-initialized")
        
    except Exception as e:
        print(f"[MotorControl] Failed to request GPIO lines: {e}")

def cleanup():
    global chip_request, pwm_r, pwm_l, ser
    
    stop() # Ensure motors are stopped
    
    if pwm_r: pwm_r.stop()
    if pwm_l: pwm_l.stop()
    
    if chip_request:
        chip_request.release()
        chip_request = None
        
    if ser:
        ser.close()
        ser = None

def release_gpio():
    """
    Release GPIO resources only, keep serial connection alive.
    """
    global chip_request, pwm_r, pwm_l
    
    stop()
    if pwm_r: pwm_r.stop()
    if pwm_l: pwm_l.stop()
    
    if chip_request:
        chip_request.release()
        chip_request = None
    
    print("[MotorControl] GPIO released, serial still active")

def set_motors_gpio(r1_state, r2_state, l1_state, l2_state):
    """
    Low level GPIO handler.
    Takes tuples (True/False, True/False) for each motor.
    """
    if not chip_request:
        return

    def to_val(b):
        return Value.ACTIVE if b else Value.INACTIVE

    # Unpacking tuples to set values on the chip request
    # R1
    chip_request.set_value(R1_PINS[0], to_val(r1_state[0]))
    chip_request.set_value(R1_PINS[1], to_val(r1_state[1]))
    # R2
    chip_request.set_value(R2_PINS[0], to_val(r2_state[0]))
    chip_request.set_value(R2_PINS[1], to_val(r2_state[1]))
    # L1
    chip_request.set_value(L1_PINS[0], to_val(l1_state[0]))
    chip_request.set_value(L1_PINS[1], to_val(l1_state[1]))
    # L2
    chip_request.set_value(L2_PINS[0], to_val(l2_state[0]))
    chip_request.set_value(L2_PINS[1], to_val(l2_state[1]))

def set_pwm_raw(val_l, val_r):
    if pwm_l and pwm_r:
        pwm_l.ChangeDutyCycle(val_l)
        pwm_r.ChangeDutyCycle(val_r)

def set_wheel_states(fl, fr, bl, br, speed_l=None, speed_r=None):
    """
    Sets the direction for individual wheels to support Holonomic drive.
    Inputs: 1 (Forward), -1 (Backward), 0 (Stop)
    """
    # Map direction integers to pin states (A, B)
    # 1 -> (True, False), -1 -> (False, True), 0 -> (False, False)
    
    def get_state(val):
        if val == 1: return (Value.ACTIVE, Value.INACTIVE)
        if val == -1: return (Value.INACTIVE, Value.ACTIVE)
        return (Value.INACTIVE, Value.INACTIVE)

    map_fl = get_state(fl)
    map_fr = get_state(fr)
    map_bl = get_state(bl)
    map_br = get_state(br)
    
    set_motors_gpio(map_fr, map_br, map_fl, map_bl) # Note: Order matches R1, R2, L1, L2 logic
    
    # Use global speed if not provided
    sl = speed_l if speed_l is not None else current_speed
    sr = speed_r if speed_r is not None else current_speed
    
    set_pwm_raw(sl, sr)

def update_speed(val):
    global current_speed
    current_speed = val
    print(f"Speed set to {current_speed}")

# --- Movement Functions ---

def forward():
    set_wheel_states(1, 1, 1, 1)

def backward():
    set_wheel_states(-1, -1, -1, -1)

def stop():
    set_wheel_states(0, 0, 0, 0, 0, 0)

def turn_left():
    set_wheel_states(-1, 1, -1, 1)

def turn_right():
    set_wheel_states(1, -1, 1, -1)

def strafe_left():
    # FL: Back, FR: Fwd, BL: Fwd, BR: Back
    set_wheel_states(-1, 1, 1, -1)

def strafe_right():
    # FL: Fwd, FR: Back, BL: Back, BR: Fwd
    set_wheel_states(1, -1, -1, 1)

def diagonal_front_left():
    set_wheel_states(0, 1, 1, 0)

def diagonal_front_right():
    set_wheel_states(1, 0, 0, 1)

def diagonal_back_left():
    set_wheel_states(-1, 0, 0, -1)

def diagonal_back_right():
    set_wheel_states(0, -1, -1, 0)

def forward_left():
    # Curve
    set_wheel_states(0.5, 1, 0.5, 1) # Note: set_wheel_states logic above expects int 1/-1, 
                                     # but the bytecode implies floating point handling or simplified turn logic.
                                     # Based on standard mecanum, likely reduced speed on one side.
                                     # Actually, the bytecode for forward_left uses `set_wheel_states` 
                                     # likely passing distinct logic or the function handles it.
                                     # Re-reading set_wheel_states: it strictly maps 1/-1. 
                                     # The dump for forward_left calls set_pwm_raw separately or uses set_pwm.
    # Looking at the dump: It calls set_wheel_states(0, 1, 0, 1) or similar? 
    # Actually, the dump `forward_left` calls `set_wheel_states(0, 1, 0, 1)` (Stopped L, Fwd R).
    set_wheel_states(0, 1, 0, 1)

def forward_right():
    set_wheel_states(1, 0, 1, 0)

def backward_left():
    set_wheel_states(0, -1, 0, -1)

def backward_right():
    set_wheel_states(-1, 0, -1, 0)

# --- Pivot Turns ---

def turn_rear_axis_left():
    """
    Pivots the robot around the REAR axle.
    The rear wheels lock (anchor), and the front swings Left.
    """
    set_wheel_states(-1, 1, 0, 0)

def turn_rear_axis_right():
    """
    Pivots the robot around the REAR axle.
    The rear wheels lock (anchor), and the front swings Right.
    """
    set_wheel_states(1, -1, 0, 0)

def turn_front_axis_left():
    """
    Pivots the robot around the FRONT axle.
    The front wheels lock (anchor), and the rear swings Right (turning the nose Left).
    """
    set_wheel_states(0, 0, -1, 1)

def turn_front_axis_right():
    """
    Pivots the robot around the FRONT axle.
    The front wheels lock (anchor), and the rear swings Left (turning the nose Right).
    """
    set_wheel_states(0, 0, 1, -1)

def swing_turn_right():
    """
    Pivots the robot around the RIGHT side wheels.
    Right wheels lock (anchor), Left wheels drive Forward.
    Result: Tight turn to the Right.
    """
    set_wheel_states(1, 0, 1, 0)

def swing_turn_left():
    """
    Pivots the robot around the LEFT side wheels.
    Left wheels lock (anchor), Right wheels drive Forward.
    Result: Tight turn to the Left.
    """
    set_wheel_states(0, 1, 0, 1)

def swing_turn_back_right():
    """
    Reverse Pivot around the RIGHT side wheels.
    Right wheels lock, Left wheels drive Backward.
    """
    set_wheel_states(-1, 0, -1, 0)

def swing_turn_back_left():
    """
    Reverse Pivot around the LEFT side wheels.
    Left wheels lock, Right wheels drive Backward.
    """
    set_wheel_states(0, -1, 0, -1)

def set_pwm(left_duty, right_duty):
    """
    Control motors with signed integers (-100 to 100).
    Compatible with non-holonomic tank drive logic.
    """
    # Determine direction
    l_dir = 1 if left_duty > 0 else (-1 if left_duty < 0 else 0)
    r_dir = 1 if right_duty > 0 else (-1 if right_duty < 0 else 0)
    
    # Get Absolute duty cycle
    abs_l = abs(left_duty)
    abs_r = abs(right_duty)
    
    # Pass to wheel states (Assuming tank drive behavior for all 4 wheels)
    # FL, FR, BL, BR
    set_wheel_states(l_dir, r_dir, l_dir, r_dir, abs_l, abs_r)

if __name__ == "__main__":
    init()
    try:
        print("Moving Forward...")
        forward()
        time.sleep(2)
        print("Stopping...")
        stop()
    finally:
        cleanup()