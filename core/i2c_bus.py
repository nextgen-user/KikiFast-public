"""
core/i2c_bus.py — one process-wide lock for the shared I2C bus (/dev/i2c-1).

The 16x2 char LCD (RPLCD/smbus2 @0x27) and the SSD1306 OLED (busio/adafruit
@0x3c) sit on the SAME physical I2C bus but use DIFFERENT library handles, so
neither knows about the other's transactions. With the OLED render loop writing
continuously, their start/stop conditions interleave and corrupt each other
("[Errno 121] Remote I/O error", frozen/garbled LCD).

Every hardware I2C transaction in this codebase must run under this lock:

    from core.i2c_bus import I2C_LOCK
    with I2C_LOCK:
        ...do the I2C write...
"""

import threading

# A single re-usable lock shared by all I2C device drivers in the process.
I2C_LOCK = threading.Lock()
