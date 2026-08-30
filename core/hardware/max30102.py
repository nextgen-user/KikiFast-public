#!/usr/bin/env python3
"""MAX30102 heart rate + SpO2 over the Raspberry Pi's I2C-1 bus.

    python3 -m core.hardware.max30102            # fingertip (default)
    python3 -m core.hardware.max30102 --wrist    # wrist preset
    python3 -m core.hardware.max30102 --seconds 20 --led 0x40

Reports a signal-quality verdict alongside the numbers and refuses to print a
reading it does not trust. SpO2 is UNCALIBRATED -- see the closing note.
"""
import argparse
import sys
import threading
import time

import numpy as np
from smbus2 import SMBus, i2c_msg

I2C_BUS = 1
ADDR = 0x57

# --- register map -----------------------------------------------------------
INT_ENABLE_1, INT_ENABLE_2 = 0x02, 0x03
FIFO_WR_PTR, OVF_COUNTER, FIFO_RD_PTR, FIFO_DATA = 0x04, 0x05, 0x06, 0x07
FIFO_CONFIG, MODE_CONFIG, SPO2_CONFIG = 0x08, 0x09, 0x0A
LED1_PA, LED2_PA = 0x0C, 0x0D          # LED1 = RED, LED2 = IR
REV_ID_REG, PART_ID_REG = 0xFE, 0xFF
PART_ID_MAX30102 = 0x15

# Sample rate 400 Hz with 4x on-chip averaging -> 100 Hz effective, 18-bit.
FIFO_CFG = (0b010 << 5) | (1 << 4)                    # avg 4, rollover on
SPO2_CFG = (0b11 << 5) | (0b011 << 2) | 0b11          # 16384nA range, 400 Hz, 411us
IR_SATURATED = 240_000                                # 18-bit full scale is 262_143


class Preset:
    """Per-site tuning. The wrist returns far less light than a fingertip."""

    def __init__(self, site, led, contact_ir, clear_ir, ir_low, ir_high,
                 seconds, pi_min, min_peaks):
        self.site = site
        self.led = led                  # starting LED current
        self.contact_ir = contact_ir    # absolute floor for "something is on it"
        self.clear_ir = clear_ir        # below this the sensor is uncovered
        self.ir_low = ir_low            # auto-gain target window
        self.ir_high = ir_high
        self.seconds = seconds
        self.pi_min = pi_min            # perfusion floor for a trusted reading
        self.min_peaks = min_peaks


FINGER = Preset("finger", led=0x1F, contact_ir=15_000, clear_ir=20_000,
                ir_low=85_000, ir_high=200_000, seconds=35.0,
                pi_min=0.20, min_peaks=12)

WRIST = Preset("wrist", led=0x7F, contact_ir=3_000, clear_ir=2_000,
               ir_low=30_000, ir_high=200_000, seconds=40.0,
               pi_min=0.05, min_peaks=15)

SETTLE_S = 2.0
WAIT_S = 45.0


class MAX30102:
    def __init__(self, bus, addr=ADDR):
        self.bus = bus
        self.addr = addr
        self.glitches = 0

    def _retry(self, fn, *args):
        """One bus glitch must not end a 30-second measurement."""
        last = None
        for _ in range(5):
            try:
                return fn(*args)
            except OSError as exc:
                last = exc
                self.glitches += 1
                time.sleep(0.004)
        raise last

    def _read(self, reg):
        return self._retry(self.bus.read_byte_data, self.addr, reg)

    def _write(self, reg, value):
        return self._retry(self.bus.write_byte_data, self.addr, reg, value)

    def identify(self):
        return self._read(PART_ID_REG), self._read(REV_ID_REG)

    def reset(self):
        self._write(MODE_CONFIG, 0x40)
        for _ in range(100):
            if not self._read(MODE_CONFIG) & 0x40:
                return True
            time.sleep(0.01)
        return False

    def configure(self):
        self._write(INT_ENABLE_1, 0x00)
        self._write(INT_ENABLE_2, 0x00)
        self.flush()
        self._write(FIFO_CONFIG, FIFO_CFG)
        self._write(MODE_CONFIG, 0x03)      # SpO2 mode: RED + IR
        self._write(SPO2_CONFIG, SPO2_CFG)

    def set_current(self, value):
        value = max(0x08, min(0xFF, int(value)))
        self._write(LED1_PA, value)
        self._write(LED2_PA, value)
        return value

    def flush(self):
        self._write(FIFO_WR_PTR, 0)
        self._write(OVF_COUNTER, 0)
        self._write(FIFO_RD_PTR, 0)

    def read_fifo(self):
        """Return (red[], ir[]) for whatever samples are pending."""
        write_ptr, read_ptr = self._read(FIFO_WR_PTR), self._read(FIFO_RD_PTR)
        pending = (write_ptr - read_ptr) & 0x1F
        if pending == 0:
            if self._read(OVF_COUNTER) == 0:
                return [], []
            pending = 32

        raw = None
        for _ in range(5):
            try:
                tx = i2c_msg.write(self.addr, [FIFO_DATA])
                rx = i2c_msg.read(self.addr, pending * 6)
                self.bus.i2c_rdwr(tx, rx)
                raw = list(rx)
                break
            except OSError:
                self.glitches += 1
                time.sleep(0.004)
        if raw is None:
            return [], []

        red, ir = [], []
        for i in range(0, len(raw) - 5, 6):
            red.append((raw[i] << 16 | raw[i + 1] << 8 | raw[i + 2]) & 0x03FFFF)
            ir.append((raw[i + 3] << 16 | raw[i + 4] << 8 | raw[i + 5]) & 0x03FFFF)
        return red, ir


# --- signal processing ------------------------------------------------------

def moving_average(x, width):
    if width < 2:
        return x.copy()
    return np.convolve(x, np.ones(width) / width, mode="same")


def bandpass(x, fs):
    """~0.7-8 Hz: smooth sensor noise, then subtract the slow baseline."""
    smoothed = moving_average(x, max(2, int(round(fs * 0.10))))
    baseline = moving_average(smoothed, max(3, int(round(fs * 1.20))))
    return smoothed - baseline


def find_peaks(x, fs):
    min_gap = int(fs * 60.0 / 200.0)          # cap at 200 bpm
    threshold = 0.35 * np.std(x)
    peaks = []
    for i in range(1, len(x) - 1):
        if x[i] > threshold and x[i] >= x[i - 1] and x[i] > x[i + 1]:
            if peaks and i - peaks[-1] < min_gap:
                if x[i] > x[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    return np.array(peaks)


def analyse(red, ir, fs):
    red = np.asarray(red, dtype=float)
    ir = np.asarray(ir, dtype=float)
    out = {"dc_ir": ir.mean(), "dc_red": red.mean()}

    ac_ir = bandpass(ir, fs)
    ac_red = bandpass(red, fs)
    edge = int(fs * 0.7)                       # trim convolution artefacts
    ac_ir, ac_red = ac_ir[edge:-edge], ac_red[edge:-edge]

    peaks = find_peaks(ac_ir, fs)
    out["n_peaks"] = len(peaks)
    if len(peaks) >= 3:
        rr = np.diff(peaks) / fs
        median = np.median(rr)
        rr = rr[(rr > 0.5 * median) & (rr < 1.8 * median)]
        if len(rr):
            out["bpm"] = 60.0 / np.median(rr)
            out["rr_cv"] = float(np.std(rr) / np.mean(rr))
            out["sdnn_ms"] = float(np.std(rr) * 1000.0)

    # Split into thirds and compare -- a real pulse holds its rate across the
    # capture, noise does not. Only meaningful on a long window.
    if len(peaks) >= 9:
        third = len(ac_ir) // 3
        rates = []
        for seg in range(3):
            seg_peaks = peaks[(peaks >= seg * third) & (peaks < (seg + 1) * third)]
            if len(seg_peaks) >= 3:
                seg_rr = np.diff(seg_peaks) / fs
                rates.append(60.0 / np.median(seg_rr))
        if len(rates) == 3:
            out["bpm_thirds"] = rates
            out["bpm_spread"] = float(max(rates) - min(rates))

    rms_ir = float(np.sqrt(np.mean(ac_ir ** 2)))
    rms_red = float(np.sqrt(np.mean(ac_red ** 2)))
    out["pi"] = (rms_ir / out["dc_ir"] * 100.0) if out["dc_ir"] else 0.0

    if out["dc_ir"] and out["dc_red"] and rms_ir:
        ratio = (rms_red / out["dc_red"]) / (rms_ir / out["dc_ir"])
        out["R"] = ratio
        # Maxim's empirical curve. Uncalibrated for this optical assembly, and
        # it pins at ~100% for R below about 0.5, where it carries no meaning.
        out["spo2"] = -45.060 * ratio * ratio + 30.354 * ratio + 94.845
    return out


def verdict(a, preset):
    reasons = []
    if a.get("dc_ir", 0) < preset.contact_ir:
        reasons.append("no skin contact detected")
    if a.get("n_peaks", 0) < preset.min_peaks:
        reasons.append(f"too few clean beats ({a.get('n_peaks', 0)})")
    if a.get("pi", 0) < preset.pi_min:
        reasons.append(f"perfusion too low ({a.get('pi', 0):.2f}%)")
    if a.get("rr_cv", 1.0) > 0.25:
        reasons.append(f"beat timing erratic (CV {a.get('rr_cv', 0):.2f})")
    if reasons:
        return "POOR", reasons
    if a.get("pi", 0) > preset.pi_min * 3 and a.get("rr_cv", 1.0) < 0.12:
        return "GOOD", []
    return "FAIR", []


# --- acquisition helpers ----------------------------------------------------

def pump(sensor, seconds, red_sink=None, ir_sink=None, cancel_event=None):
    """Drain the FIFO for `seconds`; return the mean IR over that window."""
    deadline, seen = time.time() + seconds, []
    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            break
        red, ir = sensor.read_fifo()
        if ir:
            seen.extend(ir)
            if ir_sink is not None:
                ir_sink.extend(ir)
                red_sink.extend(red)
        time.sleep(0.02)
    return float(np.mean(seen)) if seen else 0.0


def wait_until(sensor, predicate, timeout, label, cancel_event=None,
               progress=None, quiet=False):
    """Poll until predicate(mean_ir) holds for 0.5 s. Returns (ok, last_ir)."""
    start, held, last_print, level = time.time(), 0.0, 0.0, 0.0
    while time.time() - start < timeout:
        if cancel_event is not None and cancel_event.is_set():
            return False, level
        _, ir = sensor.read_fifo()
        if ir:
            level = float(np.mean(ir))
            elapsed = time.time() - start
            if elapsed - last_print >= 1.0:
                last_print = elapsed
                flag = "SAT" if level >= IR_SATURATED else ("ok" if predicate(level) else "..")
                if not quiet:
                    print(f"  {elapsed:4.1f}s  IR {level:8.0f}  {flag}   [{label}]", flush=True)
                if progress is not None:
                    progress({"phase": label, "elapsed_seconds": round(elapsed, 1),
                              "ir_level": round(level), "signal": flag})
            held = held + 0.05 if predicate(level) else 0.0
            if held >= 0.5:
                return True, level
        time.sleep(0.05)
    return False, level


def preset_for(site="finger", seconds=None):
    """Return an independent preset; library callers must never mutate globals."""
    source = WRIST if str(site).strip().lower() == "wrist" else FINGER
    preset = Preset(
        source.site, source.led, source.contact_ir, source.clear_ir,
        source.ir_low, source.ir_high, source.seconds, source.pi_min,
        source.min_peaks)
    if seconds is not None:
        preset.seconds = max(10.0, min(60.0, float(seconds)))
    return preset


def _signal_payload(analysis, sensor, current):
    keys = ("pi", "n_peaks", "rr_cv", "sdnn_ms", "bpm_spread",
            "dc_ir", "dc_red")
    signal = {}
    for key in keys:
        if key in analysis:
            value = analysis[key]
            signal[key] = round(float(value), 4) if isinstance(
                value, (float, np.floating)) else int(value)
    signal["i2c_glitches"] = int(sensor.glitches)
    signal["led_current"] = int(current)
    return signal


def prepare_heart_rate(site="finger", clear_timeout=12.0, progress=None,
                       cancel_event=None, bus_factory=SMBus):
    """Quality-gated clear-sensor phase for voice/agent orchestration.

    Returns JSON-serializable state. It never speaks and never prints procedural
    guidance; the voice agent owns that interaction.
    """
    preset = preset_for(site)
    cancel_event = cancel_event or threading.Event()
    try:
        with bus_factory(I2C_BUS) as bus:
            sensor = MAX30102(bus)
            part, rev = sensor.identify()
            if part != PART_ID_MAX30102:
                return {"status": "hardware_error", "reason": "unexpected_sensor",
                        "part_id": part, "expected_part_id": PART_ID_MAX30102}
            if not sensor.reset():
                return {"status": "hardware_error", "reason": "reset_failed"}
            sensor.configure()
            time.sleep(0.2)
            sensor.flush()
            sensor.set_current(preset.led)
            ok, level = wait_until(
                sensor, lambda value: value < preset.clear_ir,
                float(clear_timeout), "checking_clear_sensor",
                cancel_event=cancel_event, progress=progress, quiet=True)
            if cancel_event.is_set():
                return {"status": "cancelled"}
            if not ok:
                return {"status": "retryable_failure",
                        "reason": "sensor_not_clear_or_ambient_light",
                        "ir_level": round(level)}
            baseline = pump(sensor, 1.5, cancel_event=cancel_event)
            if cancel_event.is_set():
                return {"status": "cancelled"}
            gate = max(preset.contact_ir, min(baseline * 2.0, 40_000))
            return {
                "status": "ready_for_contact",
                "site": preset.site,
                "ambient_baseline": round(baseline),
                "contact_gate": round(gate),
                "part_id": part,
                "revision_id": rev,
            }
    except (OSError, IOError) as exc:
        return {"status": "hardware_error", "reason": "i2c_error",
                "detail": str(exc)[:200]}
    except Exception as exc:
        return {"status": "hardware_error", "reason": "unexpected_error",
                "detail": str(exc)[:200]}


def capture_heart_rate(preparation, seconds=None, contact_timeout=15.0,
                       progress=None, cancel_event=None, bus_factory=SMBus):
    """Detect contact, auto-gain, capture and return a trusted BPM or rejection."""
    preparation = preparation if isinstance(preparation, dict) else {}
    if preparation.get("status") != "ready_for_contact":
        return {"status": "not_prepared", "reason": "run_prepare_first"}
    preset = preset_for(preparation.get("site", "finger"), seconds)
    cancel_event = cancel_event or threading.Event()
    gate = float(preparation.get("contact_gate") or preset.contact_ir)
    try:
        with bus_factory(I2C_BUS) as bus:
            sensor = MAX30102(bus)
            part, _rev = sensor.identify()
            if part != PART_ID_MAX30102:
                return {"status": "hardware_error", "reason": "unexpected_sensor",
                        "part_id": part, "expected_part_id": PART_ID_MAX30102}
            if not sensor.reset():
                return {"status": "hardware_error", "reason": "reset_failed"}
            sensor.configure()
            time.sleep(0.2)
            sensor.flush()
            current = sensor.set_current(preset.led)

            ok, level = wait_until(
                sensor, lambda value: value > gate, float(contact_timeout),
                "checking_skin_contact", cancel_event=cancel_event,
                progress=progress, quiet=True)
            if cancel_event.is_set():
                return {"status": "cancelled"}
            if not ok:
                return {"status": "retryable_failure", "reason": "no_contact",
                        "site": preset.site, "ir_level": round(level)}

            for _ in range(8):
                level = pump(sensor, 0.7, cancel_event=cancel_event)
                if cancel_event.is_set():
                    return {"status": "cancelled"}
                if level >= IR_SATURATED:
                    nxt = max(0x08, int(current * 0.55))
                elif level > preset.ir_high:
                    nxt = max(0x08, int(current * 0.75))
                elif level < preset.ir_low:
                    nxt = min(0xFF, int(current * 1.7) + 4)
                else:
                    break
                if nxt == current:
                    break
                current = sensor.set_current(nxt)
                time.sleep(0.15)

            pump(sensor, SETTLE_S, cancel_event=cancel_event)
            sensor.flush()
            red_all, ir_all = [], []
            start, last_progress = time.time(), -5.0
            while time.time() - start < preset.seconds:
                if cancel_event.is_set():
                    return {"status": "cancelled"}
                red, ir = sensor.read_fifo()
                if ir:
                    red_all.extend(red)
                    ir_all.extend(ir)
                elapsed = time.time() - start
                if progress is not None and elapsed - last_progress >= 3.0:
                    last_progress = elapsed
                    progress({"phase": "capturing", "elapsed_seconds": round(elapsed, 1),
                              "duration_seconds": preset.seconds,
                              "samples": len(ir_all)})
                time.sleep(0.02)

            count = len(ir_all)
            if count < 300:
                return {"status": "retryable_failure",
                        "reason": "too_few_samples", "samples": count}
            analysis = analyse(red_all, ir_all, count / preset.seconds)
            quality, reasons = verdict(analysis, preset)
            signal = _signal_payload(analysis, sensor, current)
            if quality == "POOR" or "bpm" not in analysis:
                return {"status": "retryable_poor_signal", "quality": quality,
                        "site": preset.site, "reasons": reasons, "signal": signal}
            return {
                "status": "trusted_reading",
                "measurement": "heart_rate",
                "bpm": int(round(float(analysis["bpm"]))),
                "unit": "bpm",
                "quality": quality,
                "site": preset.site,
                "duration_seconds": preset.seconds,
                "signal": signal,
            }
    except (OSError, IOError) as exc:
        return {"status": "hardware_error", "reason": "i2c_error",
                "detail": str(exc)[:200]}
    except Exception as exc:
        return {"status": "hardware_error", "reason": "unexpected_error",
                "detail": str(exc)[:200]}


def main():
    ap = argparse.ArgumentParser(description="MAX30102 heart rate / SpO2 reader")
    ap.add_argument("--wrist", action="store_true",
                    help="wrist preset: stronger LEDs, lower gates, longer capture")
    ap.add_argument("--seconds", type=float, help="override measurement duration")
    ap.add_argument("--led", type=lambda v: int(v, 0), help="starting LED current, e.g. 0x40")
    args = ap.parse_args()

    preset = preset_for("wrist" if args.wrist else "finger", args.seconds)
    if args.led:
        preset.led = args.led

    print(f"=== site: {preset.site.upper()} ===")
    with SMBus(I2C_BUS) as bus:
        sensor = MAX30102(bus)
        part, rev = sensor.identify()
        print(f"part id 0x{part:02X}  rev 0x{rev:02X}", flush=True)
        if part != PART_ID_MAX30102:
            print(f"!! expected 0x{PART_ID_MAX30102:02X} (MAX30102), got 0x{part:02X}")
            return 1
        if not sensor.reset():
            print("!! soft reset did not clear")
            return 1

        sensor.configure()
        time.sleep(0.2)
        sensor.flush()
        current = sensor.set_current(preset.led)
        print(f"ADC range 16384nA | LED 0x{current:02X} (~{current * 0.2:.1f} mA)")

        # 1. establish an ambient baseline with nothing on the sensor
        print(f"\n[1/4] TAKE THE SENSOR OFF your skin.", flush=True)
        ok, _ = wait_until(sensor, lambda v: v < preset.clear_ir, 25.0, "want clear")
        if not ok:
            print("\nStill reading high with nothing on it -- strong ambient light?")
            print("Shade the sensor or move away from a lamp, then rerun.")
            return 2
        baseline = pump(sensor, 1.5)
        gate = max(preset.contact_ir, min(baseline * 2.0, 40_000))
        print(f"  ambient baseline {baseline:.0f}   contact gate {gate:.0f}", flush=True)

        # 2. wait for skin contact
        press = ("Press it firmly onto your wrist" if preset.site == "wrist"
                 else "Rest a fingertip on it")
        print(f"\n[2/4] NOW {press}, covering both LEDs.", flush=True)
        ok, level = wait_until(sensor, lambda v: v > gate, WAIT_S, f"want {preset.site}")
        if not ok:
            print(f"\nNo contact held above {gate:.0f}.")
            return 3
        print("  contact detected", flush=True)

        # 3. auto-gain, correcting in both directions
        print(f"\n[3/4] Auto-gain (target IR {preset.ir_low:,}-{preset.ir_high:,})", flush=True)
        for _ in range(8):
            level = pump(sensor, 0.7)
            if level >= IR_SATURATED:
                nxt = max(0x08, int(current * 0.55))
            elif level > preset.ir_high:
                nxt = max(0x08, int(current * 0.75))
            elif level < preset.ir_low:
                nxt = min(0xFF, int(current * 1.7) + 4)
            else:
                break
            if nxt == current:
                break
            current = sensor.set_current(nxt)
            print(f"  IR {level:8.0f} -> LED 0x{current:02X} (~{current * 0.2:.1f} mA)", flush=True)
            time.sleep(0.15)
        print(f"  settled: LED 0x{current:02X}, IR ~{level:.0f}", flush=True)
        if level >= IR_SATURATED:
            print("  !! still saturated at minimum current -- press much more lightly")

        pump(sensor, SETTLE_S)
        sensor.flush()

        # 4. measure
        print(f"\n[4/4] Measuring {preset.seconds:.0f}s -- hold completely still.\n", flush=True)
        red_all, ir_all = [], []
        start, last_print = time.time(), 0.0
        while time.time() - start < preset.seconds:
            red, ir = sensor.read_fifo()
            if ir:
                red_all.extend(red)
                ir_all.extend(ir)
            elapsed = time.time() - start
            if elapsed - last_print >= 3.0:
                last_print = elapsed
                print(f"  {elapsed:4.1f}s   {len(ir_all):5d} samples   "
                      f"IR {np.mean(ir_all[-50:]):8.0f}", flush=True)
            time.sleep(0.02)

        count = len(ir_all)
        fs = count / preset.seconds
        print(f"\ncollected {count} samples ({fs:.1f} Hz)   i2c glitches: {sensor.glitches}")
        if count < 300:
            print("!! too few samples -- FIFO not keeping up")
            return 4

        a = analyse(red_all, ir_all, fs)
        quality, reasons = verdict(a, preset)
        trusted = quality != "POOR"

        print("\n" + "=" * 46)
        if "bpm" in a:
            tag = "" if trusted else "   << LOW CONFIDENCE"
            print(f"  HEART RATE     {a['bpm']:.0f} bpm{tag}")
        else:
            print("  HEART RATE     -- (no beats detected)")
        if "spo2" in a:
            tag = "(uncalibrated)" if trusted else "(uncalibrated, LOW CONFIDENCE)"
            print(f"  SpO2           {min(a['spo2'], 100):.0f} %   {tag}")
        else:
            print("  SpO2           -- (no signal)")
        print("-" * 46)
        print(f"  signal quality {quality}")
        for reason in reasons:
            print(f"                 - {reason}")
        print(f"  perfusion idx  {a.get('pi', 0):.2f} %")
        print(f"  beats found    {a.get('n_peaks', 0)}")
        if "rr_cv" in a:
            print(f"  RR variability CV {a['rr_cv']:.3f}   SDNN {a.get('sdnn_ms', 0):.0f} ms")
        if "bpm_thirds" in a:
            t = a["bpm_thirds"]
            print(f"  rate over time {t[0]:.0f} / {t[1]:.0f} / {t[2]:.0f} bpm"
                  f"   (spread {a['bpm_spread']:.0f})")
        if "R" in a:
            print(f"  ratio R        {a['R']:.3f}")
        print(f"  DC  IR {a['dc_ir']:.0f}   RED {a['dc_red']:.0f}   LED 0x{current:02X}")
        print("=" * 46)
        print("\nNote: SpO2 uses Maxim's generic R-curve and is NOT calibrated for")
        print("this optical assembly. Below R~0.5 the curve pins at 100% and")
        print("carries no information. Treat it as a trend, never a clinical value.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
