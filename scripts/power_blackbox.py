#!/usr/bin/env python3
"""Flight recorder for the Pi's power and thermal state.

Answers one question: when the box dies, was it a brownout, an overheat, or a
software shutdown?  Each is a different signature in this log.

  brownout        the log stops mid-sample with no STOP marker, samples are
                  evenly spaced right up to the end, and EXT5V is usually
                  already sagging or the under-voltage bit is set
  watchdog reset  also stops with no STOP marker, but the samples fall behind
                  first (PID 1 has to stall past the 60s hardware timeout for
                  it to fire) and the NEXT boot reports a different rsts /
                  wd_bootstatus than the baseline recorded here
  overheat        temp_c climbs past the soft limit and the throttle bits set
                  long before the end; a thermal poweroff is still *graceful*,
                  so a STOP marker is written
  software / OOM  a clean STOP marker with normal voltage and temperature

The STOP marker is the whole trick: systemd sends SIGTERM on any orderly
shutdown, so its absence means either the power died or the watchdog reset the
board.  Every sample is fsynced for the same reason -- the last line before a
hard cut is the evidence, and buffered output would lose exactly that line.

Brownout and watchdog reset are otherwise identical from userspace, so each
BOOT marker also records the reset-cause registers.  Comparing them against a
boot known to be clean is what separates the two.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

DEFAULT_LOG = "/var/log/kiki-power-blackbox.csv"
FIELDS = ("time", "uptime_s", "ext5v", "vdd_core_v", "rails_w",
          "temp_c", "throttled", "flags", "load1", "mem_avail_mb", "late_s")

# Raspberry Pi throttle bits: 0-3 are live, 16-19 are sticky "has happened
# since boot".  The sticky ones survive the event itself, which a 1 Hz sampler
# would otherwise miss entirely.
_BITS = {
    0: "undervolt-now", 1: "freq-capped-now", 2: "throttled-now",
    3: "soft-temp-limit-now", 16: "undervolt-seen", 17: "freq-capped-seen",
    18: "throttled-seen", 19: "soft-temp-limit-seen",
}


def _run(command: list[str], timeout: float = 5.0) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout, check=False)
        return result.stdout if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def decode_throttled(raw: str) -> tuple[str, str]:
    """Return (hex value, human-readable flag list) for get_throttled output."""
    value = raw.strip().split("=")[-1].strip()
    try:
        bits = int(value, 16)
    except ValueError:
        return ("?", "")
    names = [name for bit, name in sorted(_BITS.items()) if bits & (1 << bit)]
    return (value or "?", "|".join(names))


def read_temp() -> float:
    """SoC temperature straight from sysfs -- no fork, unlike measure_temp."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as handle:
            return int(handle.read().strip()) / 1000.0
    except (OSError, ValueError):
        return -1.0


def read_pmic() -> tuple[float, float, float]:
    """Return (EXT5V volts, VDD_CORE volts, summed rail watts).

    EXT5V is the board's own measurement of what the power supply is actually
    delivering, which is the number a brownout shows up in first.
    """
    text = _run(["vcgencmd", "pmic_read_adc"])
    amps: dict[str, float] = {}
    volts: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2 or "=" not in parts[1]:
            continue
        name, reading = parts[0], parts[1]
        try:
            value = float(reading.split("=")[1].rstrip("AV"))
        except ValueError:
            continue
        if name.endswith("_A"):
            amps[name[:-2]] = value
        elif name.endswith("_V"):
            volts[name[:-2]] = value
    watts = sum(amps[rail] * volts[rail] for rail in amps if rail in volts)
    return (volts.get("EXT5V", -1.0), volts.get("VDD_CORE", -1.0), watts)


def read_uptime() -> float:
    try:
        with open("/proc/uptime") as handle:
            return float(handle.read().split()[0])
    except (OSError, ValueError):
        return -1.0


def read_load1() -> float:
    try:
        return os.getloadavg()[0]
    except OSError:
        return -1.0


def read_mem_available_mb() -> float:
    """MemAvailable, the number that actually predicts an OOM or swap stall."""
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return -1.0


def reset_cause() -> str:
    """Registers that say why the board last came up.

    Neither value is worth decoding by hand -- the Pi's RSTS layout differs by
    model and the bcm2835 driver only surfaces one bit of it. What matters is
    that a boot following a watchdog reset reads back differently from a boot
    following a clean start, so both are recorded verbatim for comparison.
    """
    rsts = _run(["vcgencmd", "get_rsts"]).strip().split("=")[-1].strip() or "?"
    try:
        with open("/sys/class/watchdog/watchdog0/bootstatus") as handle:
            bootstatus = handle.read().strip()
    except OSError:
        bootstatus = "?"
    return f"rsts={rsts} wd_bootstatus={bootstatus}"


def boot_id() -> str:
    try:
        with open("/proc/sys/kernel/random/boot_id") as handle:
            return handle.read().strip()
    except OSError:
        return "unknown"


class Recorder:
    def __init__(self, path: str, interval: float, max_bytes: int):
        self.path = path
        self.interval = interval
        self.max_bytes = max_bytes
        self.handle = None
        self.running = True

    def _open(self) -> None:
        new_file = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        self.handle = open(self.path, "a", encoding="utf-8")
        if new_file:
            self._write("#" + ",".join(FIELDS))

    def _write(self, line: str) -> None:
        if self.handle is None:
            return
        self.handle.write(line + "\n")
        # Flush *and* fsync: a brownout gives no warning, and an unflushed
        # final line is precisely the sample that would have identified it.
        self.handle.flush()
        try:
            os.fsync(self.handle.fileno())
        except OSError:
            pass

    def _rotate_if_needed(self) -> None:
        try:
            if os.path.getsize(self.path) < self.max_bytes:
                return
        except OSError:
            return
        if self.handle is not None:
            self.handle.close()
        try:
            os.replace(self.path, self.path + ".1")
        except OSError:
            pass
        self._open()

    def _marker(self, kind: str, detail: str = "") -> None:
        stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        self._write(f"#{kind} {stamp} boot={boot_id()} uptime={read_uptime():.1f}"
                    + (f" {detail}" if detail else ""))

    def stop(self, signum, _frame) -> None:
        # systemd sends SIGTERM for every orderly shutdown, reboot included.
        # Reaching here at all rules out a power cut.
        self.running = False
        self._marker("STOP", f"signal={signal.Signals(signum).name}")

    def run(self) -> int:
        self._open()
        self._marker("BOOT", reset_cause())
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        next_sample = time.monotonic()
        while self.running:
            # How far behind schedule this sample is. A healthy box holds ~0;
            # a box whose PID 1 is stalling towards the watchdog timeout does
            # not, and that divergence is what tells a reset from a brownout.
            late = max(0.0, time.monotonic() - next_sample)
            ext5v, core_v, watts = read_pmic()
            throttled, flags = decode_throttled(_run(["vcgencmd", "get_throttled"]))
            self._write(",".join((
                datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                f"{read_uptime():.1f}", f"{ext5v:.4f}", f"{core_v:.4f}",
                f"{watts:.2f}", f"{read_temp():.1f}", throttled, flags,
                f"{read_load1():.2f}", f"{read_mem_available_mb():.0f}",
                f"{late:.2f}",
            )))
            self._rotate_if_needed()
            next_sample += self.interval
            delay = next_sample - time.monotonic()
            if delay < 0:                       # sampling fell behind; resync
                next_sample = time.monotonic()
                delay = 0
            # A plain sleep would ignore SIGTERM until it expires on some libcs;
            # short hops keep the STOP marker prompt.
            while delay > 0 and self.running:
                time.sleep(min(0.25, delay))
                delay -= 0.25
        if self.handle is not None:
            self.handle.close()
        return 0


def _sessions(path: str) -> list[dict]:
    """Split the log into boot sessions and record how each one ended."""
    sessions: list[dict] = []
    current = None
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#BOOT"):
            current = {"boot": line, "rows": [], "end": "CUT", "stop": ""}
            sessions.append(current)
            continue
        if line.startswith("#STOP"):
            if current is not None:
                current["end"] = "CLEAN"
                current["stop"] = line
            continue
        if line.startswith("#"):
            continue
        if current is not None:
            current["rows"].append(line.split(","))
    return sessions


def report(path: str, tail_seconds: float) -> int:
    if not os.path.exists(path):
        print(f"No blackbox log yet at {path}")
        return 1
    sessions = _sessions(path)
    if not sessions:
        print(f"{path} has no sessions recorded yet.")
        return 1

    # A session that ended CLEAN proves the board was not reset, so the reset
    # registers it booted with are the known-good baseline for every CUT.
    baseline = ""
    for index, session in enumerate(sessions[:-1]):
        if session["end"] == "CLEAN" and index + 1 < len(sessions):
            baseline = _reset_cause_of(sessions[index + 1]["boot"])
            break

    print(f"{len(sessions)} session(s) in {path}")
    if baseline:
        print(f"clean-boot baseline: {baseline}")
    print()

    for index, session in enumerate(sessions):
        rows = session["rows"]
        print(f"--- session {index + 1}/{len(sessions)} ---")
        print(f"  start : {session['boot'][5:].strip()}")
        if not rows:
            print("  (no samples)\n")
            continue
        last_uptime = float(rows[-1][1])
        window = [r for r in rows if last_uptime - float(r[1]) <= tail_seconds]
        volts = [float(r[2]) for r in window if float(r[2]) > 0]
        temps = [float(r[5]) for r in window if float(r[5]) > 0]
        watts = [float(r[4]) for r in window if float(r[4]) > 0]
        mems = [float(r[9]) for r in window if len(r) > 9 and float(r[9]) >= 0]
        lates = [float(r[10]) for r in window if len(r) > 10]
        flags = sorted({f for r in window for f in (r[7] or "").split("|") if f})

        print(f"  ran   : {last_uptime:.0f}s, {len(rows)} samples, "
              f"last at {rows[-1][0]}")
        if volts:
            print(f"  EXT5V : min {min(volts):.3f} V  max {max(volts):.3f} V "
                  f"(last {tail_seconds:.0f}s)")
        if temps:
            print(f"  temp  : max {max(temps):.1f} C")
        if watts:
            print(f"  rails : max {max(watts):.2f} W")
        if mems:
            print(f"  memfree: min {min(mems):.0f} MB")
        if lates:
            print(f"  lag   : max {max(lates):.2f} s behind schedule")
        print(f"  flags : {', '.join(flags) if flags else 'none'}")

        undervolt = any(f.startswith("undervolt") for f in flags)
        sagged = bool(volts) and min(volts) < 4.75
        hot = bool(temps) and max(temps) >= 80
        stalled = bool(lates) and max(lates) >= 5.0
        starved = bool(mems) and min(mems) < 200

        # The session we are running inside has not ended at all; without this
        # every report on a healthy box would accuse it of a brownout.
        if (index == len(sessions) - 1 and session["end"] != "CLEAN"
                and f"boot={boot_id()}" in session["boot"]
                and abs(read_uptime() - last_uptime) < 30.0):
            print("  ended : still running -- this is the live session")
            print()
            continue

        if session["end"] == "CLEAN":
            print(f"  ended : CLEAN -- {session['stop'][5:].strip()}")
            if hot:
                verdict = ("THERMAL -- shutdown was orderly but the SoC was in "
                           "the throttle range")
            else:
                verdict = ("SOFTWARE -- an orderly shutdown at normal voltage "
                           "and temperature; not a power problem")
            print(f"  VERDICT: {verdict}")
            print()
            continue

        print("  ended : CUT -- no shutdown signal was ever delivered")
        after = (_reset_cause_of(sessions[index + 1]["boot"])
                 if index + 1 < len(sessions) else "")
        if after:
            match = "same as clean baseline" if after == baseline else "DIFFERS"
            print(f"  next boot reset cause: {after}" +
                  (f"  [{match}]" if baseline else ""))

        if stalled:
            evidence = [f"sampling fell {max(lates):.1f}s behind"]
            if starved:
                evidence.append(f"MemAvailable down to {min(mems):.0f} MB")
            verdict = ("WATCHDOG RESET -- the board stalled before it died ("
                       + "; ".join(evidence) + "). The 60s hardware watchdog "
                       "resets on a PID 1 stall; this is a software hang, not "
                       "a power fault.")
        elif sagged or undervolt:
            detail = (f"EXT5V sagged to {min(volts):.3f} V"
                      if sagged else "the under-voltage flag was set")
            verdict = (f"BROWNOUT -- {detail} with no shutdown. The supply or "
                       "cable cannot hold 5 V under load.")
        else:
            verdict = ("BROWNOUT (most likely) -- instant loss with no stall "
                       "and no shutdown. Voltage looked fine at this sample "
                       "rate, but a collapse faster than one sample still "
                       "reads exactly like this. Compare the reset cause "
                       "above: unchanged means power, changed means watchdog.")
        print(f"  VERDICT: {verdict}")
        print()
    return 0


def _reset_cause_of(boot_marker: str) -> str:
    """Pull the 'rsts=... wd_bootstatus=...' tail out of a #BOOT line."""
    position = boot_marker.find("rsts=")
    return boot_marker[position:].strip() if position >= 0 else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--interval", type=float, default=1.0,
                        help="seconds between samples (default 1.0)")
    parser.add_argument("--max-bytes", type=int, default=16 * 1024 * 1024,
                        help="rotate the log past this size (default 16 MiB)")
    parser.add_argument("--report", action="store_true",
                        help="summarize the recorded sessions and exit")
    parser.add_argument("--tail-seconds", type=float, default=120.0,
                        help="window before each end to summarize (default 120)")
    args = parser.parse_args()
    if args.report:
        return report(args.log, args.tail_seconds)
    return Recorder(args.log, max(0.1, args.interval), args.max_bytes).run()


if __name__ == "__main__":
    sys.exit(main())
