"""Weather and air-quality ingestion for Kiki's health-companion modes.

This is the half of the SIH statement that had no code behind it at all: heat
waves, pollution events, and early warning. Everything else in the care stack
describes what the person does; this describes what the air and the heat are
doing to them.

Design constraints, in the order they mattered:

  1. **It must never touch the speaking path.** One daemon thread polls; every
     reader gets an immutable copy of the last good result. A network stall can
     therefore cost a poll interval and nothing else -- never a voice turn.
     (`search_web` already taught this lesson the expensive way: an unbounded
     call inside a turn stalled to main.py's ceiling.)
  2. **It must fail silent and stale, never wrong.** A reading that could not be
     refreshed keeps its timestamp and ages out through fresh -> stale ->
     unavailable. Kiki saying "I don't have today's air quality" is fine. Kiki
     reciting yesterday's AQI as though it were now is the exact class of false
     claim the plan forbids.
  3. **The numbers must be the ones Indians actually use.** Open-Meteo returns
     US and European AQI; neither is what a Delhi advisory, a news bulletin, or
     a doctor means by "AQI". The CPCB sub-index is computed here instead --
     see `cpcb_aqi` for the honesty caveat that comes with it.

No API key: both Open-Meteo endpoints are free and unauthenticated, which also
means this subsystem costs nothing against `cloud_budget`.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Any, Dict, Optional, Tuple

try:
    import requests
except Exception:                                    # pragma: no cover
    requests = None

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

_FORECAST_FIELDS = "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code"
_AIR_FIELDS = "pm2_5,pm10,us_aqi"

# Freshness lifecycle (plan.md Phase G). A reading is quotable while fresh,
# quotable-with-an-age while stale, and simply gone after that.
DEFAULT_POLL_SECONDS = 900          # 15 min; both feeds update at most hourly
DEFAULT_STALE_SECONDS = 1800        # 30 min
DEFAULT_UNAVAILABLE_SECONDS = 7200  # 2 h

FRESH, STALE, UNAVAILABLE = "fresh", "stale", "unavailable"


# --- CPCB AQI -----------------------------------------------------------

# India's national AQI (CPCB, 2014). Breakpoints are (C_low, C_high, I_low,
# I_high) in ug/m3 -> index. Published as integer bands (0-30, 31-60, ...);
# used here as CONTINUOUS intervals (0-30, 30-60, ...) so a concentration of
# 120.6 lands in a band at all rather than falling down the gap between 120 and
# 121 and being silently dropped.
_CPCB_PM25 = (
    (0.0, 30.0, 0, 50), (30.0, 60.0, 51, 100), (60.0, 90.0, 101, 200),
    (90.0, 120.0, 201, 300), (120.0, 250.0, 301, 400), (250.0, 380.0, 401, 500),
)
_CPCB_PM10 = (
    (0.0, 50.0, 0, 50), (50.0, 100.0, 51, 100), (100.0, 250.0, 101, 200),
    (250.0, 350.0, 201, 300), (350.0, 430.0, 301, 400), (430.0, 510.0, 401, 500),
)

_CPCB_CATEGORIES = (
    (50, "good"), (100, "satisfactory"), (200, "moderate"),
    (300, "poor"), (400, "very poor"),
)


def _sub_index(value: Optional[float], table) -> Optional[int]:
    """CPCB sub-index for one pollutant, or None when it cannot be computed."""
    if value is None:
        return None
    try:
        concentration = float(value)
    except (TypeError, ValueError):
        return None
    if concentration < 0:
        return None
    for c_low, c_high, i_low, i_high in table:
        if concentration <= c_high:
            span = c_high - c_low
            if span <= 0:
                return i_low
            return round(i_low + (i_high - i_low) * (concentration - c_low) / span)
    # Above the published scale. CPCB itself stops at 500; reporting the ceiling
    # is honest ("as bad as the scale goes") where extrapolating would invent a
    # number no Indian source would ever print.
    return 500


def cpcb_aqi(pm25: Optional[float], pm10: Optional[float]) -> Tuple[Optional[int], Optional[str], str]:
    """India's CPCB AQI, its category, and the pollutant driving it.

    IMPORTANT CAVEAT, and it is why the returned category is worded plainly
    rather than dressed as an official figure: the real CPCB AQI is defined on
    **24-hour rolling averages** across up to eight pollutants, while the input
    here is the current hourly PM2.5/PM10 concentration. So this tracks the
    official number closely on a stable day and diverges during a sharp spike.
    It is an estimate on the CPCB scale, and callers must present it that way --
    never as "the AQI is X" with the authority of a CPCB station reading.
    """
    sub25 = _sub_index(pm25, _CPCB_PM25)
    sub10 = _sub_index(pm10, _CPCB_PM10)
    candidates = [(v, name) for v, name in ((sub25, "PM2.5"), (sub10, "PM10")) if v is not None]
    if not candidates:
        return None, None, ""
    value, driver = max(candidates)
    category = "severe"
    for ceiling, name in _CPCB_CATEGORIES:
        if value <= ceiling:
            category = name
            break
    return value, category, driver


# --- Heat ---------------------------------------------------------------

# Banded on APPARENT temperature, not the raw reading: 38 C in dry Jaipur and
# 38 C at 80% humidity in Kolkata are not the same event for a person with a
# heart condition, and only the apparent figure knows the difference.
_HEAT_BANDS = ((32.0, "none"), (38.0, "caution"), (45.0, "high"), (54.0, "very high"))


def heat_band(apparent_c: Optional[float]) -> str:
    """Heat-stress band from apparent temperature. "unknown" when absent."""
    if apparent_c is None:
        return "unknown"
    try:
        value = float(apparent_c)
    except (TypeError, ValueError):
        return "unknown"
    for ceiling, name in _HEAT_BANDS:
        if value < ceiling:
            return name
    return "extreme"


# Ordered worst-last so a caller can ask "did this get worse?" by index.
HEAT_ORDER = ("unknown", "none", "caution", "high", "very high", "extreme")
AQI_ORDER = ("unknown", "good", "satisfactory", "moderate", "poor",
             "very poor", "severe")


class EnvironmentProvider:
    """Polls weather + air quality and serves the last good reading.

    Thread-safe by copy: `snapshot()` never hands out the live dict, so a caller
    formatting a prompt cannot see it mutate underneath them mid-render.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.latitude = cfg.get("latitude")
        self.longitude = cfg.get("longitude")
        self.place = str(cfg.get("place_name") or "")
        self.timezone = str(cfg.get("timezone") or "Asia/Kolkata")
        self.poll_seconds = max(60, int(cfg.get("poll_interval_seconds", DEFAULT_POLL_SECONDS)))
        self.stale_seconds = int(cfg.get("stale_after_seconds", DEFAULT_STALE_SECONDS))
        self.unavailable_seconds = int(
            cfg.get("unavailable_after_seconds", DEFAULT_UNAVAILABLE_SECONDS))
        self.timeout = float(cfg.get("request_timeout_seconds", 10.0))

        self._lock = threading.RLock()
        self._latest: Optional[Dict[str, Any]] = None
        self._fetched_at: float = 0.0
        self._last_error: str = ""
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # -- lifecycle --

    def configured(self) -> bool:
        return (self.enabled and self.latitude is not None
                and self.longitude is not None and requests is not None)

    def start(self) -> bool:
        """Begin polling in the background. Idempotent; never raises."""
        if not self.configured():
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._poll_loop, name="EnvironmentPoll", daemon=True)
            self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh()
            except Exception as exc:                 # pragma: no cover - belt and braces
                with self._lock:
                    self._last_error = str(exc)
            # Interruptible sleep so a shutdown does not wait out a full interval.
            self._stop.wait(self.poll_seconds)

    # -- fetching --

    def refresh(self) -> bool:
        """Fetch both feeds once. Returns True when a reading was stored.

        A partial result counts: air quality alone is still worth having on a
        day the forecast endpoint is down, and vice versa. Only a total failure
        leaves the previous reading in place to keep ageing.
        """
        if not self.configured():
            return False
        weather, weather_err = self._get(
            FORECAST_URL, {"current": _FORECAST_FIELDS})
        air, air_err = self._get(
            AIR_QUALITY_URL, {"current": _AIR_FIELDS})
        if weather is None and air is None:
            with self._lock:
                self._last_error = weather_err or air_err or "no response"
            return False

        reading = self._build_reading(weather, air)
        with self._lock:
            self._latest = reading
            self._fetched_at = time.time()
            self._last_error = "" if (weather and air) else (weather_err or air_err)
        return True

    def _get(self, url: str, params: Dict[str, str]):
        query = dict(params)
        query.update({
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": self.timezone,
        })
        try:
            response = requests.get(url, params=query, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return None, f"{url.rsplit('/', 1)[-1]}: {exc}"
        current = payload.get("current")
        return (current, "") if isinstance(current, dict) else (None, "no current block")

    @staticmethod
    def _num(block: Optional[dict], key: str) -> Optional[float]:
        if not isinstance(block, dict):
            return None
        value = block.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _build_reading(self, weather: Optional[dict], air: Optional[dict]) -> Dict[str, Any]:
        temperature = self._num(weather, "temperature_2m")
        apparent = self._num(weather, "apparent_temperature")
        humidity = self._num(weather, "relative_humidity_2m")
        pm25 = self._num(air, "pm2_5")
        pm10 = self._num(air, "pm10")
        aqi, category, driver = cpcb_aqi(pm25, pm10)
        return {
            "source": "open-meteo",
            "place": self.place,
            "temperature_c": temperature,
            "apparent_temperature_c": apparent,
            "humidity_pct": humidity,
            "pm2_5": pm25,
            "pm10": pm10,
            "us_aqi": self._num(air, "us_aqi"),
            "aqi": aqi,
            "aqi_scale": "CPCB (estimated from current hourly PM, not a 24h average)",
            "aqi_category": category,
            "aqi_driver": driver,
            "heat_band": heat_band(apparent),
        }

    # -- reading --

    def freshness(self, now: Optional[float] = None) -> Tuple[str, Optional[float]]:
        """(state, age_seconds). State is fresh / stale / unavailable."""
        with self._lock:
            if self._latest is None:
                return UNAVAILABLE, None
            age = max(0.0, (now if now is not None else time.time()) - self._fetched_at)
        if age >= self.unavailable_seconds:
            return UNAVAILABLE, age
        if age >= self.stale_seconds:
            return STALE, age
        return FRESH, age

    def snapshot(self, now: Optional[float] = None) -> Dict[str, Any]:
        """The last good reading plus its freshness. Always a fresh copy.

        An `unavailable` snapshot deliberately carries NO values. Returning the
        numbers alongside a flag would sooner or later mean a caller formats the
        numbers and drops the flag, which is precisely how a two-hour-old AQI
        gets spoken as the current one.
        """
        state, age = self.freshness(now)
        with self._lock:
            latest = copy.deepcopy(self._latest) if self._latest else None
            error = self._last_error
        if state == UNAVAILABLE or latest is None:
            return {"available": False, "state": UNAVAILABLE,
                    "age_seconds": age, "error": error}
        latest.update({"available": True, "state": state,
                       "age_seconds": round(age or 0.0), "error": error})
        return latest

    def compact_line(self, now: Optional[float] = None) -> str:
        """One short line for the `CARE NOW` prompt snapshot, or "".

        Deliberately silent when nothing is noteworthy: a companion that
        announces ordinary weather every single turn is one the person stops
        listening to, and every character here is re-prefilled on each turn.
        """
        snap = self.snapshot(now)
        if not snap.get("available"):
            return ""
        parts = []
        heat = snap.get("heat_band")
        apparent = snap.get("apparent_temperature_c")
        if heat not in (None, "unknown", "none") and apparent is not None:
            parts.append(f"feels {apparent:.0f}C ({heat} heat)")
        category = snap.get("aqi_category")
        if category and category not in ("good", "satisfactory"):
            parts.append(f"AQI ~{snap.get('aqi')} {category}")
        if not parts:
            return ""
        if snap.get("state") == STALE:
            parts.append(f"{round((snap.get('age_seconds') or 0) / 60)}min old")
        prefix = f"{snap['place']}: " if snap.get("place") else ""
        return f"OUTSIDE {prefix}" + ", ".join(parts)


_provider: Optional[EnvironmentProvider] = None
_provider_lock = threading.RLock()


def get_environment_provider(config: Optional[dict] = None) -> EnvironmentProvider:
    """Process-wide singleton, built from `environment` in config.json."""
    global _provider
    with _provider_lock:
        if _provider is None:
            if config is None:
                try:
                    from tools_and_config.config_loader import get_full_config
                    config = get_full_config().get("environment", {})
                except Exception:
                    config = {}
            _provider = EnvironmentProvider(config)
        return _provider


def reset_environment_provider() -> None:
    """Drop the singleton. For tests and for a live config reload."""
    global _provider
    with _provider_lock:
        if _provider is not None:
            _provider.stop()
        _provider = None
