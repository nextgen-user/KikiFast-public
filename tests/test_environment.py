"""Weather/air-quality ingestion: freshness discipline and the CPCB scale.

Every test here mocks the network. The point of the module is what it does when
a feed is slow, partial, or gone -- which is exactly what a live-network test
cannot pin down.
"""

import pytest

from core.health import environment as env
from core.health.environment import (
    EnvironmentProvider, cpcb_aqi, heat_band, FRESH, STALE, UNAVAILABLE,
)

CONFIG = {
    "enabled": True,
    "place_name": "New Delhi",
    "latitude": 28.6139,
    "longitude": 77.2090,
    "poll_interval_seconds": 900,
    "stale_after_seconds": 1800,
    "unavailable_after_seconds": 7200,
}

# The real 2026-08-29 Delhi reading this module was first validated against.
WEATHER = {"temperature_2m": 29.2, "apparent_temperature": 34.8,
           "relative_humidity_2m": 75, "weather_code": 3}
AIR = {"pm2_5": 120.6, "pm10": 338.1, "us_aqi": 376}


def make(monkeypatch, weather=WEATHER, air=AIR, config=None):
    """A provider whose two feeds return exactly what the test says."""
    provider = EnvironmentProvider(config or CONFIG)

    def fake_get(url, params):
        if "air-quality" in url:
            return (dict(air), "") if air is not None else (None, "air down")
        return (dict(weather), "") if weather is not None else (None, "forecast down")

    monkeypatch.setattr(provider, "_get", fake_get)
    return provider


# --- CPCB AQI ---

@pytest.mark.parametrize("pm25,pm10,expected,category,driver", [
    (120.6, 338.1, 301, "very poor", "PM2.5"),   # the live Delhi reading
    (15.0, 40.0, 40, "good", "PM10"),
    (0.0, 0.0, 0, "good", "PM2.5"),
    (45.0, 80.0, 80, "satisfactory", "PM10"),    # PM10 out-drives PM2.5 here
    (300.0, 100.0, 439, "severe", "PM2.5"),
])
def test_cpcb_aqi_matches_the_published_scale(pm25, pm10, expected, category, driver):
    value, name, source = cpcb_aqi(pm25, pm10)
    assert value == expected
    assert name == category
    assert source == driver


def test_the_worse_pollutant_drives_the_index():
    """CPCB takes the MAX sub-index, not an average -- clean PM2.5 must not
    dilute a dust storm's PM10."""
    value, category, driver = cpcb_aqi(10.0, 400.0)
    assert driver == "PM10"
    assert category == "very poor"
    assert value > cpcb_aqi(10.0, 10.0)[0]


def test_a_concentration_in_the_published_gap_still_lands_in_a_band():
    """CPCB prints integer bands (0-30, 31-60...). 120.6 falls between 120 and
    121; treating the breakpoints as continuous is what stops it vanishing."""
    for concentration in (30.5, 60.5, 90.5, 120.6, 250.5):
        value, category, _ = cpcb_aqi(concentration, None)
        assert value is not None, concentration
        assert category is not None


def test_missing_or_junk_concentrations_produce_no_index():
    assert cpcb_aqi(None, None) == (None, None, "")
    assert cpcb_aqi("not a number", None) == (None, None, "")
    assert cpcb_aqi(-5.0, None) == (None, None, "")


def test_off_the_scale_reports_the_ceiling_rather_than_extrapolating():
    value, category, _ = cpcb_aqi(2000.0, None)
    assert value == 500
    assert category == "severe"


# --- heat ---

@pytest.mark.parametrize("apparent,band", [
    (25.0, "none"), (34.8, "caution"), (40.0, "high"),
    (47.0, "very high"), (60.0, "extreme"), (None, "unknown"),
])
def test_heat_bands_use_apparent_temperature(apparent, band):
    assert heat_band(apparent) == band


# --- freshness lifecycle ---

def test_no_reading_yet_is_unavailable(monkeypatch):
    provider = make(monkeypatch)
    snap = provider.snapshot()
    assert snap["available"] is False
    assert snap["state"] == UNAVAILABLE


def test_a_reading_ages_fresh_then_stale_then_unavailable(monkeypatch):
    provider = make(monkeypatch)
    assert provider.refresh() is True
    stamped = provider._fetched_at

    assert provider.snapshot(now=stamped + 60)["state"] == FRESH
    assert provider.snapshot(now=stamped + 1801)["state"] == STALE
    assert provider.snapshot(now=stamped + 7201)["state"] == UNAVAILABLE


def test_an_unavailable_snapshot_carries_no_values(monkeypatch):
    """The whole point of the lifecycle.

    Returning the numbers next to an `available: False` flag would eventually
    mean a caller renders the numbers and drops the flag -- which is how a
    two-hour-old AQI gets spoken as the current one.
    """
    provider = make(monkeypatch)
    provider.refresh()
    snap = provider.snapshot(now=provider._fetched_at + 99999)
    assert snap["available"] is False
    for banned in ("pm2_5", "aqi", "temperature_c", "apparent_temperature_c"):
        assert banned not in snap


def test_a_stale_snapshot_keeps_its_values_but_reports_its_age(monkeypatch):
    provider = make(monkeypatch)
    provider.refresh()
    snap = provider.snapshot(now=provider._fetched_at + 2000)
    assert snap["available"] is True
    assert snap["state"] == STALE
    assert snap["age_seconds"] == 2000
    assert snap["pm2_5"] == 120.6


# --- partial and total failure ---

def test_air_quality_alone_is_still_worth_storing(monkeypatch):
    provider = make(monkeypatch, weather=None)
    assert provider.refresh() is True
    snap = provider.snapshot()
    assert snap["available"] is True
    assert snap["aqi"] is not None
    assert snap["temperature_c"] is None
    assert snap["heat_band"] == "unknown"
    assert "forecast down" in snap["error"]


def test_weather_alone_is_still_worth_storing(monkeypatch):
    provider = make(monkeypatch, air=None)
    assert provider.refresh() is True
    snap = provider.snapshot()
    assert snap["available"] is True
    assert snap["heat_band"] == "caution"
    assert snap["aqi"] is None


def test_a_total_outage_leaves_the_previous_reading_ageing(monkeypatch):
    """A failed poll must not reset the clock, or a permanently-down feed would
    look permanently fresh."""
    provider = make(monkeypatch)
    provider.refresh()
    stamped = provider._fetched_at

    monkeypatch.setattr(provider, "_get", lambda url, params: (None, "network down"))
    assert provider.refresh() is False
    assert provider._fetched_at == stamped
    assert provider.snapshot()["available"] is True


def test_the_poll_loop_survives_an_exploding_refresh(monkeypatch):
    """The thread must outlive any single failure.

    A poll thread that dies on the first unexpected exception looks exactly
    like a working one -- the last reading simply ages out hours later with
    nothing in the log to say why.
    """
    provider = EnvironmentProvider(CONFIG)
    calls = []

    def boom():
        calls.append(1)
        provider._stop.set()          # one iteration, then leave
        raise RuntimeError("connection reset")

    monkeypatch.setattr(provider, "refresh", boom)
    provider._poll_loop()             # must return rather than propagate
    assert calls == [1]
    assert "connection reset" in provider._last_error


def test_the_real_get_swallows_transport_errors(monkeypatch):
    """The genuine failure path: `requests` raising must become (None, reason)."""
    provider = EnvironmentProvider(CONFIG)

    class Boom:
        @staticmethod
        def get(*a, **k):
            raise OSError("no route to host")

    monkeypatch.setattr(env, "requests", Boom)
    result, error = provider._get(env.FORECAST_URL, {"current": "x"})
    assert result is None
    assert "no route to host" in error
    assert provider.refresh() is False


# --- configuration gates ---

def test_an_unconfigured_provider_never_polls():
    for missing in ({"latitude": None}, {"longitude": None}, {"enabled": False}):
        config = dict(CONFIG)
        config.update(missing)
        provider = EnvironmentProvider(config)
        assert provider.configured() is False
        assert provider.start() is False
        assert provider.refresh() is False
        assert provider.snapshot()["available"] is False


def test_the_poll_interval_has_a_floor():
    """A misconfigured 1-second interval would hammer a free public API."""
    provider = EnvironmentProvider({**CONFIG, "poll_interval_seconds": 1})
    assert provider.poll_seconds >= 60


# --- compact prompt line ---

def test_the_compact_line_names_only_what_is_noteworthy(monkeypatch):
    provider = make(monkeypatch)
    provider.refresh()
    line = provider.compact_line()
    assert "caution heat" in line
    assert "very poor" in line
    assert "New Delhi" in line
    assert len(line) < 120       # it is re-prefilled on every turn


def test_an_ordinary_day_says_nothing_at_all(monkeypatch):
    """Silence is the default. A companion that announces pleasant weather on
    every turn is one the person stops listening to."""
    provider = make(monkeypatch,
                    weather={"apparent_temperature": 24.0, "temperature_2m": 24.0},
                    air={"pm2_5": 10.0, "pm10": 20.0})
    provider.refresh()
    assert provider.compact_line() == ""


def test_a_stale_compact_line_admits_its_age(monkeypatch):
    provider = make(monkeypatch)
    provider.refresh()
    line = provider.compact_line(now=provider._fetched_at + 2400)
    assert "40min old" in line


def test_no_reading_produces_no_line(monkeypatch):
    assert make(monkeypatch).compact_line() == ""


# --- isolation ---

def test_snapshot_hands_out_a_copy(monkeypatch):
    provider = make(monkeypatch)
    provider.refresh()
    first = provider.snapshot()
    first["pm2_5"] = 0.0
    assert provider.snapshot()["pm2_5"] == 120.6


def test_the_singleton_can_be_reset():
    env.reset_environment_provider()
    first = env.get_environment_provider({"enabled": False})
    assert env.get_environment_provider() is first
    env.reset_environment_provider()
    assert env.get_environment_provider({"enabled": False}) is not first
    env.reset_environment_provider()


def test_the_shipped_config_configures_a_usable_provider():
    from tools_and_config.config_loader import get_full_config

    provider = EnvironmentProvider(get_full_config().get("environment", {}))
    assert provider.configured() is True
    assert provider.stale_seconds == 1800
    assert provider.unavailable_seconds == 7200
