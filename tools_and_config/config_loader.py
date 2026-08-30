"""
Configuration loader for KikiFast voice assistant.
Loads .env for API keys and config.json for all tunables.

Originally config.json was read ONCE at import and required a restart to change.
The Web UI (webui/server.py) now supports LIVE editing of high-level behavioural
settings. To make that safe:

  - `reload_config()` re-reads config.json and DEEP-MERGES it into the existing
    CONFIG dict IN PLACE, so any module holding a reference to a nested block
    (e.g. get_idle_mind_config()) sees the new values without rebinding.
  - Modules that cache scalar values in __init__ should register a callback via
    `register_reload_listener()` and re-pull their tunables when it fires.
  - `LIVE_BLOCKS` marks which top-level blocks apply live; the rest are editable
    but flagged "restart required" in the UI (they're cached on latency-critical
    paths: tts / stt / llm / local_llm caching).
"""

import os
import json
import threading
from pathlib import Path
from dotenv import load_dotenv

# Load repository-local environment variables without depending on the caller's
# working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

_CONFIG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.example.json"
_configured_path = Path(os.environ.get("KIKIFAST_CONFIG", _CONFIG_DIR / "config.json"))
_CONFIG_PATH = _configured_path if _configured_path.is_absolute() else _REPO_ROOT / _configured_path
CONFIG_PATH = str(_CONFIG_PATH)  # public alias retained for compatibility


def _read_config_path() -> Path:
    """Return the user config, or the safe checked-in defaults on first run."""
    return _CONFIG_PATH if _CONFIG_PATH.exists() else _DEFAULT_CONFIG_PATH


with _read_config_path().open("r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# Top-level blocks whose changes take effect LIVE (no restart). Everything else is
# editable in the UI but cached on a latency-sensitive path → "restart required".
LIVE_BLOCKS = {
    "idle_mind", "workers", "vision_injection", "peeping",
    "agent", "face_events", "prompts",
    # Cloud cost guard (rate limits / per-category caps). Read live per call by
    # core/cloud_budget.py, so edits apply on the very next cloud request.
    "cloud_limits",
    # Routing flags (vision/summary/reasoning local↔cloud). The vision handler
    # and generate_llm_resp router both read this block LIVE per call, so edits
    # take effect without a restart.
    "use_local_llm",
}

_reload_listeners = []
_listeners_lock = threading.Lock()
_save_lock = threading.Lock()


def get_full_config():
    """Returns the full configuration dict (live reference)."""
    return CONFIG


# ---- typed accessors (all return live references into CONFIG) -------------
def get_llm_config():            return CONFIG["llm"]
def get_tts_config():            return CONFIG["tts"]
def get_stt_config():            return CONFIG["stt"]
def get_sfx_config():            return CONFIG["sound_effects"]
def get_tools_config():          return CONFIG.get("tools", {})
def get_controller_config():     return CONFIG.get("controller", {})
def get_prompts_config():        return CONFIG.get("prompts", {})
def get_face_events_config():    return CONFIG.get("face_events", {})
def get_agent_config():          return CONFIG.get("agent", {})
# Previously read ad-hoc via get_full_config().get(...) — now first-class:
def get_knowledge_base_config(): return CONFIG.get("knowledge_base", {})
def get_workers_config():        return CONFIG.get("workers", {})
def get_use_local_llm_config():  return CONFIG.get("use_local_llm", {})
def get_vision_injection_config(): return CONFIG.get("vision_injection", {})
def get_peeping_config():        return CONFIG.get("peeping", {})
def get_always_listen_config():  return CONFIG.get("always_listen_config", {})
def get_self_extend_config():    return CONFIG.get("self_extend", {})
def get_logging_config():        return CONFIG.get("logging", {})
def get_idle_mind_config():      return CONFIG.get("idle_mind", {})


# ---- live reload / save machinery ----------------------------------------
def register_reload_listener(fn):
    """Register a no-arg callback fired (best-effort) after every reload_config()."""
    with _listeners_lock:
        if fn not in _reload_listeners:
            _reload_listeners.append(fn)


def _deep_merge_inplace(dst: dict, src: dict):
    """Recursively update dst with src, mutating nested dicts in place so that
    holders of references to dst's sub-dicts observe the change. Keys removed in
    src are dropped from dst."""
    for k in list(dst.keys()):
        if k not in src:
            del dst[k]
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge_inplace(dst[k], v)
        else:
            dst[k] = v


def reload_config():
    """Re-read config.json from disk, merge in place, fire listeners.
    Returns (ok: bool, error: str|None)."""
    try:
        with _read_config_path().open("r", encoding="utf-8") as f:
            fresh = json.load(f)
    except Exception as e:
        return False, f"parse error: {e}"
    _deep_merge_inplace(CONFIG, fresh)
    with _listeners_lock:
        listeners = list(_reload_listeners)
    for fn in listeners:
        try:
            fn()
        except Exception as e:
            print(f"[config] reload listener {getattr(fn,'__name__',fn)} failed: {e}")
    return True, None


def save_config(new_config: dict = None) -> bool:
    """Atomically write CONFIG (or new_config) to config.json. Does NOT reload."""
    data = new_config if new_config is not None else CONFIG
    with _save_lock:
        try:
            _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _CONFIG_PATH.with_suffix(_CONFIG_PATH.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, _CONFIG_PATH)
            return True
        except Exception as e:
            print(f"[config] save failed: {e}")
            return False


def set_config_value(path: str, value):
    """Set a dotted key (e.g. 'idle_mind.max_sessions_per_day') in CONFIG
    (in memory only — call save_config() + reload_config() to persist/apply)."""
    parts = path.split(".")
    node = CONFIG
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value


def apply_config_update(updates: dict) -> tuple:
    """High-level helper for the Web UI: apply a {dotted_path: value} map, persist
    to disk, and reload (firing listeners). Returns (ok, error, restart_needed)."""
    restart_needed = False
    for path, value in updates.items():
        set_config_value(path, value)
        top = path.split(".", 1)[0]
        if top not in LIVE_BLOCKS:
            restart_needed = True
    if not save_config():
        return False, "could not write config.json", restart_needed
    ok, err = reload_config()
    return ok, err, restart_needed
