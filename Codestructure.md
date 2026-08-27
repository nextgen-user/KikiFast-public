**BEFORE ANY CHANGE YOU DO , FIRST COMMIT AND PUSH  THE CURRENT STATE with git add . , git commit and git push. THis will help in restoring code if you mess up.**
# KikiFast — Code Structure & Architecture Reference

> Last updated: 2026-07-21. Line numbers refer to the files as of this date; they drift as
> code changes, so anchors are given as `function/class @ ~line`. When in doubt, grep the
> symbol name.

> **Unified Idle Mind (2026-07-26):** `core/brain/unified_idle_mind.py` is
> Kiki's only background cognition process. It chooses `no_action`, `reflect`,
> `light_research`, or `deep_research`; ambient listening is capture-only;
> `recall_memory` searches knowledge, conversations, and dated journal research;
> and one model-selected next-turn note is the only background prompt injection.
> The process is cloud-only and continues safely if foreground speech begins.

KikiFast is a low-latency voice companion robot ("Kiki") running on a Raspberry Pi 5,
speaking through a local llama.cpp box (gemma-4-26B over Tailscale), with a local
omnivoice.cpp TTS server, local whisper.cpp STT, OpenWakeWord hotword detection, a Hailo-based
face/vision pipeline, mecanum-wheel chassis, and a layered memory system (knowledge base,
thinking journal, conversation summaries).
Always run  source /srv/kikifast/.venv/bin/activate before main.py to get into the venv..
---

## 1. Hardware / Service Topology

| Component | Where | Endpoint |
|---|---|---|
| Main orchestrator (`main.py`) | Raspberry Pi 5 (this repo) | — |
| Foreground speaking LLM (gemma-4-26B-A4B, llama-server) | Alex's laptop (RTX 4060) over Tailscale | `http://192.0.2.10:8080/v1` |
| TTS (omnivoice.cpp tts-server, 24 kHz PCM) | Same laptop | `http://192.0.2.10:8082` |
| STT (whisper.cpp server, HTTP POST + Silero VAD) | Same laptop (Tailscale) | `http://192.0.2.10:5555/inference` |
| Cloud LLM fallback | Vertex AI Gemini (via litellm) / google-genai / Groq | API keys in `.env` + Vertex service-account key `example-project-347dc1f51500.json` (project `example-project`, location `global`) |
| Web search | Exa API | key hardcoded in `tools.py` (⚠) |
| Face recognition + neck/relay control (KikiController) | Hailo pipeline process | ZMQ `192.0.2.20:5555` (REQ commands) / `:5556` (SUB events) |
| Chassis motor server (collision-safe movement) | local process (`hailo_follower_webcam_only.py`) | ZMQ `127.0.0.1:5557` |
| Camera | local MJPEG stream | `http://localhost:5000/mjpeg` |
| WhatsApp bridge + MCP | local Go bridge + Python stdio MCP daemon | bridge REST `http://127.0.0.1:8080/api`; MCP owned by Kiki |
| Audio out | ALSA via `aplay` (TTS) and `mpv` (music/sfx); configured A2DP sink guarded by `core/audio_output.py` | — |

**The llama.cpp box is single-slot (`-np 1`)**: one request at a time runs fast. The entire
client architecture is built around protecting that slot for the *speaking* path and making
every background request instantly killable. See §4 — this is the most important section
in this document.

llama-server launch command (current, correct shape):
```
./build/bin/llama-server -m gemma-4-26B-A4B-it-UD-Q4_K_S.gguf -ngl 99 --n-cpu-moe 30 \
  -fa on -c 7000 -ctk q8_0 -ctv q8_0 -np 1 --ctx-checkpoints 6 --cache-prompt \
  --mmproj gemma-4-26B-it-mmproj.gguf --mmproj-offload \
  --model-draft gemma-4-26B-A4B-it-MTP-BF16.gguf --spec-type draft-mtp --spec-draft-n-max 1 \
  --host 0.0.0.0 --port 8080 -l 64980-100,27388-100,45518-100,15081-100
```
Critical: **no `--reasoning-budget` flag** (it would silence the per-request
`thinking_budget_tokens` field every client request sends), `-np 1` (single slot, full ctx),
comma-separated `-l` (multiple `-l` flags silently drop all but the last).

---

## 2. File Tree

```
KikiFast/
├── main.py                     # Orchestrator: wake→listen→think→speak loop + all background tasks
├── kiki_boot.py                # System boot: config wizard/Wi-Fi → services → exec main.py
├── kiki_control_client.py      # Async ZMQ client for the Hailo face/neck/relay controller
├── kiki_startup.sh             # Boot script: env (Pulse/X), launches the full process stack
├── myname.md                   # Historical session transcript (the big June refactor notes)
├── localllmspeedup.md          # Notes on llama.cpp speedups
├── Codestructure.md            # ← this file
├── knowledge_base.json         # Long-term memory (people/learnings/facts/...) — gitignored data
├── thinking_journal.json       # Dated Unified Idle Mind research + open questions
├── workers.json                # Persisted background workers
├── conversation_summary.txt    # Last session summary (single file)
├── conversations/              # Per-session timestamped summaries + cached_past_summary.txt
├── logs/kiki.log               # Tee'd stdout/stderr with timestamps (rotated at 5 MB)
├── kiki.onnx                   # OpenWakeWord model for the "kiki" wake word
├── .env                        # API keys (Deepgram, Gemini, Groq, Smithery, GitHub...)
│
├── core/
│   ├── llm.py                  # SPEAKING path: SSE streaming → sentences, tool-call parsing, cloud fallback
│   ├── local_llm.py            # Single-slot coordinator: background gen, preempt, rewarm, cache hashing
│   ├── stt.py                  # Local whisper.cpp STT with mute/unmute + server-side Silero VAD
│   ├── noise_suppression.py    # RNNoise at capture time: kills steady noise, NOT voices (§5.4c)
│   ├── near_field_gate.py      # Crowd fix: adaptive noise floor + near-field level gate (§5.4c)
│   ├── tts.py                  # TTS streamers (local omnivoice / Groq / Inworld) + tag sanitizer
│   ├── audio_output.py         # Repairs Pulse auto_null fallback + pins aplay to configured A2DP sink
│   ├── speech_recorder.py      # record_enabled: off-path wav archive of what Kiki actually said (§5.5a)
│   ├── tts_sync.py             # Calibration-profile word timing; no runtime Whisper dependency
│   ├── media_manager.py        # Exact-video music state/likes/playlist controls + timer alarms
│   ├── lcd_display.py          # 16x2 I2C char-LCD status/streaming display (singleton, async worker)
│   ├── oled_display.py         # 128x64 SSD1306 pixel-crab face: 35 states + <oled:> expression tags (§5.18f)
│   ├── oled_log_feed.py        # stdout/stderr tap → the resting face's background-activity ticker
│   ├── ir_controls.py          # Two IR sensors (GPIO22/17): left-wave interrupt + LCD settings menu
│   ├── gesture_controls.py     # Thread-safe camera-gesture mute + activity-stop state
│   ├── startup_config.py       # Boot IR/LCD Wi-Fi/provider/volume/language/mode wizard
│   ├── wifi_setup.py           # Offline IR/LCD Wi-Fi picker + local Whisper password dictation
│   ├── runtime_controls.py     # Spoken mode/follow-up state + deterministic command parsing
│   ├── observability.py        # Recorder singleton: flat events + grouped SESSIONS for the Web UI (§5.23)
│   ├── agent_loop.py           # run_agent_loop: shared JSON-protocol multi-turn tool agent (+ session logging)
│   ├── brain/
│   │   ├── unified_idle_mind.py# One cloud background agent: decisions, tools, notes, scheduling
│   │   ├── ambient_listening.py# Capture-only crash-safe ambient transcript buffer
│   │   ├── thinking_journal.py # Dated research/open-question persistence and dedupe
│   │   ├── knowledge_base.py   # Hierarchical long-term memory + context summary
│   │   ├── memory_search.py    # Ranked recall over KB + all conversations + journal
│   │   ├── summary_manager.py  # Conversation summaries, past-summary cache
│   │   └── token_counter.py    # tiktoken-based context counting with per-message cache
│   ├── workers/
│   │   ├── worker_engine.py    # Dataclasses: Worker, WorkerTrigger, WorkerCondition, enums
│   │   ├── worker_brain.py     # execute_worker (builds prompt → run_agent_loop), face/vision buffers
│   │   └── worker_manager.py   # Scheduler thread, persistence, lifecycle events, live-toggle reload listener
│   ├── vision/
│   │   ├── camera.py           # capture_photo_b64 from the MJPEG stream
│   │   ├── vision_handler.py   # Periodic capture/QA logic, silent scene injection
│   │   └── instant_vision.py   # Live-image questions → Groq multimodal Qwen (streamed), §5.2a
│   ├── cloud_budget.py         # Cost guard: rate limits + per-feature caps for paid cloud calls (§5.23)
│   └── self_extend/
│       ├── skill_manager.py    # Local SKILL.md skills (list/create/summarize)
│       ├── smithery_cli.py     # subprocess wrapper around the `smithery` CLI
│       ├── whatsapp_mcp.py     # Long-lived bundled WhatsApp MCP/bridge lifecycle + compact calls
│       ├── mcp_manager.py      # MCP registry/config/codegen helpers
│       └── kiki_self_extend_agent.py # Autonomous JSON-loop agent for installing skills/MCPs
│
├── webui/server.py             # Flask dashboard on :8090 — Controls/Sessions/Live Feed/Context/Advanced (§5.23)
├── hotwords/hotword_recog.py   # OpenWakeWord recognizer + pause/quiet-resume logic
├── robot/
│   ├── face_handler.py         # Async listener/router for face + hand-gesture ZMQ events
│   ├── movement.py             # <tag> parsing for movement, strip for TTS, legacy GPIO executor
│   └── motor_control.py        # Low-level GPIO/SoftPWM chassis driver (used by the motor SERVER, not tools.py)
├── sound_effects/
│   ├── sound_effects.py        # ThinkingSoundPlayer: one random filler wav while LLM thinks
│   ├── soundeffects/fillers/   # Generated filler wavs (from tests/streaming_tts.py)
│   └── audioeffects/           # Wake-word audio stingers (currently unused on wake path)
├── tools_and_config/
│   ├── config.json             # THE config: all tunables, prompts, personalities
│   ├── config_loader.py        # Loads .env + config.json once at import
│   ├── logger.py               # stdout/stderr tee → logs/kiki.log with rotation
│   └── tools.py                # All tool implementations + OpenAI schemas + dispatch
├── tests/streaming_tts.py      # Standalone gapless TTS lib + FILLERS list + filler wav generator
├── tests_llamaserver/
│   ├── test_prefill_e2e.py     # E2E KV-cache tests against the live box (3 tests)
│   └── test_preempt_prefill.py # Earlier preemption test
├── skills/                     # Installed SKILL.md skills (injected into system context)
└── whatsapp-mcp/               # Bundled Go WhatsApp bridge + 12-tool Python FastMCP server
```

---

## 3. Runtime Flows

### 3.0 System boot and offline Wi-Fi setup (`kiki_boot.py`)

`kikifast.service` runs `kiki_boot.py` in the Kiki venv. Before any laptop or Hailo
service is touched, the boot orchestrator checks `auto_start`; when false, it exits and
starts nothing. Otherwise, before Wi-Fi setup or any boot sound, it powers on Bluetooth,
connects the paired `bluetooth_speaker` (Example Speaker; configured MAC first with paired-name
discovery as fallback), waits for its A2DP PulseAudio sink, and selects that sink as default.
It retries for `connect_timeout_seconds`; by default a powered-off speaker is logged but does
not prevent Kiki from booting (`required: false`).

After Bluetooth audio is ready, `core/startup_config.py` briefly takes ownership of
GPIO22/17 and offers `Edit config?` on the LCD. Tap LEFT/RIGHT to navigate and hold either
sensor to select; each recognized action plays the short `startup_config_beep.wav` feedback
sound. Selecting No continues immediately, and no input auto-selects No after 20 seconds
so unattended boots cannot stall. Selecting Yes walks through Wi-Fi
(`Keep current`/`Change WiFi`), speaking brain (`local`/`cerebras`), Bluetooth volume
(tap LEFT +10%, tap RIGHT -10%, hold to confirm), language (`English`/`Hindi`), and every
configured `assistant_modes` startup mode, then atomically saves `config.json` and continues
boot. The confirmed volume is applied live and saved as
`bluetooth_speaker.volume_percent` for future startups. English does not alter prompts;
Hindi saves `assistant_modes.language_on_startup = "hindi"` and appends
`Always output hindi devnagri.` to the effective system prompt for every mode, including
after runtime mode switches.
After the startup mode, the wizard asks `Sync LCD+audio?` — the LCD/audio word-timing
calibration (§5.18d). It is the last question because every earlier choice can invalidate
the saved profile (a different speaker, TTS server, or mode voice), and it preselects
**Yes** whenever `startup_config.calibration_status()` finds the stored profile unusable,
so the ordinary "my speaker changed" case is a single hold. The answer is **not** written
to `config.json`: it is a one-boot request returned alongside the config by
`offer_startup_config()`, because the calibration cannot run here — the TTS and Whisper
servers it needs are only started later. The boot orchestrator therefore defers it to step
10, after all four services are live: it stops the looping boot sound first (the script
measures real speaker→microphone latency, so any other audio would corrupt it), runs
`tests/calibrate_lcd_sync.py` in the venv via `run_lcd_sync_calibration()`, and streams the
script's progress lines onto the LCD (`Calibrating.../<voice>`, `Speaker delay`,
`Checking sync`). main.py is not running yet, so nothing else holds the microphone.
Calibration is never a boot gate: a missing script, a crash, a validation failure, or the
900s timeout (`tts.display_sync.calibration_timeout_seconds`) all log, show
`Calib failed / using old sync`, and continue to `main.py` with the previous profile
intact.

The Wi-Fi selector remembers the exact connection active on entry and restores it when a
change is canceled or fails. The feedback beep is scoped to this startup wizard (including
its Wi-Fi subflow); normal runtime IR gestures remain silent.

After the wizard, the orchestrator checks for a connected Wi-Fi device and gives
NetworkManager `wifi_setup.connection_grace_seconds` to reconnect a saved network
automatically. If still offline, `core/wifi_setup.py` takes temporary ownership of
GPIO22/17 and the LCD:

```
nmcli scan → strongest SSID on LCD
  tap LEFT / RIGHT → next / previous SSID
  hold either      → select
    saved/open network → connect directly
    secured network    → password screen
      hold LEFT + speak characters + release → local whisper-cli → parse → nmcli --ask
      tap RIGHT                             → backspace
      hold RIGHT                            → clear
      hold BOTH                             → return to SSID list
```

`wifi_setup.preferred_ssids` is probed with directed NetworkManager scans and remains
selectable at the end of the LCD list when absent from ordinary beacon results. This
supports idle mobile hotspots and hidden SSIDs such as `ExampleNet`; their password connection
uses NetworkManager's `hidden yes` mode. A failed attempt still restores the exact Wi-Fi
profile that was active before the startup selector opened. A missing hotspot returns to
the network list with `Hotspot not found` instead of presenting the failure as a bad
password or continuously polling NetworkManager.

The laptop Whisper server cannot be reached before Wi-Fi exists, so this path records
16 kHz PCM from the configured STT device and runs the checked-in
`whisper.cpp/build/bin/whisper-cli` with `models/ggml-base.en.bin`. Dictation accepts forms
such as `k i k i`, `capital k i k i`, spoken `clear`, and spoken `backspace`. Whisper may
join spelled letters into a word; the password parser splits that word back into
characters. The LCD shows the entered password directly through `lcd.write_raw` (so special
characters are not stripped), using both rows and retaining the newest 32 characters if it
exceeds the panel. Logs never contain the transcript/password, and `nmcli --ask` receives
the password over stdin rather than argv.

Once connected, the GPIO lines and microphone are released, the WhatsApp Go bridge is
started at low priority without waiting, then the existing critical sequence continues:
start the looping boot sound → POST the laptop manager `/restart` → restart
`hailo_follower.service` → wait for llama/TTS/Whisper/Hailo → stop sound → exec `main.py`.
Bridge compilation/reconnect overlaps the laptop model-load window and is never a boot gate.
The `--force` settings-menu restart uses this same Wi-Fi gate while ignoring `auto_start`.

### 3.1 A normal speaking turn
```
"kiki" → HotwordRecognizer (hotwords/hotword_recog.py, own thread)
  → main.py hotword handler:  stt.unmute() FIRST (zero added latency)
                              → set_motor_relay(True) (daemon thread)
                              → local_llm.preempt_background()  (non-blocking; kills any bg prefill)
                              → idle_mgr.interrupt()/mark_activity()  (marks conversation HOT)
user speech → STTEngine (Deepgram Flux) → ("final", text) events → stt_queue
"endpoint" event → main loop:
  mute mic, pause hotword recognizer, start ThinkingSoundPlayer (one filler wav)
  append context (time every 5 min, pending vision, one Unified Idle Mind note)
  append {"role":"user", ...}
  stream_response(message_history)            # core/llm.py
    └─ _stream_local(): note_user_activity + note_speaking(True) + preempt, then SSE POST
       sentences yielded as they complete → TTSStreamer.add_sentence()
       <tool_call>{...}</tool_call> in stream → parsed → tool runs in bg thread
         (canned TOOL_FILLER spoken if nothing said yet) → result appended as system note
         → followup_llm_and_tts() streams the answer onto the SAME TTS streamer
  tts_streamer.finish() blocks until audio drains → recognizer.request_resume()
  append {"role":"assistant", clean_response}
  register_history(message_history)           # ← registers + rewarms EXACT prefix (see §4)
  fire after_response workers, maybe vision update, token count, maybe auto-summarize
  unmute after post_speech_unmute_delay_seconds (0.35s); 15s of silence → mute (hotword mode)
```

Clear settings utterances are routed before generation: “switch to funny mode” calls
`switch_mode`, volume requests call `adjust_volume`, and follow-up requests call
`set_followups`. A mode switch applies its configured TTS voice immediately, replaces the
first system message only after the acknowledgement finishes, then explicitly re-warms the
new prompt. Disabling follow-ups closes the query window after every answer and forces true
hotword-only listening even when `always_listen` is configured.

### 3.2 Startup sequence (`main.py main()` top half)
1. `setup_logging()` before any other import — every print everywhere is tee'd to the log.
2. Eager module imports (~2s) — each module pre-initializes at import time.
3. Build context: KB summary + latest conversation + skills summary → `message_history[0..1]`.
4. `load_past_summary_bg()` task — loads `conversations/cached_past_summary.txt` with
   `prefer_cache=True` (instant, even if stale) and splices into `message_history[1]`.
5. `refresh_past_summary_cache_bg()` task — 3 min later regenerates the cache **for the
   next boot only** (never touches the live context).
6. STT engine + hotword recognizer + worker manager start; `fire_event("startup")` workers run.
7. `warmup_when_context_ready()` task: awaits the summary splice (≤180s), bakes pending
   Unified Idle Mind next-turn note into the prefix, then runs the ONE startup warmup
   (`llm_warmup(message_history)`) — full prefill on the box (~15–60s, in background).
8. Unified Idle Mind monitor, non-blocking WhatsApp MCP daemon, face listener,
   periodic-question loop, peeping loop, STT thread start.
9. Main `while True` event loop over `stt_queue`.

### 3.3 Unified Idle Mind
```
main.py post-turn → idle_mgr.note_turn(user, assistant)
15s monitor tick → meaningful conversation lull OR model-selected next-check time
                  OR debounced new WhatsApp messages
  → build one cloud prompt from new turns, ambient snapshot, KB, recent journal,
    open questions, current note, recent intents, and untrusted WhatsApp updates
  → model chooses no_action | reflect | light_research | deep_research
  → shared agent loop executes bounded, validated tools
  → deliberate writes: knowledge, dated research, open questions, one next-turn note
  → model chooses the next check (30–360 minutes) or waits for a new conversation
```

The agent never uses the local speaking slot. User speech does not cancel its cloud
request; resource-conflicting actions queue until the conversation is no longer hot.
Physical movement and tracking are blocked. Exact and near-duplicate tool intents are
rejected using the last twelve completed sessions. Light research permits two
investigative calls, deep research permits six, and action/persistence budgets prevent
runaway loops.

WhatsApp is polled locally through the long-lived MCP session; polling itself uses no
cloud model. New incoming messages are debounced and passed to the next Unified Idle Mind
session as untrusted private content. The model may deliberately save durable personal
facts with `update_knowledge` and select one time-sensitive event with
`set_next_turn_note`. It must never autonomously reply: sends require a recent explicit
request from Alex or a configured worker. WhatsApp sends/media actions requested by
the idle mind are treated as resource-conflicting and queue while conversation is hot.

### 3.4 Ambient listening

`AmbientListeningManager` only stores finalized passive STT sentences in
`ambient_listen_buffer.json`. It has no timer, cloud model, journal writer, knowledge
writer, or prompt injector. Unified Idle Mind snapshots up to 24 buffered snippets and
consumes those IDs only after a successful session.

### 3.5 Background work vs. the speaking slot

Unified Idle Mind is cloud-only. Shared summary/vision routes may use
`local_llm.generate_background()`, which is refused while conversation is hot,
preemptible by socket shutdown, and followed by speaking-prefix rewarm. The single local
slot therefore remains owned by foreground speech.

### 3.6 Always-listen capture boundary

When `always_listen: true`, STT remains in ambient capture mode while Kiki is idle.
The wake word or IR input switches STT to query mode immediately; no cloud flush occurs
on that path. Kiki's own TTS is excluded because microphone capture and hotword inference
remain paused while output audio is active.
---

## 4. The KV-Cache Contract (read this before touching message_history)

gemma-4 is an **SWA model** (sliding-window attention, n_swa=1024). On the single slot,
any byte-level divergence between the cached prompt and a new request forces re-prefill
from the divergence point — and if no context checkpoint covers that point, **full**
re-processing (~40–60s). Four hard rules keep turns at ~1–2s TTFT:

1. **Never mutate an existing message in `message_history` mid-session.** Appending new
   messages extends the cached prefix; editing `message_history[1]` invalidates everything
   after it. (The old code rewrote msg[1] every 5 minutes — that was the "context
   invalidated every 4-5 turns" bug.) Workers context is now append-only-on-change
   (`last_workers_context` guard in main.py ~line 614).
2. **Store the assistant reply VERBATIM, and rewarm from that same history.** The box's
   KV cache holds the reply *exactly as generated* — `<neck:…>`, `<oled:…>` and `<tool_call>…</tool_call>`
   tags included. So `message_history` must store the reply verbatim (tags and all), or the
   resent history diverges from the cache right where the first tag was and forces a
   re-prefill. main.py captures the raw `done` text (`raw_first` / `raw_followup`) and
   appends THAT; the tag-stripping for TTS/logging happens on a separate `clean_response`
   copy. For a **tool turn** the order in history must mirror the slot:
   `assistant(first_gen with <tool_call>)` → `system(result note)` → `assistant(follow-up)`
   — the pre-tool assistant message is appended *before* the result note (omitting it
   diverged the cache for both the follow-up request and the next turn). If the follow-up
   itself emits a `<tool_call>` (§5.1a) the same three rows repeat, with the follow-up
   generation now playing the "first_gen" role. `register_history()`
   derives the prefix from `message_history` itself and is called *after* these appends; do
   not re-introduce prefix registration inside the streaming path.
3. **One warmup at startup, after the context is final.** The past summary loads from
   cache instantly (`prefer_cache=True`) and talking points are baked in pre-warmup
   precisely so the warmup sees the final prefix. Competing warmups invalidate each other.
4. **Background prompts evict the speaking prefix.** That's why every background request
   auto-rewarms afterwards, why rewarms are hash-deduped (`_warm_prefix_hash`), and why
   the conversation-hot window reroutes background work to the cloud.

Verified end-to-end by `tests_llamaserver/test_prefill_e2e.py` (cold 24s → warm 1.26s;
preempt 0 ms, mid-prefill abort; mutation demo). Run it whenever you touch this machinery.

---

## 5. File-by-File Deep Dive

### 5.1 `main.py` (~1063 lines) — Orchestrator

| Segment | What it does |
|---|---|
| lines 1–44 | Module docstring (the 11-step flow), `setup_logging()` installed before all other imports so every module's prints land in `logs/kiki.log`. |
| 45–74 | Eager imports of every subsystem (each pre-initializes at import for latency); timing print. |
| 76–87 | `active_tts_streamer` global (so the hotword thread can abort playback); `TOOL_FILLERS` config (canned spoken lines for silent tool calls). |
| `set_motor_relay` @94 | Fire-and-forget relay on/off via KikiController in a daemon thread — must never block the wake path (it used to block ~5s when the controller was down). |
| `stt_stream_worker` @120 | Bridges the blocking STT generator (thread) into the asyncio `stt_queue`. |
| `main()` @133 | Everything below is inside this coroutine. |
| 136–180 | Config load; system prompt + `get_tts_system_prompt_note()` (voice-tag restriction appended **before** warmup so the cached prefix matches); state vars (`summarizing`, `turn_counter`, `last_workers_context`, peeping/vision config). |
| 180–240 | Context build: KB summary (max 50 lines), latest conversation file, skills summary → `additional_context` → `message_history = [sys_prompt, sys_context]`. Note msg[0] content is a list-of-parts (`{"type":"text",...}`); msg[1] is a plain string. |
| `load_past_summary_bg` @254 | Loads the past-conversations summary with `prefer_cache=True` (0.002s from cache) and splices it into `message_history[1]` **before** warmup awaits it. |
| `refresh_past_summary_cache_bg` @284 | Sleeps 180s, then regenerates the cache with `force_refresh=True` for the NEXT startup; retries (box may be hot); never touches the live context. |
| 305–333 | STT + ThinkingSoundPlayer init, `stt.mute()`, HotwordRecognizer init (device_index from `stt.device_index`). |
| `mute_stt`/`reset_mute_timer`/`cancel_mute_timer` @334–356 | 15-second silence timer → back to hotword-only mode (and relay off). |
| `open_listen_window`/`extend_listen_window` | The 15 s timer is reset by the STT `interim` heartbeat so a long sentence isn't cut off mid-thought — but in a crowded room that heartbeat never stops, so the timer was reset forever and Kiki listened indefinitely. `open_listen_window()` stamps a fresh budget on real progress (wake word, IR hold release, committed `final`, follow-up unmute); `extend_listen_window()` is what `interim` calls and stops resetting the timer past `stt.max_listen_window_s` (45 s, `0`/absent disables the cap), letting the ordinary silence timer close the window. |
| `is_idle_mind_active` @359 | Late-bound checker (idle_mgr is created further down) used by vision/peeping loops. |
| `hotword_thread_func` @363 | THE wake path. Order is load-bearing: **unmute STT first**, then relay, then `preempt_background()` (non-blocking), then idle interrupt/mark_activity. Also handles `stop_music`/`stop_it` → `pkill mpv` + `active_tts_streamer.abort()`. |
| Camera control gestures | The existing controller SUB task routes debounced `hand_gesture` events directly to `handle_hand_gesture`: `mute` toggles LCD-only output, `open_palm` aborts current output/generation/activity, `peace` uses the established idle cleanup path, and `thumbs_up` ends listening (clears any IR hold, then `stt.commit_now()` → immediate `final`+`endpoint`). No gesture polling or inference runs in the speaking loop. |
| Always-listen integration | Idle STT events enter the capture-only ambient buffer. Wake/IR switches STT directly to query mode; Unified Idle Mind interprets buffered snapshots later. |
| 403–414 | VisionHandler, WorkerManager (+ scheduler), face/vision history buffers, `fire_event("startup")`. |
| `warmup_when_context_ready` @432 | Awaits past-summary task (≤180s) → bakes the one pending next-turn note into the prefix → single `llm_warmup(message_history)` in executor. |
| 469–477 | UnifiedIdleMindManager created + monitor started; `face_event_listener` task. |
| `periodic_question_loop` @480 | On its interval, vision asks for a source-grounded proactive prompt. Only a meaningful live scene and the one active next-turn note qualify; otherwise no autonomous event is queued. |
| `peeping_loop` @510 | Every `peeping.interval_seconds` (1200): unmute 10s, collect ambient speech, inject as a `[Peeping…]` system message. Skips during Unified Idle Mind / active conversation. |
| 545–560 | STT thread start; `collected_sentences`, `available_tools`. |
| Main loop 561–... | Pulls `(event, text)` from `stt_queue` with 0.5s timeout. `"final"` → collect (or route to peep buffer); idle interrupt/mark_activity; reset mute timer. |
| `endpoint`/`autonomous_vision`/`face_wake` branch | Cancel mute timer → `preempt_background()` → mute mic → `recognizer.pause()` (suspends wake-word so Kiki can't hear itself) → `sfx.start()` (one filler wav). |
| Time injection @~605 | Every ≥5 min appends ONE `Right now it's {time}` system message; workers context appended only when it **changed** (`last_workers_context`). Append-only — see §4 rule 1. |
| `autonomous_vision` branch @~635 | Injects the vision context + `periodic_question_instruction` as a system message; no user message. |
| normal branch @~650 | User utterance assembled; pending vision context injected; in-conversation question instruction (same 300–500s timer); user message appended. |
| Injections @~700 | The one Unified Idle Mind next-turn note is appended only when new and prewarmed off the speaking path. |
| TTS setup @~718 | `TTSStreamer()` factory → start; `stop_sfx_on_first_play` thread waits on `first_play_event` to kill the filler sound exactly when real speech begins. |
| `llm_and_tts` @739 | Runs `stream_response` in an executor: `"sentence"` → extract `<movement>` tags → `add_sentence(clean)`; `"tool_calls"` → start `run_tools_bg` thread immediately + speak a canned filler if nothing spoken yet; `"done"` → strip `<tool_call>` XML, final `full_response`. |
| Tool rounds @~2096 | `while pending_tool_calls:` — wait for `tool_result_ready` (abort-aware, bounded), append the **verbatim generation that requested the tool** (with `<tool_call>`) as an assistant message, then the result as a `tool_result_note()` system note, then `followup_llm_and_tts()` streams the answer onto the SAME streamer (filler + answer play back-to-back). The pre-tool assistant append keeps history aligned with the box's KV cache (rule 2). A `<tool_call>` in the follow-up starts another round — see §5.1a. |
| 845–860 | `tts_streamer.finish()` blocks until playback drains; `finally:` always `recognizer.request_resume()` (resumes only once the room is acoustically quiet). |
| Post-turn @~862 | Append the assistant message **verbatim** (`raw_first`/`raw_followup`, tags included — KV-cache rule 2) while keeping a tag-stripped `clean_response` for logging/notes → apply any pending mode prompt change → **`register_history(message_history)`** (explicit rewarm when the prompt changed) → execute movements in a thread → `idle_mgr.note_turn(user, assistant)` → maybe vision update → `fire_event("after_response")`. |
| Auto-summarize @~915 | `token_counter.count_tokens` vs `agent.token_limit` (6000). Over limit → background `summarize_task`: summary on the local box (abortable), save, then rebuild history as [sys_prompt, summary] **mutating in place** (`message_history[:] = new_history` — other modules hold references) and `register_history` to pre-warm the new short prefix. |
| Follow-up @~965 | `should_skip_followup()` (music) or runtime follow-ups disabled → straight back to hotword mode; otherwise wait 0.35s, drain stale queue events, unmute, arm the 15s timer. |
| `finally:` @~985 | Cancel tasks, `idle_mgr.stop()`, preempt, `fire_event("shutdown")`, stop scheduler, generate + save the session summary (Ctrl+C again skips to raw save), `stt.stop()`. |

### 5.1a Three ways one turn lost a tool call (2026-07-29)

One live session produced all three, and each had a different owner. They are
grouped here because the visible symptom was identical every time: Kiki says
something that sounds like the action happened, and nothing happened.

**1. The schema gate rejected a call the handler would have run.**
`tools.validate_tool_arguments` type-checked with bare `isinstance`, so
`adjust_volume({"action":"set","amount":"60"})` — a JSON *string* — was refused
with `amount must be integer`, even though `adjust_volume` opens with
`int(amount)`. The rejection returns *before* `get_recorder().span(...)` in
`execute_tool`, so a refused call leaves **no tool event in `events.jsonl`** at
all while `kiki.log` still prints `[Tool] Executing` — that asymmetry is how you
identify one. The gate now **coerces scalars in place** (`_coerce_scalar`) for
lossless string→int/number/bool conversions only: `"60"` and `"60.0"` pass,
`"sixty"` and `"60%"` still fail, and **enums are never coerced** because a
category outside the allowed set is a real mistake whose error text is what
teaches the model the right value.

*Why the model quoted it:* `_get_tools_instruction` printed signatures without
types and its only format example was `{"param_name": "value"}` — every value
the model had ever been shown there was a quoted string. Non-string params now
carry a short type (`amount?:int`) and the example includes one unquoted number.

**2. The model's own retry was parsed, logged and dropped.**
The follow-up generation (the one that turns a tool result into speech) can emit
a `<tool_call>` — here it re-sent `adjust_volume` with a correct unquoted `60`.
`followup_llm_and_tts` handled only `sentence`/`done`, so the fix was discarded:
Kiki said *"Wait, did I mess that up? Let me try that again properly"* and the
volume never moved. The tool block in main.py is now a `while` loop over tool
**rounds**, bounded by `llm.tool_calling.max_followup_tool_rounds` (1). Each
round appends the requesting generation verbatim before its result note, so the
history order still mirrors the box's KV cache (§4 rule 2). The bridge connector
is round-0 only (a second one mid-answer sounds like a stutter). Raising the
bound to chain tools is the wrong instinct — that is `complex_query`'s job
(§5.2c) — and every extra round holds the turn open, which holds the wake word
closed.

**3. `tool_result_note` taught the next turn to answer without calling.**
`704acd8` softened this note to *"Answer in YOUR OWN VOICE, fully in character"*
to stop service-desk register flattening a 253-char roleplay prompt. That is
right for a character mode and wrong for `default`, whose own 7.9k prompt
already carries the voice: the note is stored in history, so the prose answer it
produced became the nearest precedent, and two identical repeat requests were
answered *"Playing Maafi again"* with no tool call either time. The note now
splits on `runtime_controls.mode_has_own_character()` — the in-character wording
for modes that declare their own `system_prompt`, the directive wording for
`default` — and **both** forms end by saying a repeat request needs its own call.
The `complex_query` form is unchanged.

Diagnostic note: `⚠️ Tool execution timed out after 15.0s` used to print for an
IR/gesture abort too (`_wait_tool_or_abort` returns `False` for both), so a 1.7s
open-palm barge-in read as a slow tool. The two exits now log differently.

Regression tests: `tests/test_tool_call_recovery.py`.

### 5.2a `core/vision/instant_vision.py` — Live-image path (Groq Qwen VLM)

The local speaking box is **blind** on the speaking path (image parts are dropped by
`_normalize_messages_for_local`). So a genuinely visual question — "does my shirt look
good?", "look at this phone, should I buy it?", "describe what you see" — is routed to
Groq's multimodal `qwen/qwen3.6-27b`: a fresh camera frame is captured and the answer is
**streamed** to TTS, box untouched. Config: `llm.instant_vision`.

**Two triggers, one Groq path:**
1. **Regex fast-path** (`is_instant_image_query`) — a high-precision regex on the user
   utterance. When it matches, `stream_response` routes to Groq BEFORE touching the box →
   lowest latency (no box round-trip). Keyed on visual verbs / "what do you see" /
   appearance judgements / "should I buy THIS" / "read this".
2. **`look_at_scene` tool call** (the smart-model safety net for regex misses) — the tool
   is in the speaking model's catalog (`llm.main_tools`, schema+no-op handler in `tools.py`);
   when the local model recognises a vision question mid-stream it emits
   `<tool_call>{"name":"look_at_scene"}</tool_call>`. `_stream_local_inner` intercepts it
   (sets `vision_switch["on"]`, ends the local stream with NO "done"), and `stream_response`
   hands the turn to Groq — **transparently**, so `main.py`/`llm_and_tts` just see sentences +
   done like any other turn (the still-playing thinking sound covers the box→Groq gap).

- **8K TPM cap**: qwen3.6-27b is rate-limited to 8K tokens/min, so `build_capped_messages`
  keeps the whole request under it — personality system prompt + current question (with the
  image attached as a `data:image/jpeg;base64` url) are always kept; older history/summary is
  added newest-first only while it fits `max_context_tokens` (5000); the reply is bounded by
  `max_completion_tokens` (900). Text tokens are counted via `token_counter`.
- **Key source**: `GROQ_API_KEY_LIST` in `.env` (JSON array, tried in order → a bad/throttled
  key rolls to the next; falls back to single `GROQ_API_KEY`). Same pool `generate_llm_resp` uses.
  The keys are **separate orgs with independent 8K TPM budgets**, so rotation genuinely
  multiplies the budget — rotating is the recovery, never waiting.
- **`max_retries=0` is load-bearing** (measured). The Groq SDK defaults to **2 retries and
  SLEEPS for the server's `retry-after` on a 429** — observed 11s, 18s, 40s. That sleep, on the
  same already-exhausted key, was a real **50s time-to-first-word** in production. Failing fast
  + rotating gives 1-2s under the same conditions, and ~3s to exhaust every key and fall back to
  the local box. 401 keys are remembered in `_DEAD_KEYS` and skipped for the session; a 20s
  request timeout prevents a stalled socket from hanging the turn.
- **Token budget reality** (measured, not estimated): the **image costs a flat 786 prompt tokens
  at any resolution** — Qwen normalizes it, so downscaling saves upload time but *zero* tokens.
  The **text context is what exhausts the 8K/min cap**, which is why `max_context_tokens` is
  small (1800) here: the current question + personality is enough to answer "what do you see".
- **Reasoning**: this model rejects `reasoning_effort` of `low`/`high` (only `none`/`default`);
  `none` disables thinking for lowest latency. `<think>…</think>` spans are still stripped from
  the stream defensively (`_think_filter`, tag-split-safe across SSE deltas). `_create_stream`
  retries without optional kwargs on a 400, but re-raises auth/rate errors so key rotation fires.
- **Wiring** (`core/llm.py`): the regex fast-path fires BEFORE the local path; the tool-call
  path fires from WITHIN the local stream (`vision_switch`). Both only on the PRIMARY turn
  (`verify_prefill and not use_fallback and not local_only`). On any pre-first-token failure
  (no frame / all keys down) the regex path raises `_InstantVisionUnavailable` and falls through
  to the (blind) local path; the tool-call path speaks a short "can't see clearly" line.
  `abort_event` (IR barge-in) cuts the Groq stream mid-flight.
- **Speculative turns**: a spec pre-gen runs blind on the box (`local_only`). If the model emits
  `look_at_scene` there, `stream_response` yields a `("vision_requested", None)` event (it does
  NOT fire the camera/Groq speculatively); `SpeculativeTurn` records `result["vision"]=True` and
  main.py **refuses to adopt** it (also refuses when the regex flags the utterance) — the real
  turn re-runs and routes to Groq.
- **KV-cache correctness**: a Groq turn never touches the box, so `last_turn_used_instant_vision()`
  tells main.py to do a **real** rewarm afterwards (`after_speaking=False`) instead of the usual
  `--prefill-after-response` shortcut (which would mark a stale prefix "warm"). The image is never
  stored in `message_history` — only the text reply — and for a tool-call turn the model's
  lead-in + `<tool_call>` text is NOT stored either (only the Groq answer), so the real rewarm
  re-prefills the correct prefix and the next local turn stays a cache hit.

### 5.2b Speaking provider toggle — `llm.speaking_provider` (`local` | `cerebras` | `groq` | `openrouter`)

**`cerebras` — the current default.** `gemma-4-31b` on Cerebras' OpenAI-compatible endpoint
DIRECTLY (`_stream_cerebras_speaking`), streamed with raw `requests` SSE over the shared
keep-alive session — the same lean path as `_stream_local_inner`, no SDK layer. **Measured
0.56–0.89s to first token**, whole reply in one burst. Going direct is what made this viable:
the same model *via OpenRouter* was stuck behind a shared pool returning 429 every ~60s.

**Image-cost guard (`llm.cerebras_speaking`)**: images are NEVER sent on ordinary turns —
`_normalize_messages_for_local` drops image parts, and periodic scene context arrives as
Gemini-written TEXT. Pictures leave only on an explicit `look_at_scene` tool call, served by
`instant_vision` on Groq. Each response prints `prompt/completion/image` token counts, so
`image=0` is verifiable per turn rather than assumed. Combined with speculative turns being
off for cloud providers, **one spoken turn == exactly one billed request.**


All cloud providers share ONE protocol implementation, `_scan_cloud_deltas` — it turns a raw
content-delta iterator into the same `sentence`/`tool_calls`/`done` events the local path emits,
including the `<tool_call>` scanner, the `look_at_scene` vision switch, and the eager
first-clause flush. A provider function only has to produce deltas.

**`openrouter`** — `google/gemma-4-31b-it` pinned to **Cerebras** (`_stream_openrouter_speaking`).
Measured **0.91–1.74s to first token at 260–1700 tok/s** (the reply effectively lands at once),
with no TPM squeeze (131K ctx) so the conversation is sent uncapped. **Blocked on BYOK**: the
shared Cerebras pool (`is_byok:false`) returns 429 with `retry_after 59s` after ~one request —
about 1 turn/min, unusable for conversation. Adding a personal Cerebras key at
`openrouter.ai/settings/integrations` removes that ceiling and makes this the fastest option.
`allow_fallbacks:false` is deliberate — failing fast to the warm box (~1.5s) beats silently
landing on SiliconFlow/Novita (measured 7–8s TTFT). Key: `OPENROUTER_API_KEY` in `.env`.


One config value picks the brain that generates spoken replies:
- **`local`** (default) — the llama.cpp box (`_stream_local`): KV-cache warmed, speculative
  turns, ~1-2s warm TTFT, unlimited context.
- **`groq`** — `_stream_groq_speaking` streams every reply from Groq's Qwen instead. It reuses
  `_normalize_messages_for_local` (so the SAME `<tool_call>` protocol + tools instruction),
  caps context for the model's TPM budget (`groq_speaking.max_context_tokens`), and emits the
  SAME `("sentence"|"tool_calls"|"done")` events + `vision_switch` handling — so main.py's tool
  execution, follow-ups, history, and **speculative turns** all work unchanged. No box ⇒ no
  KV cache to manage on the hot path, and latency is pure network+generation — **measured
  0.75-1.3s to first sentence** at conversational pacing, including turns that hit a 429.

**What makes Groq mode actually fast (all three are load-bearing):**
1. **Key-pool capacity.** 8K TPM is *per key*, and each key in `GROQ_API_KEY_LIST` is a separate
   org, so 4 valid keys = 32K TPM ≈ 5.5 turns/min at this repo's ~5800-token context. Rotation
   is round-robin with per-key 429 cooldowns (`_ordered_keys`), so the first key tried almost
   always has budget instead of re-probing a drained one.
2. **Speculative turns are gated on `speaking_is_local()`** — they run on the box ONLY, never on
   any cloud provider. A spec turn is a full duplicate generation: free on the box's own slot,
   but on Groq/OpenRouter it doubles spend against a per-minute budget, and the discarded spec
   is exactly what pushes the REAL turn into a 429. Same gate disables `prefill_partial`.
3. **The box is kept warm as a HOT STANDBY.** `warmup`/`register_history`/the instant-vision
   pre-warm all still run under Groq (the box is idle then, so it costs nothing, and
   `--cache-prompt` makes each post-turn rewarm incremental). This is not cosmetic: when every
   key is in cooldown we fall back to the box, and an *unwarmed* box paid a **45s cold prefill**
   (~5900 tokens at ~13 tok/s) in testing. main.py must pass `after_speaking=False` for these
   turns (`groq_turn`) — the box never generated the reply, so there is no server-side prefill
   to defer to.

**Remaining caveat**: a general chat model **over-calls tools** vs the tuned local gemma

### 5.2c `core/brain/action_agent.py` + `fast_cloud.py` — the `complex_query` action agent

The speaking model emits **one** tool call per turn, so "check the messages at project circle
and set a reminder if there's an event tomorrow" was structurally impossible. This is the
same two-trigger shape that makes `look_at_scene` seamless (§5.2a), applied to multi-step
work: a specialised cloud handler behind a normal tool call, returning into the ordinary
tool-follow-up machinery. `main.py` needs no knowledge that it ran.

**The 12 WhatsApp tools were REMOVED from `llm.main_tools`** (and senior's override); they
remain in `tools.TOOLS` for agents. The speaking tools instruction went 2186 → 2044 chars.

**Two triggers, one path** (both land on the `complex_query` tool):
1. **Code-level router** — `_should_route_complex_query` / `_auto_complex_query_tool_event`
   in `core/llm.py`, modelled on `_auto_memory_tool_event`. Fires in **0.00s** (no box
   round-trip). This is **load-bearing**: with WhatsApp gone from the catalog, a missed
   model decision leaves no tool at all, and the failure mode is a confident *"sure, I sent
   it!"* for a message that never left. Negatives (`play`, `song`, `what did we discuss`,
   `switch mode`, …) win outright — a false positive would cost every normal turn seconds.
2. **Model-emitted** — `complex_query` is in the catalog for everything the regex misses.

**Provider split (`action_agent` config).** Cerebras gets the **FULL** context; Groq gets a
**compacted** one. Measured on the Pi with `tests/bench_action_agent.py`, 4-turn loop:

| route | loop | notes |
|---|---|---|
| **cerebras `gemma-4-31b`** (default) | **3.07s** | one clean JSON object per turn, 0 rate limits, reports `cached_tokens` |
| cerebras `gpt-oss-120b` | 2.88s | fabricates follow-on turns + its own tool results, ~3× output tokens |
| groq `openai/gpt-oss-120b` | 5.83s | **429 on turn 4 of ONE query** |
| groq `qwen/qwen3.6-27b` (fallback) | 5.92s | 429 on turn 4 |

**Groq's 8000 TPM is per key** (confirmed via `x-ratelimit-limit-tokens`; a token bucket
refilling in ~24s, not a daily cap) — one complex query is roughly one whole key's minute,
and the pool is **shared with `instant_vision`/`look_at_scene`**. That, not latency, is why
Cerebras is the default. End-to-end live runs land at **2.9–6.9s**.

**Four guards exist because live testing produced these exact failures** — all four are
about never letting a non-action sound like a completed one:
- `fast_cloud.first_json_object` — `gpt-oss` emits its tool call, then *fabricates the
  result*, then continues. Only the first balanced object is kept.
- `action_agent._placeholder_arg` — the model batched `list_messages(chat_jid=
  "<PLACEHOLDER_JID_FROM_FIRST_CALL>")` in the same turn as the `list_chats` meant to
  supply it; the empty result became "there are no new messages". Refused, not executed.
- `min_tool_calls=1` — a run that touches no tool has invented its answer (observed: a
  detailed WhatsApp conversation about taco night and guacamole that did not exist).
- `_is_meaningful` — a `"..."` summary is not a success.

On failure or deadline the returned string explicitly tells the follow-up model **not** to
claim success. The agent's `deadline_seconds` (22) sits below
`llm.tool_calling.exec_timeout_overrides.complex_query` (26) so it reports partial progress
itself rather than being truncated by main.py. `run_agent_loop`'s existing `progress_fn`
drives the LCD/OLED line. Cloud-only — the local speaking slot is never touched.

**Related fixes made for this feature:**
- `whatsapp.py list_chats` selected `messages.content` without its JOIN when
  `include_last_message=False`, so it silently returned **zero chats**.
- `search_contacts` excludes groups in SQL (`AND jid NOT LIKE '%@g.us'`), so a group name
  could **never** resolve. `whatsapp_mcp._resolve_recipient` now also sweeps `list_chats`
  and fuzzy-scores (`_name_score`): "project cereal" → the real group "project circle club" (0.92).
  An exact match always wins; a genuine tie returns `matches` so Kiki asks instead of
  messaging the wrong person. `tools.list_chats` retries fuzzily when the LIKE finds
  nothing, which saves the agent a wasted turn.

### 5.2d `core/self_extend/whatsapp_contacts.py` — the address book

**`messages.db` has no names.** Measured on the device: **292 direct chats, 0 with a human
name** — every DM is a bare identifier like `915106634622667`. So "send a message to
Marina" could never resolve, and a chat summary read out as a list of phone numbers.

The Go bridge's **own** store (`whatsapp-bridge/store/whatsapp.db`) held the answer all
along, unused until now: `whatsmeow_contacts` (~2400 rows of full/first/push/business
name against a phone JID — the real WhatsApp address book) and `whatsmeow_lid_map`
(~1200 rows mapping the opaque `@lid` message senders back to phone numbers). Chaining
them turns `954785219286295` into "Casey" — verified 8/8 on live group senders.

This module owns that chain in **both** directions, read-only, cached against the store's
(mtime, size) because the bridge syncs contacts continuously:
- `resolve_name(q)` → who to SEND to. Fed into `_candidate_pool` **first**.
- `display_name(id)` → who to READ OUT. `whatsapp_mcp._label_people` walks every MCP
  payload adding `sender_name`, and replaces a numeric DM chat name with the contact.
- `jid_variants(jid)` → **the phone-vs-`@lid` split**: a direct chat is *addressed* by
  phone but *filed* under its `@lid`. "Taylor Studio" resolves to
  `99671885147@s.whatsapp.net` (0 message rows) while its 60 real messages live under
  `97103140899455@lid` — querying only the first made the agent state "there are no
  messages" as fact. `tools.list_messages`/`get_chat` retry the alternate form before
  believing an empty result.

Contacts are stored twice (once per identifier form) so `_load` canonicalizes via the lid
map — otherwise one person looks like two and Kiki asks a needless clarifying question.

**`whatsapp.contacts` in config.json** is the manual override: `{"name": "number"}` for
nicknames WhatsApp doesn't know ("family contact") or to settle a name several contacts share (three
people called Casey). Config entries beat the address book outright in
`_resolve_recipient` — that is the documented way to disambiguate. Keys starting with `_`
are treated as comments, not people. Restart to apply.

Ranking prefers a real **chat** over an address-book-only entry at equal score: most of
those ~2400 names have never been messaged, so an existing conversation is the likelier
target than a namesake in the contact list.

### 5.2e Inline image reading — and the three ways it hung the turn

An image message arrives as `content: ""` with `media_type: "image"` — indistinguishable
from a blank message — so the agent skipped pictures entirely (measured: 18 images in a
chat, 0 reads). `tools._describe_images_inline` now describes the newest few up front
through the **same free Groq qwen VLM `look_at_scene` uses**, so a summary includes what
the pictures say without the agent spending extra turns. Config: `whatsapp.describe_images`
/ `describe_images_limit` (3) / `describe_images_timeout` (5s).

Getting this bounded took three separate fixes, each worth keeping:

1. **Slice before doing per-image work.** The cap was applied *after* checking every
   image row, so an active group did unbounded work *outside* the timeout.
2. **Never fetch cold media inline.** Media that isn't downloaded goes over the network
   and serializes behind the single MCP session lock. A background prefetch thread looked
   free but was worse — it holds that same lock, so the agent's own next call queued
   behind it (a summary went 3.8s → 25.4s). Cold images are simply marked with the exact
   `read_whatsapp_image` call needed; on-disk presence is a cheap stat via the `filename`
   column, no MCP round trip.
3. **Give image work its own threads** (`_IMAGE_POOL`). On a timeout the workers keep
   running until their Groq call returns; on the shared default executor those stragglers
   occupied the same pool the agent uses for its next tool call, so a 5s image cap still
   produced a **108s turn**. Isolating them made the timeout real: 108s → 10.6s.

On timeout the chat is summarised **without** the pictures rather than the turn hanging.

Separately, `execute_tool` used `with ThreadPoolExecutor()`, whose `__exit__` calls
`shutdown(wait=True)` — so a timed-out tool still blocked until its worker finished. It
now shuts down with `wait=False`, or the 30s cap is not a cap at all.

**Answer length**: the agent returned 242 chars because four things capped it — the agent
prompt, `summary_max_chars`, `execute_tool_calls`' 1500-char result cap, and (largest)
main.py telling the model to "answer briefly in one or two spoken sentences". That last
instruction now switches on the tool: `main.tool_result_note` tells the model to relay a
`complex_query` result **completely**, since the agent already wrote a finished spoken
reply rather than raw data to distil.
- New tools: `read_whatsapp_image` (downloads media → `instant_vision.describe_image_file`,
  the same free Groq VLM; ~1.5s) and `record_voice_note` (taps the STT `mic_reader` ring
  buffer via `STTEngine.record_clip` — a second PyAudio stream would fail with "Device or
  resource busy"; works while muted, which is the normal mid-turn state).

### 5.2f Name matching and row size — why summaries were thin and sends were refused

Reported live (2026-07-26): "summarize chats by marinaa" and "send a message to marina"
both still failed. Four separate causes, all now fixed and regression-tested.

**1. Mid-word containment.** Both scorers boosted a plain substring match to ~0.9. But
`"arin"` is literally inside `"marinaa"`, so the contact **Arin** scored 0.90 against the
real **Marina**'s 0.92 — inside `_FUZZY_MARGIN` (0.08), so `_resolve_recipient` declared it
ambiguous and refused to send. A length-ratio guard cannot separate these: `arin/marinaa`
is 0.57 and the legitimate `project circle/project circle club` is 0.58. **Word boundaries can** —
`whatsapp_contacts.contained_at_word_boundary` requires the shorter name's tokens to be a
contiguous run of the longer's. Shared by `whatsapp_contacts._score` and
`whatsapp_mcp._name_score` so the rule cannot drift between the read and send paths.

**2. `chat_name` was never labelled.** `_label_people` de-numbered a chat row's `name` but
not a *message* row's `chat_name`. Every message in Marina's chat therefore read
`"96298496998124"`, the agent concluded it had fetched the wrong chat, and burned a whole
extra turn re-fetching byte-identical messages under the `@lid`.

**3. Fat rows, not a short model.** The 308-char summary was not the model being lazy: the
agent asked for 100 messages and `max_tool_result_chars` handed back 1500 characters —
**five rows** — because each MCP row spends ~300 chars repeating `chat_jid`, `sender` and a
32-char `id`. `tools._compact_message_rows` cuts a row to `{time, from, text}` ≈ 60 chars
(`id`/`chat_jid` survive only on media rows, where `read_whatsapp_image` needs them), and
the budget rose to 4500. **308 → 999 chars of genuinely specific summary.**

**4. A wasted turn on every send.** The agent prompt said to resolve recipients with
`search_contacts` first — but `send_message`/`send_file`/`send_audio_message` already pass
`resolve_recipient=True` and do the fuzzy address-book resolution themselves. The lookup
turn was pure latency, and its result was *worse* than the resolver's. **11.5s → 7.6s.**

Measured after: `marinaa`→Marina, `taylor studio`→Taylor Studio, `project cereal`→the
project circle group, `jordan`→Jordan Lee; `taylor` alone stays ambiguous (three real Taylor entries),
which is correct. Summaries 9.3s/999 chars and 12.9s/685 chars (the latter reads 2 images).

### 5.2g What the agent knows besides the request

The code router synthesises its tool call from the user's **words alone**, so until
2026-07-27 the agent received `"summarize my chat with him"` with no referent whatsoever
and no idea who Kiki or Alex are. It now gets two things, both assembled in
`action_agent._background()`:

- **`llm.conversation_snapshot(turns, chars)`** — recent spoken turns as `Alex:`/`Kiki:`
  lines, trimmed **from the front** so the newest turns (the ones a pronoun points at)
  always survive. Fed by `llm._note_conversation(messages)`, called once per primary turn.
- **`llm.persona_brief(chars)`** — the *identity opening* of `llm.system_prompt`, clipped at
  a sentence boundary. Not the whole 7.9k prompt: the rest is behavioural rules for
  free-form conversation that would only distract a model whose job this turn is calling
  tools correctly. The summary is re-voiced by the speaking model, which still has all of it.

**Why a module snapshot and not a tool argument** — this is the load-bearing detail. A tool
call's `arguments` become assistant text that `register_history` writes into the warm
speaking prefix. Putting the conversation in there would rewrite that prefix on **every**
routed turn and force a full re-prefill, breaking the §4 cache contract. Reading it
out-of-band costs the speaking path one list comprehension and zero prompt bytes.
`test_context_never_enters_the_tool_call_arguments` guards this.

Verified: with "taylor studio has been messaging me all week" in history, **"summarize my
chat with him"** picks Taylor Studio out of three Taylor entries — 9.4s, 974 chars.

### 5.2h `core/brain/history_view.py` — the history, rendered for readers

Traced from one question: *"if I asked a while ago to play music, when will 'send the
music link' work?"* Answer: **never.** `play_music` returns `"Now playing X - <url>"`, and
main.py files every tool result under `role: "system"`. Kiki only *speaks* the follow-up
line, which has no URL. Both readers of the history dropped that role, so the link existed
only in the one row neither of them read — and vanished for good at the next compaction.

One tool turn writes **four** rows:

```
user      "play some music"
assistant '<tool_call>{"name":"play_music",...}</tool_call>'     ← protocol, not speech
system    'Here is the result ... :\nNow playing X - https://...'  ← the link lives HERE
assistant "Playing X for you."
```

`history_view` renders that once, correctly, and is **shared by both consumers** so they
cannot drift apart again:

- `render()` → typed records: `user`, `kiki`, `tool_call`, `tool_result`, `memory`,
  `time`, `context`. The `<tool_call>` tag becomes `[Kiki used play_music(song="…")]` —
  previously it was passed through verbatim and read as something Kiki said out loud.
- `as_text(max_chars)` → trimmed **from the front**; the newest records are what "it",
  "him" and "that link" point at.
- `harvest_artifacts()` → URLs, file paths and jids with a little surrounding text
  (`https://… (Now playing Blinding Lights -)`). The durable half: even when a long result
  is clipped, the link a follow-up needs survives.

**Budget**: `action_agent.history_chars` = 28000 (~7000 tokens), deliberately matched to
`agent.token_limit` — the ceiling the *local* model runs against. The Pi's box model was
seeing the entire `message_history`, tool results and all, while the cloud agent got 1800
chars of user/assistant text: **the 3B-class local model was ~14× better informed than the
agent built to act on what it heard.** `max_prompt_chars` rose 22000 → 60000 so history and
tool results stop competing (what `_compact_conversation` evicted was the results).

**Summariser** (`main.build_summary_input`): the same fix. It kept user/assistant rows plus
`[TIME]` anchors and `continue`d past everything else, so nothing Kiki learned by *calling*
a tool was ever written into long-term memory — it survived verbatim until the token limit
tripped, then disappeared. That is why "a while ago" failed as a **cliff, not a fade**. It
now carries `[TOOL RESULT]` rows and the previous `[EARLIER MEMORY]` block forward (without
that, each summary covered only what happened since the last one and memory reset at every
compaction instead of accumulating).

Verified: with the music turn buried under 8 later exchanges, **"send the music link to
… on whatsapp"** → one `send_message` call with the correct URL, **3.3s**.

### 5.2i Address-book staleness — a 4 ms hole in the cache signature

`whatsapp_contacts` cached against `(st_mtime, st_size)`. **Measured on this Pi**: inode
timestamps advance in **4 ms** steps (kernel jiffies, CONFIG_HZ=250), and a small INSERT
into the 3 MB store reuses a free page so `st_size` never moves. A bridge contact-sync
landing inside that tick is invisible to the signature — and stays invisible, because
nothing later changes it either. A contact added at that instant would **never** be
findable: exactly the "send a message to someone I just added" failure. Reproduced 2 runs
in 6. Fixed with `_CACHE_TTL_SECONDS = 30`; a full reload is ~0.05s for 2434 contacts, so
the sweep costs nothing and bounds staleness. The reload log only fires when the entry
count actually changes, or the TTL would print every 30s forever.

### 5.2 `core/llm.py` (620 lines) — Speaking path

- **Import-time** (1–102): loads `.env`, reads the Vertex service-account key json (`_VERTEX_SA_FILE` → `vertex_credentials_json`; auth matches `apiusage.py`),
  caches `llm` config, **lazy litellm** (`_get_completion` @33 — importing litellm eagerly cost
  seconds on the Pi). Constants: `_USE_LOCAL`, `_LOCAL_URL`, `_MAIN_TOOL_NAMES` (the curated
  tool subset exposed to the fast model), `_SENTENCE_RE` (split after `.!?`), `_THINK_RE` +
  `_scrub_reasoning` @92 (defensive scrub of leaked reasoning markers before TTS),
  `_CLAUSE_RE` + `_FIRST_FLUSH_MIN_CHARS=18` (eager first-clause flush so TTS starts sooner).
- `_extract_sentences` @105 — splits buffer into complete sentences + remainder.
  Hindi Devanagari (`।॥`) terminators flush immediately at the current end of the
  stream, so Hindi replies do not accumulate into one large TTS request; Latin (`.!?`)
  retains its whitespace guard to avoid splitting streamed decimals/abbreviations.
- `_get_tools_instruction` @119 — builds the `<tool_call>{json}</tool_call>` protocol text from
  `_MAIN_TOOLS` schemas; injected into the FIRST system message by the normalizer (stable
  across turns → cache-safe).
- `_normalize_messages_for_local` @150 — **the canonical normalization**: flattens
  list-of-parts content to plain strings, drops `tool` role messages and empty content,
  injects the tools instruction. Byte-for-byte stability of its output across turns is what
  makes `--cache-prompt` hit. Anything that builds a prefix for the box must go through it.
- `warmup` @197 — registers the normalized prefix + synchronous `rewarm()`; called once at
  startup by `warmup_when_context_ready`.
- `_stream_local` @228 — speaking entry: `note_user_activity()` (marks hot) →
  `note_speaking(True)` → `preempt_background()` → delegates to `_stream_local_inner`;
  `finally:` only `note_speaking(False)`. **Deliberately does NOT register the prefix** (§4 rule 2).
- `register_history` @257 — public; normalize live history → `update_speaking_prefix` →
  `schedule_rewarm`. Called by main.py after the assistant append and after summarization.
- `_stream_local_inner` @278 — raw SSE loop: posts with `cache_prompt: true` and
  `thinking_budget_tokens: 0` (REQUIRED — without it gemma thinks on every voice turn since
  the server runs without `--reasoning-budget`). Char-by-char scanner detects
  `<tool_call>...</tool_call>` mid-stream → yields `("tool_calls", ...)`; otherwise eager
  first-clause flush then sentence extraction → `("sentence", s)`; ends with `("done", full)`.
  Raises `_LocalUnavailable` only if the connection fails before any token (clean fallback).
- `execute_tool_calls` @417 — runs parsed calls via `tools.execute_tool`, caps each result at
  1500 chars (keeps the follow-up prefill small).
- `stream_response` @455 — top-level: local path first; `_LocalUnavailable` → cloud fallback
  via litellm (`_FALLBACK_MODEL`), with a content-timeout fallback chain, native tool_calls
  handling and recursive follow-up. The cloud path is the emergency path only.

### 5.3 `core/local_llm.py` (491 lines) — Single-slot coordinator

- **Header** (1–60): module docstring documenting the abort/rewarm design; config constants:
  `BASE/URL`, parsed `_HOST/_PORT/_PATH` for the raw-socket path, `_REWARM_MAX_TOKENS=1`
  (prefill-only — generated tokens during a rewarm would themselves diverge the cache),
  `REASONING_OVERRIDE_FIELDS` (`chat_template_kwargs.enable_thinking`), thinking-block regexes,
  shared keep-alive `SESSION`.
- **Coordination state** (70–105): `_bg_lock` (serializes background tasks), `_bg_abort`,
  `_bg_active`, `_bg_current["conn"]` — **the in-flight `http.client` connection**, whose
  socket exists from the moment of POST (unlike a `requests.Response`, which only exists
  after headers = after the whole prefill). `_PROTECT_S` + `_last_user_activity` implement
  the conversation-hot window.
- `note_user_activity` @106 / `conversation_hot` @111.
- `preempt_background` @115 — sets abort, then **socket `shutdown(SHUT_RDWR)` in a throwaway
  daemon thread** (never blocks the caller — a blocking close once delayed mic unmute 7.6s).
  llama-server frees the slot on disconnect even mid-prefill.
- `_warm_prefix_hash` + `_prefix_hash` @150 — md5 of the normalized prefix currently warm in
  the box's KV cache; lets rewarms be skipped when redundant. Cleared by: any non-rewarm
  background request, `note_speaking(True)` (generated reply tokens diverge the cache).
- `update_speaking_prefix` @159 / `note_speaking` @166.
- `rewarm` @179 — skip if hash matches, else `generate_background(..., is_rewarm=True,
  rewarm_hash=h)` with max_tokens=1; hash recorded on success (`_last_bg_failed` False).
- `schedule_rewarm`/`_schedule_rewarm` @202 — fire-and-forget, coalesced
  (`_rewarm_scheduled`), skipped while speaking or when no prefix registered.
- `strip_thinking*` @228 — removes closed AND unclosed inline thinking blocks.
- `post_stream` @244 — requests-based streaming POST (used by tests/spec paths; speaking has
  its own in llm.py). Always sends `thinking_budget_tokens` (0 default).
- `iter_sse` @271 (requests) / `_iter_sse_httpclient` @293 (readline-based for http.client).
- `generate_background` @311 — THE background entry point. Order of operations inside:
  1. refuse if `conversation_hot()` and not a rewarm (→ caller falls back to cloud);
  2. build messages (prompt string or full list; optional `image_b64` → mmproj vision part);
  3. under `_bg_lock`: re-check `rewarm_hash` (two queued rewarms race the pre-lock check);
     clear abort; mark active; reset `_last_bg_failed`;
  4. open `http.client.HTTPConnection`, store in `_bg_current` **before** posting,
     invalidate `_warm_prefix_hash` for non-rewarms, POST, `getresponse()` (blocks during
     prefill — abortable because the socket is already exposed);
  5. SSE loop collecting `delta.content` and `delta.reasoning_content` separately, with a
     client-side thinking cap (fallback if the server budget isn't enforced);
  6. inline-thinking extraction; **salvage**: if the model spent all tokens thinking and
     produced no answer, return the thinking text for distillation instead of None;
  7. `finally` (inner): record warm hash on rewarm success, close conn;
     `finally` (outer, **outside the lock**): `_schedule_rewarm()` for non-rewarms.

### 5.4 `core/stt.py` (~430 lines) — Local Whisper.cpp STT (client-side Silero VAD)

Optimized pipeline ported from the standalone `whisper_tts.py` benchmark: per-frame
**client-side** Silero VAD endpointing + **speculative finalization** drops end→text
from ~2s to ~200–300ms. Replaces the old "re-transcribe a growing rolling buffer every
0.4s + RMS-silence endpoint" design. **Same event API** (drop-in for main.py / vision /
face handlers).

- `STTEngine.__init__` — config: device 2, **16 kHz mono, fixed 512-sample (32 ms)
  frames** (the window Silero requires), whisper server at `stt.whisper_url`. New
  endpointer knobs (all in `_ep_cfg`, defaults shown): `vad_threshold` 0.5,
  `endpoint_ms` 200 (trailing silence that commits), `spec_silence_ms` 80 (silence after
  which the speculative ASR fires), `min_speech_ms` (reuses `vad_min_speech_ms`, default
  150), `preroll_ms` 250, `max_utterance_seconds` 20. `_muted` event controls behavior.
- `_AsrClient` — small `ThreadPoolExecutor` (2 workers) + keep-alive `Session`; one
  whisper request per utterance with short-clip params (`vad:false` — client already did
  VAD, `single_segment`, `no_timestamps`, `audio_ctx` sized to the clip via
  `_compute_audio_ctx`). Read timeout scales with clip length (`request_timeout` floor →
  `max_request_timeout` cap) so a 20s monologue doesn't spuriously time out.
- `_Endpointer` — per-frame Silero state machine with a pre-roll ring buffer (so the
  first phoneme isn't clipped). On trailing silence ≥ `spec_silence_ms` it fires the
  final ASR **in the background** and keeps listening; at ≥ `endpoint_ms` it commits using
  that in-flight result (critical path = max(endpoint_ms, asr_time)). Speech resuming
  discards stale speculation; utterances < `min_speech_ms` are dropped as noise (no
  event); a pause-less monologue is force-committed at `max_utterance_seconds`.
  A frame counts as voiced only if Silero says speech **and** `NearFieldGate` says
  near-field (see 5.4c) — or a push-to-talk hold is active, which bypasses the gate.
- `commit_now()` — explicit "I'm done talking" (thumbs-up gesture): clears the hold and
  sets the force-commit flag so the VAD worker ends the utterance on its next iteration
  rather than waiting for trailing silence that a noisy room may never produce.
- `mute`/`unmute` — instant flag flips (no reconnect). On unmute a **flush** is requested
  FIRST (discards buffered frames + resets the endpointer) so TTS-era audio isn't echoed
  back. While muted the mic reader keeps draining ALSA but frames are dropped — zero-latency unmute.
- `stream()` — generator; starts two threads: **mic_reader** (continuous 512-frame reads
  → frame queue) and **vad_worker** (runs the endpointer, fires speculative ASR, pushes
  events). Emits `("interim","…")` on speech onset and ~every 1s while speaking (keeps
  the LCD live + resets main.py's 15s mute timer on long utterances), then `("final", text)`
  (never None) + `("endpoint", None)` at commit. A slow/dead server never blocks capture.
- `stop()` — closes stream/PyAudio + ASR pool/session.
- `set_capture_mode("ambient"|"query")` — resets the current VAD boundary without draining
  newly queued frames. Query events retain the existing names; passive commits emit
  `ambient_final`/`ambient_endpoint`. The mode is stamped at speech onset so a slow ambient
  ASR completion cannot race a wake word and become the user's query. Rolling partial ASR
  is disabled in ambient mode (it only exists to prefill an imminent speaking request).

### 5.4c `core/noise_suppression.py` + `core/near_field_gate.py` — noise handling

Two different problems, two different mechanisms. **They are not interchangeable**, which is
the key thing to remember before tuning either one.

**`noise_suppression.py` — RNNoise (steady noise).** A ctypes wrapper over the
`librnnoise.so` bundled with `pyrnnoise` (the package itself is never imported — it drags in
a file/plotting stack the mic path doesn't need). The mic is opened at RNNoise's native
**48 kHz / 480-sample (10 ms)** frames, denoised, then `StreamingDecimator3` resamples to the
16 kHz / 512-sample frames Silero needs via a causal 63-tap anti-aliased filter (<0.7 ms group
delay). All work happens *at capture time*, so nothing is added after endpoint and
time-to-first-word is untouched. Measured on this Pi: **~2.0 ms per 10 ms frame (~21 % of one
core)**. A real-time guard bypasses suppression permanently for the session after
`slow_frame_limit` frames exceed `max_process_ms` (5 ms), so an overloaded denoiser degrades
to raw audio rather than adding latency. If the mic can't do 48 kHz, capture falls back to
16 kHz with suppression off. RNNoise state is reset at each mute/unmute boundary so Kiki's own
TTS never pollutes the noise history. Removes fans, AC, traffic, motors. **Does not remove
background speech** — it is trained to preserve voices.

**`near_field_gate.py` — NearFieldGate (crowds).** The crowd fix. Silero answers "is this
speech?", not "is this speech addressed to Kiki", so in a crowded room every frame reads as
voiced, `silence_ms` never accumulates, no endpoint ever fires, and the 1 s interim heartbeat
keeps main.py's listen window open forever. What actually separates the user from the crowd is
**level**: the person talking to Kiki is near-field and sits well above the room's babble.
`NearFieldGate` tracks the noise floor with an asymmetric envelope follower (falls fast at
`floor_fall_per_s_db`, rises slowly at `floor_rise_per_s_db`, and only *non-speech* frames may
push it up, so the user's own voice cannot gate them out mid-sentence), then requires speech to
stand `open_margin_db` above it. Thresholds are hysteretic — `close_margin_db` to keep
counting — so quiet trailing syllables aren't clipped. Crucially the gate **stays entirely
inert until the floor itself rises above `engage_floor_dbfs`** (-50 dBFS), so at home it never
engages and endpointing behaves exactly as before. A push-to-talk hold bypasses the gate's
verdict (the floor keeps tracking): a hand on the IR sensor is unambiguous intent, and gating
a quiet user out there would capture nothing at all.

Covered by `tests/test_noise_suppression.py`, `tests/test_near_field_gate.py`, and
`tests/test_endpointer_crowd.py` (drives the real `_Endpointer` with a stubbed VAD to prove
babble alone never commits, a near-field speaker still endpoints *while the room stays noisy*,
and a quiet room is unaffected).

### 5.4a `core/brain/ambient_listening.py` — Always-listen distillation

- Buffers timestamped finalized ambient transcripts in `ambient_listen_buffer.json` using
  atomic replacement; failed/denied cloud calls retain the batch for retry.
- A randomized scheduler flushes every `batch_min_minutes`–`batch_max_minutes`; wake-word
  and IR query activation request an immediate flush via the event loop without blocking.
- The cloud returns original transcript indices, not rewritten speech. Only indexed coherent
  snippets can produce a context summary, `focus=ambient_listening` journal entry, and
  conservative `knowledge_updates` (speaker identity must never be guessed).
- Live context is queued after distillation and drained by `main.py` only between turns,
  preserving the append-only KV-cache contract. Calls use the `ambient_listen` cloud-budget
  category and `purpose=summary`, which hard-pins them to cloud rather than the local slot.

### 5.4b `core/runtime_controls.py` — Spoken runtime settings

- Owns thread-safe, in-memory active mode, mode revision, and persistent follow-up state.
- Resolves `assistant_modes.modes.<name>.system_prompt`; the default mode's null prompt
  inherits the existing `llm.system_prompt`. `switch_mode` applies the mode's `voice` and
  increments the revision consumed by `main.py` at a cache-safe boundary.
- Mode-name resolution returns the exact config key using normalized equality first,
  contained-string/token matching second, and typo-tolerant similarity last. Close fuzzy
  ties are rejected, preventing an arbitrary switch between similarly named modes.
- `parse_spoken_control` conservatively recognizes explicit mode, volume, and follow-up
  commands. `core/llm.py` converts them to the normal tool-call event protocol before model
  generation, so the same execution/history/follow-up machinery handles them.

### 5.5 `core/tts.py` (612 lines) — TTS providers

- Provider chosen by `tts.provider` ("local" in production). Import-time config caching.
- `SUPPORTED_TAGS`/`TAG_MAP` + `sanitize_for_local_tts` @102 — the local voice model only
  understands 13 bracket tags; everything else (motion tags `<...>`, emojis, markdown chars)
  is stripped or remapped ([laugh]→[laughter], [gasp]→[surprise-ah], ...). Keep in sync with
  omnivoice `voice_api.py` and `tests/streaming_tts.py`.
- `get_tts_system_prompt_note` @118 — appends the tag restriction to the system prompt
  (local provider only). Called by main.py BEFORE warmup (cache safety).
- `get_local_voice` / `set_local_voice` / `list_local_voices` select preloaded omnivoice
  references at runtime. Switching clears the text-only PCM cache so audio from a previous
  voice cannot leak into the new mode.
- `_trim_edge_silence` @138 — strips model-baked silence at PCM edges (keeps 50 ms).
- `GroqTTSStreamer` @159 — sentence queue → Groq API → temp wav files → `mpv` playback.
- `InworldTTSStreamer` @250 — bidirectional websocket, OGG_OPUS chunks piped into a
  long-lived `mpv` stdin.
- `LocalTTSStreamer` @404 — **production path**: synth worker posts each sanitized sentence
  to `POST /v1/audio/speech` (`response_format: pcm`) on a keep-alive session; trims edges;
  prepends a 120 ms gap **only from the 2nd sentence on** (appending it after the last
  sentence used to delay mic unmute); playback worker holds a single long-lived `aplay`
  pipe and writes the first audible PCM immediately. Because omnivoice buffers each response
  body behind a full synthesis pass, short Hindi continuations are opportunistically joined
  into a >=48-character request while prior audio has safe playback headroom. Sentence 1 is
  never joined or delayed, preserving TTFW; English request boundaries are unchanged.
  `abort()` (the "stop it" hotword) kills the aplay pipe directly. `first_play_event` is the
  signal main.py uses to stop the thinking sound; a `finally` safety always sets it.
- `LCDOnlyStreamer` — TTS-compatible, audio-free output used while the camera mute toggle is
  active. It preserves speculative hold/release semantics and reveals the reply on the 16x2
  LCD at a readable pace; it never opens an audio process or contacts the TTS server.
- `TTSStreamer()` @563 — factory. Its unmuted provider path is unchanged; one in-memory mute
  flag selects `LCDOnlyStreamer` before any TTS work. `speak_sentence` @573 is the simple
  blocking helper used by workers.

### 5.5a `core/speech_recorder.py` — `record_enabled` speech archive

With `record_enabled: true` in config.json, every spoken reply is also saved as a wav in
`record_dir` (default `kiki_speeches/`, gitignored), named after the **first two words** of
the reply (`Hey_there.wav`; a `_2`, `_3` … suffix is added on collision). One file per
assistant response, not per sentence — a tool turn's filler + answer land in the same file
because they share one streamer.

It records the **exact PCM handed to the sink**, tapped inside
`LocalTTSStreamer._play_worker` immediately *after* `self._write_pcm(pcm)` (§5.5). Three
properties keep it off the latency path, and all three are load-bearing:

- **Capture is a `list.append` of a bytes object the playback thread already holds** — no
  copy, no encode, no I/O, no lock. It sits after the aplay write, so time-to-first-word is
  byte-for-byte the old path.
- **`close()` only puts the buffer on a queue.** It runs first in the play worker's
  `finally`, *before* the aplay drain, so it never adds to `finish()` — which main.py blocks
  on before unmuting the mic. The wav encode and the disk write happen on one daemon writer
  thread (`speech-recorder`).
- **`new_recording()` returns `None` when disabled**, so with the flag off the streamer's
  only cost is an `is not None` check per PCM chunk.

Because the tap is at the sink, an aborted reply ("stop it" / open-palm) saves exactly the
audio that was actually spoken, and a `_MAX_BYTES` cap (~10 min) bounds the buffer. Only the
**local** provider is recorded — the production path; Groq/Inworld play through `mpv` and are
not captured, and `LCDOnlyStreamer` (camera mute) produces no audio to record. Filenames keep
their own script: combining marks are preserved explicitly because `\w` drops them and would
mangle Devanagari (नमस्ते → नमसत).

### 5.6 `core/brain/unified_idle_mind.py` — Unified Idle Mind

- `UnifiedIdleMindManager` owns scheduling, conversation buffering, ambient snapshots,
  prompt construction, tool policy, queued actions, and one next-turn note.
- Model/provider/fallback/thinking level come from `idle_mind` in config.
- `_SessionPolicy` blocks physical actions, repeated intents, and excess research,
  persistence, or proactive actions. The live-data tools `read_gmail`,
  `read_gmail_message`, `read_gmail_thread`, `search_notion`, and `read_notion`
  are deliberately exempt
  from cross-session duplicate blocking because inbox/workspace contents may change;
  the shared agent loop still suppresses an exact duplicate inside one session.
- Routine Gmail/Notion reads use those five compact tools directly. Raw
  `self_extend_tool_call` requests for `fetch_emails`, Gmail message fetches,
  `notion-search`, or `notion-fetch` are redirected by policy so full HTML, MIME
  headers, MCP envelopes, and oversized workspace results cannot enter agent context.
- `_run_session` uses the shared `run_agent_loop`, records a complete observability
  session, consumes ambient snippets on success, and persists the next check.
- Foreground activity calls `interrupt`, but this deliberately does not cancel cloud
  work. Conflicting tools and prompt mutation wait until speech ends.
- Proactive context accepts a meaningful live scene and the single active next-turn note;
  there is no multi-point or automatic journal surfacing queue.

### 5.7 `core/brain/thinking_journal.py` — Dated research

- Schema v2 stores `entries` and `open_questions` only.
- `save_background_research` validates and deduplicates model-selected writes.
- `recent_summaries` supplies anti-repetition context to Unified Idle Mind.
- `recall_memory` searches full topic/summary/details and labels matches as dated
  background research.
- Open questions persist curiosity between sessions and can be resolved by word overlap.

### 5.8 `core/brain/knowledge_base.py` (542 lines) — Long-term memory

- `KnowledgeBase` over `knowledge_base.json`: categories `people` (incl. self "Kiki"),
  `environments`, `learnings`, `experiences`, `facts`, `personality`, `metadata`.
- People CRUD: `add_person`, `add_person_attribute` (list attrs append-dedup; scalar attrs
  overwrite; **setting `current_ongoing` also stamps `current_ongoing_updated`**),
  `add_note_to_person`, `set_current_ongoing`, etc.
- `get_summary(max_lines=50)` @~407 — the context injection: Kiki self-section (mood + last
  3 notes), people (appearance/character/interests/routine[:3]/last 2 notes),
  **`current_ongoing` aging**: ≤14 days → "Currently:", 15–60 days → "A while back (N days
  ago, probably finished):", >60 days → dropped, undated → "At some point (undated, may be
  old):". This stops months-old "currently working on X" from surfacing as live context.
- Module singleton: `get_knowledge_base()`, `get_knowledge_summary()`, `save_knowledge_base()`.

### 5.8a `core/brain/memory_search.py` — Human-like cross-store recall

- Backs the `recall_memory` speaking tool and searches **granular records** from the full
  knowledge base (including experience outcomes/details and archives), every timestamped
  conversation file, the current summary, and thinking-journal entries.
- Query scaffolding ("search my memories for...") is removed; remaining terms use exact
  phrase/token scoring, a small autobiographical concept map (humor, study, sleep, music,
  etc.), conservative stemming, and typo-tolerant fuzzy matching. Rare terms, title hits,
  full-query coverage, source value, and recency influence rank.
- Large people dictionaries are split into individual facts/notes before ranking, preventing
  the speaking path's 1500-character tool-result cap from hiding later matches. Similar
  records are deduplicated and selection applies source diversity.
- A filesystem-signature cache keeps repeat calls cheap but automatically rebuilds when any
  memory file changes. If no meaningful term matches, deterministic salience + diversity
  returns an explicitly labelled approximate assortment instead of "nothing found".
- Output is capped below the speaking tool limit and tells the follow-up model to synthesize
  dated evidence while treating fallback memories as leads rather than exact matches.
- The fast-model tool catalog stays deliberately terse. `core/llm.py` gives `recall_memory`
  one mandatory rule plus one example, and renders compact parameter signatures instead of
  verbose JSON schemas. A conservative code-level router emits the same normal tool event for
  unmistakable autobiographical prompts ("what did we discuss", "search your memory", "funny
  memories", etc.), so a missed decision by the small model cannot become a hallucinated answer.

### 5.9 `core/brain/summary_manager.py` (215 lines) — Summaries

- Single-file summary (`conversation_summary.txt`): `load_saved_summary`/`save_summary`.
- Timestamped per-session files in `conversations/` (newest-first list);
  `save_summary_to_conversations_folder`, `load_latest_conversation` (raw injection of the
  very last session).
- `generate_past_conversations_summary(n, prefer_cache, force_refresh)` @135 — combines the
  N previous session files into one LLM-written memory, cached in
  `conversations/cached_past_summary.txt`. **`prefer_cache=True`** (startup) returns the
  cache even if stale — every shutdown writes a new conversation file, so the strict mtime
  check missed on every boot and a ~60s box-blocking regen ran at the worst possible time.
  **`force_refresh=True`** (the mid-session background refresher) regenerates uncached and
  writes the cache for the next boot. Generation runs on the local box via
  `generate_background` (refused while hot → retry later).

### 5.10 `core/brain/generate_llm_resp.py` (227 lines) — Brain/vision/summary router

`generate(content, b64_image, thinking_level, websearch, purpose)`: routing by
`use_local_llm` flags (currently `summary: true`, `vision: false`, `reasoning: false`) —
local primary goes through `local_llm.generate_background` (so it's preemptible and
hot-window-aware); then Gemini key rotation (`GEMINI_KEY_LIST`, gemini-3-flash-preview →
gemini-2.5-flash, optional Google-Search grounding, image support), then Groq
(`GROQ_API_KEY_LIST`, gpt-oss-120b, text-only), then local as last resort. This is the
SLOW/quality path — completely separate from the speaking pipeline.

### 5.12 `core/brain/token_counter.py` (133 lines)

`count_tokens(messages, model)` with tiktoken (char/4 fallback), per-message hash cache
(2048 entries, cleared when full) because history is append-mostly and re-tokenizing every
message every turn wasted Pi CPU. Handles both dict and object message shapes.

### 5.13 `core/workers/` — Background agent system

**`worker_engine.py` (173)** — dataclasses only. `Worker` (id/name/task_description/
trigger/conditions/status/retries), `WorkerTrigger` (scheduled_time ISO | event name |
recurring interval), `WorkerCondition` (person_seen / time_range / custom),
`VALID_EVENTS = {startup, shutdown, sleep, wake, after_response, face_detected}`.
`mark_failed` keeps status `pending` until `max_retries` is hit.

**`worker_brain.py` (520)** — the shared agent loop:
- `FaceHistoryBuffer` @30 / `VisionContextHistory` @86 — thread-safe rolling buffers filled
  by face_handler/vision_handler, read by worker conditions and prompts.
- `check_conditions` @132 — person_seen (within N min), time_range (hour window).
- Context budget constants @~175: `MAX_PROMPT_CHARS=20000`, `MAX_TOOL_RESULT_CHARS=3000`.
- `_truncate_middle` — keeps head (70%) + tail of oversized tool results.
- `_compact_conversation` — **smart context compression** for the 7k-token box: the task
  prompt (conversation[0], with all tool/JSON instructions the small model needs) is kept
  VERBATIM, the last 2 entries kept whole, older middle entries squeezed to fit the budget.
- `run_agent_loop(prompt, llm_fn, max_turns, label, stop_event, min_tool_calls,
  max_tool_calls, max_calls_per_turn, max_prompt_chars, max_tool_result_chars,
  continue_guidance_fn)` @~230 — the engine shared
  by workers and Unified Idle Mind. JSON protocol: model replies either `{"tool_calls":[{tool,args}]}` or
  `{"status":"completed","summary",...,"speak":bool,"speak_text"}` or `{"status":"failed",...}`.
  Invalid JSON → the parse error is fed back for self-correction; `min_tool_calls` enforcement
  counts TOTAL executed calls (`total_tool_calls`, not unique tool names — 5 search_web calls
  count as 5). **Hard tool-call budget** (stops runaway research loops — on cloud the model can
  batch a 20-call `tool_calls` array AND keep doing so turn after turn, so `max_turns` alone
  doesn't bound executed calls; one idle cycle fired 50+ searches): `max_calls_per_turn`
  (default 5) caps how many calls one turn executes (rest deferred to the next turn);
  `max_tool_calls` (0 = unlimited; idle 6 / deep 12 / reflection 6 / worker 12 from config) is a
  TOTAL ceiling — once spent the loop refuses further calls and forces a final-JSON wrap-up
  (2 nudges via `_MAX_FORCED_WRAPS`, then bails with what's gathered).
  `continue_guidance_fn(total_tool_calls, tools_used)` optionally replaces the
  generic "Continue with your task" tail after each tool round (deep-research coaching), but the
  budget wrap-up message overrides it once the ceiling is hit;
  `_compact_conversation` clamps the RECENT entries too when head+recent+middle-floor bust
  the budget (head always verbatim); `stop_event` checked between turns and
  before each tool call (Unified Idle Mind interruption); `execute_python_code` runs via a temp
  file in a subprocess. Returns `(success, result, speak_text, final_json, tools_used)`.
- `execute_worker(worker)` @~395 — builds the worker prompt (task + time + tools + recent
  vision/face context + retry note) and runs the loop.

**`worker_manager.py` (510)** — `WorkerManager` singleton (`get_worker_manager`):
persistence to `workers.json`, CRUD (`create_worker` validates triggers, caps at 20 active),
scheduler thread (30s tick → time + recurring triggers), `fire_event(name)` (event-triggered
workers; face_detected filtered by person condition), `_execute_worker_background` (runs on
the main loop; recurring/event workers reset to pending instead of completed; result
injected into chat history as a system message; optional spoken result via `_speak_text` →
TTSStreamer + assistant-message append).

### 5.14 `core/vision/`

**`camera.py`** — `capture_photo_b64()`: one frame from `http://localhost:5000/mjpeg` via
OpenCV → JPEG → base64. Debug frame save gated behind `KIKI_DEBUG_SAVE_FRAME` env var.

**`vision_handler.py` (≈128)** — `VisionHandler`: two timers — the question timer
(`agent.question_*_interval_seconds`, currently 1000–3000s, drives both in-conversation question
injections and the periodic loop) and the slower vision-QA timer
(`vision_injection.qa_interval_*`, 30–60 min). `run_vision_update(force_trigger, force_qa)`:
skips while Unified Idle Mind; captures a photo; analyzes on the local box
(`generate_background`, preemptible) **with cloud fallback when the box refuses/fails**;
records into the shared vision history; then either queues an `("autonomous_vision", ctx)`
event (spoken proactive comment, QA path) or silently injects the text. Silent injection
is now **eager + pre-warmed**: `history_inject_fn` (set by main.py) appends the context to
`message_history` immediately and calls `register_history` so the background rewarm bakes
it into the warm prefix — the next turn's prefill is a cache hit instead of paying those
tokens on the speaking path. Before a spoken QA event is queued,
`proactive_prompt_fn` requires a fresh grounded source and suppresses low-value
empty/quiet-scene interruptions. Refused while a turn is mid-flight (`turn_active`) or during
summarization → falls back to the old `pending_vision_context` (injected at next turn).
Known people list comes from the Hailo train directory.

### 5.15 `core/self_extend/`

- **`skill_manager.py` (128)** — local SKILL.md skills under `skills/`: list (name + 400-char
  preview), create (dir + SKILL.md + extra files), `get_skills_summary()` injected into the
  system context at startup.
- **`smithery_cli.py` (221)** — subprocess wrapper over the authenticated `smithery` CLI:
  mcp search/add/list/remove, tool find/list/call, skill search/view, and
  `install_skill_to_kiki` (full SKILL.md content → local skills dir).
- **`mcp_data_access.py`** — compact model-facing access over the connected `gmail`
  and `notion` Smithery services. Gmail list reads force metadata-only mode and return
  bounded snippets; individual messages decode the preferred text/plain MIME part and
  discard raw headers/HTML. Notion search sets server-side page/highlight limits and
  `read_notion` bounds selected enhanced-Markdown content. These helpers unwrap duplicate
  Smithery envelopes before returning results. The Instagram connection was removed.
- **`whatsapp_mcp.py`** — owns one long-lived stdio session to the bundled 12-tool
  WhatsApp FastMCP server on a daemon event-loop thread. It starts/checks the Go bridge
  without waiting, resolves unique contact names for sends, bounds returned message
  content, and exposes startup/poll helpers without importing the MCP SDK on Kiki's
  foreground import path.
- **`mcp_manager.py` (396)** — MCP registry client, Claude-Desktop-config management,
  FastMCP server code generation (`MCPServerCreator`).
- **`kiki_self_extend_agent.py` (449)** — autonomous JSON-loop agent (same
  thought/tool/done protocol) over `generate_llm_resp.generate`; goal-driven skill/MCP
  installation. Triggered via the `self_extend_run_task` tool or (when enabled) Unified Idle Mind.

### 5.16 `hotwords/hotword_recog.py` (179 lines)

OpenWakeWord on `kiki.onnx`, threshold 0.5, 4.5s detection cooldown. The key mechanism is
**pause/resume around Kiki's own speech**: `pause()` while the bot talks (frames are still
drained so the buffer never overflows, but no inference — the bot can't wake itself);
`request_resume()` arms quiet-detection: the loop resumes only after 3 consecutive frames
under RMS 500 (~240 ms of silence) or a 2.5s hard cap, then `owwModel.reset()` clears
activations accumulated from the bot's own audio and the cooldown is shortened to 0.3s so
the user isn't locked out. `listen()` yields `"heyy"` (main.py's expected wake token).

### 5.17 `robot/`

- **`face_handler.py` (99)** — resilient async loop: connect to KikiController, consume
  face and `hand_gesture` events. Hand controls bypass face rate limits and are routed to
  main.py immediately (the controller connection remains active for gestures even when
  conversational face injection is disabled). Face events record into FaceHistoryBuffer; fire `face_detected`
  workers; rate-limit injections to 2 per 5 min; known face during Unified Idle Mind →
  `interrupt` + `("face_wake", name)` event (forces a spoken vision QA); inject
  `[System: '<name>' has just appeared...]` into history (append-only).
- **`movement.py` (113)** — `<turn(90)>`, `<forward(50)>`, `<move(angle,dist)>`-style tag
  regexes; `extract_movement_tags` (→ dicts), `strip_movement_tags` (applied to each
  sentence before TTS; the stripped text is the spoken/`clean_response` copy, while the
  copy STORED in history stays verbatim with its tags — see §4 rule 2),
  `execute_movements` (legacy direct-GPIO path via `motor_control`; the `move`/`dance`
  TOOLS use the ZMQ motor server instead).
- **`motor_control.py` (380)** — low-level gpiod + SoftPWM mecanum driver (pins, trims,
  serial). Primarily used by the separate motor-server process; tools.py deliberately never
  touches GPIO directly ("Device or resource busy" elimination).

### 5.18 `tools_and_config/`

- **`config_loader.py` (73)** — loads `.env` + `config.json` ONCE at import into module
  global `CONFIG`; the `get_*_config()` accessors return live references (mutations would be
  shared — treat as read-only). **A config change requires a restart.**
- **`logger.py` (111)** — `_Tee` wraps stdout/stderr: every write mirrored to
  `logs/kiki.log` with a per-line `HH:MM:SS.mmm` prefix; size rotation at `max_bytes`
  (5 MB, one `.1` backup). `debug(tag, msg)` prints only when `logging.debug` is true
  (Unified Idle Mind prompts/responses, stream stats).
- **`tools.py` (1678)** — all tools. Layout:
  - utilities @25–170: `should_skip_followup` (one-shot flag set by play_music/dance so
    main.py skips the follow-up listen), `set_motor_relay` (sync variant),
    `KikiMotorClient` (stateless per-call ZMQ REQ to the motor server @5557 with
    `VALID_MOTOR_ACTIONS` — the full mecanum vocabulary), Exa client lazy-init
    (**⚠ hardcoded API key @149**), shared KikiController getter.
  - tool impls @174–1052 (all async): `search_web` (Exa, 3 results, highlights, IN locale,
    time ranges), `execute_shell_command` (10s timeout), `get_current_time`,
    `recall_memory` (→ journal.search), `switch_voice`, `switch_mode`, `set_followups`,
    `adjust_volume` (PulseAudio default sink), stateful YouTube music tools (`play_music`,
    exact-video likes, liked-song playlist, last song, pause/resume/next/previous) backed by
    `core/media_manager.py`, `set_timer` (validated in-process countdown + mpv alarm),
    `update_knowledge` (full KB CRUD grammar over categories/actions/attributes),
    `remember_me` (face training via controller), `track_person`, `follow_me`,
    `move(steps)` (threaded step sequencer via motor server; per-step interval/duration/
    speed clamps), `dance(song, steps)` (music + choreography interleaved; waits for mpv
    audio, **hardcoded 20s pre-choreo sleep @657**, pause steps, cleanup killpg),
    worker tools (`schedule_worker`/`cancel_worker`/`list_workers`), `execute_python_code`
    (subprocess with the `/usr/bin/python3` venv), five compact
    Gmail/Notion read tools, and `self_extend_*` wrappers over
    skill_manager/smithery_cli/mcp_manager/agent.
  - `TOOLS` @1058 — OpenAI function schemas for every tool (the `dance` schema embeds an
    entire choreography style guide; `move` documents the turn-rate constant).
  - dispatch @1575: `_ASYNC_TOOL_HANDLERS` map; `execute_tool` (sync — used by the
    speaking-path loop; runs the coroutine in a throwaway thread+loop when already inside
    an event loop, 30s cap); `execute_tool_async`; `get_tool_descriptions` (name→desc) and
    `get_detailed_tool_descriptions` (full text for agent prompts).

### 5.18b `core/ir_controls.py` (≈330 lines) — IR-sensor + LCD control surface

Two active-LOW IR proximity sensors on `/dev/gpiochip4` (hand over sensor ⇒
`Value.INACTIVE`): **LEFT = GPIO 22** (Kiki's left), **RIGHT = GPIO 17** (Kiki's right).
`IRControls` polls both at ~33 Hz on a daemon thread and degrades to disabled if
`gpiod`/the lines are unavailable (runs off-robot fine).

- **NORMAL mode** — holding either sensor is push-to-talk: it immediately interrupts
  music/speech/generation, opens STT, and commits on release. Double-tapping the same
  sensor within `DOUBLE_TAP_GAP_S` fires `on_double_tap`; `main.py` aborts active audio/
  generation, discards the gesture's pending STT events, closes the query window, and
  returns Kiki to idle. *Both sensors held* `BOTH_HOLD_S` ⇒ open settings.
- **Release debounce is asymmetric.** A hold asserts on the *first* present sample so
  push-to-talk feels instant, but cancelling a pending release needs
  `HOLD_REASSERT_SAMPLES` (2, ≈60 ms) consecutive present reads. These sensors chatter at
  the edge of their cone and pick up ambient IR, and a single spurious read used to reset
  the release window on every poll — the hold then never ended and Kiki kept listening
  after the hand was gone. `HOLD_MAX_S` (25 s) is the backstop for a line stuck asserted:
  it commits the turn and refuses to re-arm until the sensors genuinely read clear.
  `clear_hold()` drops a hold without committing (used by the thumbs-up path).
- **SETTINGS mode** (modal; main.py mutes STT + pauses the hotword recognizer via
  `on_enter_settings`, restores via `on_exit_settings`): tap RIGHT ⇒ selection left, tap
  LEFT ⇒ selection right; long-hover (≥`LONG_HOVER_S`) either sensor ⇒ select. Menu =
  `["BT Volume", "Restart", "Exit"]`. **BT Volume** sub-screen: tap LEFT `+VOL_STEP` / tap
  RIGHT `-VOL_STEP` applied live via `pactl set-sink-volume @DEFAULT_SINK@ N%` (the BT
  speaker is the default bluez sink), long-hover confirms. **Restart** re-execs the process
  (`os.execv(sys.executable, [sys.executable]+sys.argv)` — restarts main.py in the same
  venv). Auto-exits after `IDLE_EXIT_S` of no gesture.
- **Tap vs long-hover** is per-sensor press timing; `_settings_armed` ignores input until
  both sensors clear once after entry/select so the resting hands don't auto-navigate.
  A same-sensor double-tap in settings exits the menu and returns to idle as well.
- Wired in `main.py` right after `idle_mgr` (closures `ir_talk_hold_start`/
  `ir_talk_hold_end`/`ir_return_to_idle`/settings callbacks), `ir_controls.stop()` in
  the shutdown `finally`.

Camera gestures use the same immediate, event-driven control principles. With
`hailo_follower_webcam_only.py --hands`, the hand worker reuses its already-smoothed
classification result and publishes `hand_gesture` over the existing controller PUB socket.
Holding a control pose emits once; that same pose must be absent continuously for 0.8s before
it can fire again. Poses listed in `CONTROL_GESTURE_HOLD` get a stricter false-positive gate:
their smoothed *and* current raw label must both agree at a minimum confidence for a dwell
time (with a 0.15s dropout grace) before firing — `mute` needs >=92% for 1.5s, `thumbs_up`
>=85% for 0.4s. The control mapping is:

- `mute`: toggle persistent output mute. A reply uses `LCDOnlyStreamer`, so model generation
  and prompt caching stay unchanged while TTS synthesis/playback are skipped entirely.
- `open_palm`: invalidate in-flight activities, stop music/filler/TTS, abort current model
  generation, and turn off active neck tracking. A music URL resolution racing the gesture
  checks the stop generation before it is allowed to launch `mpv`.
- `peace`: perform the same terminal cleanup as IR double-tap and return to hotword/idle.
- `thumbs_up`: **end of listening** — "I'm done, answer now." Clears any IR hold via
  `ir_controls.clear_hold()` and calls `stt.commit_now()`, which drops the push-to-talk
  hold and sets the force-commit flag so the endpointer emits `final` + `endpoint`
  immediately instead of waiting out the silence window. This is the manual escape hatch
  for rooms so noisy that the VAD never sees enough trailing silence on its own.

### 5.18c `core/lcd_display.py` — 16x2 char LCD

`LCDDisplayManager` singleton (`lcd_manager`) over an I2C PCF8574 16x2 LCD (RPLCD; emulated
print-only when the lib/panel is absent). Async write worker thread coalesces a backlog to
the latest frame; `write`/`clear`/`update_status(action,details)` (maps states→layouts),
`display_stream`/`wrap_text_16x2` (sliding 2-line scroll), `_clean_text` (strips
markdown/bracket/XML tags). Speech frames carry a playback-session id, so barge-in or turn
completion invalidates queued words before they can overwrite the next status. Physical
writes overwrite both padded rows without clearing the panel, avoiding flicker and reducing
I2C latency; commit timestamps are exposed only for calibration/tests. Used across main.py,
tts.py, main.py, and ir_controls.py.

### 5.18d `core/tts_sync.py` + `tests/calibrate_lcd_sync.py` — calibrated LCD speech clock

Local TTS keeps its existing first-PCM streaming path: **LCD synchronization never buffers
audio and never delays the current ~0.8s time-to-first-word**. On the first audible PCM chunk,
`LocalTTSStreamer` creates a word schedule from the already-known TTS text and a preloaded
per-voice/per-script calibration profile. Every PCM chunk advances the master audio clock;
if synthesis or a tool-result follow-up starves the `aplay` pipe, future word times are
re-anchored to the new audible window instead of accumulating drift. Expression tags consume
calibrated time but are not printed. Cached fillers, tool bridges, normal responses, and
follow-up responses all use this one path.

Devanagari speech cues are converted to ASCII Hinglish only in the independent LCD worker,
after PCM has already entered `aplay`. The dependency-free Unicode converter makes no API or
network calls and never changes the text sent to TTS, the timing schedule, or the playback
critical path; mixed Latin/Devanagari replies retain their existing Latin text.

Whisper is **calibration-only**. Running `tests/calibrate_lcd_sync.py` manually synthesizes
known English, Hindi and mixed-language phrases for every server voice, requests token
timestamps from the existing whisper.cpp server, fits the small timing profile, and refuses
to replace the profile unless the configured 150ms validation bound passes. Neither
`core/tts.py` nor `core/tts_sync.py` imports/calls Whisper during a conversation. A missing,
stale-speaker, or missing-voice profile fails closed: audio still streams normally while the
LCD shows `Speaking... / Calibration req` rather than knowingly mistimed words.
Run it with Kiki stopped so it can briefly own the microphone and speaker:
`source /srv/kikifast/.venv/bin/activate && python tests/calibrate_lcd_sync.py`.
The startup wizard's `Sync LCD+audio?` step (§3.0) runs this same script unattended at the
end of boot — it is still the only thing that writes the profile, and it is still never
imported or called by main.py.
It speaks the short English latency phrase three times, measures real Bluetooth + LCD write
latency, and writes `tools_and_config/lcd_sync_calibration.json`. Hindi/mixed Whisper text is
treated as non-authoritative: mismatches are skipped and the Devanagari fallback is fitted
from deterministic source-PCM duration instead of forcing an incorrect transcript match.

### 5.18f `core/oled_display.py` + `robot/oled_tags.py` — the face

A single 128x64 SSD1306 at 0x3c sharing bus 1 (and `I2C_LOCK`) with the char LCD. There
are no bitmap assets: every frame is procedurally drawn PIL geometry around a 15x16
logical-pixel crab sprite (`_crab`), rendered directly at native x4 scale for crisp 1-bit
edges. `OLEDDisplayManager` is a singleton with a
daemon render thread; `set_state` is a lock-guarded flag flip, so **no caller ever blocks
on an animation**. Each state is one `_draw_<name>` method resolved by `getattr`, plus a
`VALID_STATES` and `_STATE_FPS` entry — that is the whole contract for adding one.

Two layers drive it:

1. **System state** — `speaking`, `listening`, `thinking`, `tool`, `music`, `workers`,
   `idle_mind`, `face`, … pushed by the runtime, and by `lcd_display.update_status`, whose
   keyword mapper fans every LCD status string out to the OLED.
2. **Expression tags** — the speaking model emits `<oled:name>` inline and Kiki's face
   changes to match what it is saying. `EXPRESSION_STATES` (19 moods) is the single source
   of truth for tag validity *and* for the prompt vocabulary
   (`get_oled_tag_prompt_note()`), so what the model is taught can never drift from what
   can be drawn.

**Priority.** `set_expression` only overrides `_EXPRESSION_OK_FROM` (`speaking`, `tool`,
another expression) and only accepts `EXPRESSION_STATES`, so a reply can colour its face
but can never claim to be listening or running a tool, and a late tag cannot stomp a face
card or worker progress. The one inversion: `set_state("speaking")` will *not* replace a
held expression, because `speaking` is the baseline face for a turn and main.py's
first-play callback would otherwise race the player and win. The turn's closing
`set_state("idle")` releases the hold.

**Timing (why this is not the neck-tag dispatch).** Neck gestures are collected for the
whole turn and fired after it. That is wrong for a face: the LLM streams several sentences
ahead of the voice, so firing on parse puts the expression ahead of the words. Instead the
tag rides in-band through `add_sentence`, is read off the raw text in
`LocalTTSStreamer._synth_worker` (before `sanitize_for_local_tts` strips all `<…>`),
travels on the `("meta", speakable, gap, voice, oled)` marker, and is applied by
`_play_worker` on the sentence's **first audible PCM chunk — after `_write_pcm`**. TTFW is
therefore untouched. Cloud/LCD-only streamers have no such marker and fire at queue time
via `_fire_oled_tag`, which also strips neck tags (previously only the local path did, so
cloud voices read them aloud).

**Latency rule.** A tag must never open a reply or a sentence — tokens spent before the
first word delay the voice directly. The prompt note says so, and `core/llm.py`
`_has_spoken_word` makes it a safety net rather than a dependency on model behaviour: a
flushed fragment containing only silent tags no longer clears `first_sentence_pending`, so
a leading tag can't disarm the eager first-sentence flush. Measured on the cloud scanner,
a leading tag cost 46 chars-to-first-audible before the fix and 21 after; a mid-sentence
or trailing tag costs exactly the baseline 6, i.e. nothing.

**KV-cache.** Identical contract to the neck tags (§4 rule 2): `message_history` stores
the reply VERBATIM with `<oled:…>` in it; only `clean_response` and the TTS text are
stripped. The prompt note is a *static* suffix appended once next to
`get_tts_system_prompt_note()` — it joins the warmed prefix, so turns still prefill nothing
but the user's new message. It is generated from a `sorted()` set precisely so it stays
byte-stable across restarts.

Covered by `tests/test_oled_tags.py` (parsing, registry consistency, priority, TTFW).

### 5.18e `core/wifi_setup.py` — boot-time Wi-Fi provisioning

Standalone synchronous boot UI using the same active-LOW GPIO22/17 sensors, but not the
runtime `IRControls`/`STTEngine` objects (they do not exist yet). `IRWifiSetup` scans and
connects through passwordless-sudo `nmcli` calls (the system service has no active desktop
PolicyKit session), retains each AP BSSID for reliable activation, sorts/de-duplicates
SSIDs by signal, and releases its gpiod line request in a `finally` block.
`LocalPasswordDictation` captures
only while LEFT is held and runs local whisper.cpp after release. Pure helpers
`parse_nmcli_networks` and `apply_password_dictation` contain the escaped-SSID and spoken
character/edit parsing logic and are covered by `tests/test_wifi_setup.py`.

### 5.19 `tests/streaming_tts.py` (≈500 lines)

Self-contained gapless streaming-TTS library (mirrors `core/tts.py` LocalTTSStreamer
mechanics: sentence splitter with eager first split, synth-ahead worker, prebuffer-gated
single aplay pipe, edge-silence trim) + the **`FILLERS` list** (72 Kiki-personality filler
lines, grouped by style: robot-body humor, teasing Alex, mock drama, TARS deadpan,
swagger, mock exasperation, curious hums, warm beats, quirky one-offs — all TTS-safe, no
contractions, only `SUPPORTED_TAGS`) and `generate_fillers(tts_url)`
(`--generate-fillers`) which synthesizes each filler to
`sound_effects/soundeffects/fillers/filler_N.wav` — the directory `ThinkingSoundPlayer`
plays from. CLI: `--demo`, stdin mode, `--wav` sink.

### 5.20 `tests_llamaserver/test_prefill_e2e.py`

Three live tests against the box (run after touching any cache machinery):
1. cold turn → rewarm (history+assistant, max_tokens=1) → next turn must be ≫ faster
   (verified: 24s → 1.26s TTFT);
2. mutating an earlier system message → demonstrates re-prefill cost (why §4 rule 1 exists);
3. `generate_background` aborted mid-prefill via `preempt_background()` — preempt must
   return in ~0 ms and the request must die promptly.

### 5.21 `kiki_control_client.py` (≈380 lines)

`KikiController`: async ZMQ REQ (commands: neck_movement, mode, target person, train_person,
full_body_movement, state queries) + SUB (`listen_events` async iterator: face_detected /
face_lost / training_complete) against the Hailo pipeline at `controller.host` (192.0.2.20).
`quick_command()` one-shot helper (used for the motor relay).

### 5.22 `kiki_startup.sh`

Boot script: exports Pulse/DBus/X env (audio from systemd context), traps for graceful
shutdown of all child PIDs, launches the process stack (Hailo pipeline, motor server,
camera stream, main.py) each in its own venv.

### 5.23 `core/observability.py` + `webui/server.py` — Dashboard

`Recorder` keeps non-blocking flat events plus grouped sessions for speaking turns,
Unified Idle Mind, workers, tools, vision, and summarization. Unified Idle Mind opens one
`idle_mind` session and `run_agent_loop` records every model turn and tool result under
that session.

The Flask dashboard on `:8090` exposes Controls, Sessions, Live Feed, Context, and the
full config editor. Curated controls cover Unified Idle Mind scheduling/tool budgets,
active cloud limits, workers, summaries, vision, and routing.

`CloudBudget` enforces global limits plus active categories
(`idle_mind`, `vision`, `summary`, `reasoning`, and `face_enroll`). A cap of zero
means unlimited. Unified Idle Mind reads its dedicated provider/model/fallback/thinking
settings at process startup; restart after changing those values.

### 5.24 `core/senior/` — Senior Citizen Mode (elderly-care addon)

Strictly **additive** addon; active only while the `senior` assistant mode is selected. Reuses the
existing speaking path, workers scheduler, face recognition, memory search, Unified Idle Mind and ambient
listening — it adds a caregiver **care plan** and family **email** on top.

- **`care_plan.py`** — `CarePlan` over `care_plan.json` (gitignored; path from `senior_mode.care_plan_file`).
  Atomic tmp+`os.replace` saves, module singleton (`get_care_plan_store`). Sections: `senior` (profile+
  language), `family_contacts` ({name,email,relationship,notify_on:[alert,daily_summary]}), `reminders`
  ({id,category,message,schedule,enabled}), `exercises` ({id,name,steps,schedule,prescribed_by}),
  `approved_music`, `approved_topics`, `care_log` (rolling ≤500). Schedule shape:
  `{"kind":"recurring","value":<sec>}` | `{"kind":"daily","value":"HH:MM"}` | `{"kind":"once","value":"<ISO>"}`.
- **`senior_care_manager.py`** — `SeniorCareManager` bridges the care plan onto `WorkerManager`.
  `activate()` materializes one worker per enabled reminder/exercise + a daily-summary worker;
  `deactivate()` cancels every `senior:*` worker; `sync_workers()` rebuilds after a voice edit.
  Daily schedules are recurring 86400s workers whose `last_fired_at` is back-dated so the first fire
  lands at HH:MM (§workers scheduler uses elapsed-since-last-fired). Reminder/exercise workers speak via
  the normal `execute_worker → _speak_text` path in Hindi; the daily-summary worker reads `care_log` and
  calls `send_care_email`.
- **Tools** (in `tools.py`, on senior mode's per-mode `main_tools`): `update_care_plan(section,action,data)`
  and `get_care_plan(section)` (voice-first plan editing → auto `sync_workers`), `alert_family(reason,
  urgency)` (emails all alert contacts on distress/emergency + logs it), `send_care_email(to,subject,body)`
  (used by the daily-summary worker). Email goes through a **Gmail MCP**: `send_care_email` reads
  `senior_mode.email.{connection,tool,arg_map}` and calls `smithery_cli.tool_call` (same path as
  `self_extend_tool_call`). **One-time setup**: `smithery mcp add <gmail>`, `self_extend_tool_list <conn>`
  to find the send tool + args, then fill `senior_mode.email` in config.
- **Per-mode `main_tools`**: `core/llm.py _effective_main_tools()` honors an optional
  `assistant_modes.modes.<mode>.main_tools` override (senior adds the care/alert tools), else the global
  `llm.main_tools`. Cache-safe — a mode switch already replaces msg[0] and re-warms (§4).
- **Wiring** (`main.py`): the manager singleton is built right after `worker_manager.start_scheduler()`
  (activates if `active_on_startup == "senior"`), and `sync_mode_prompt` (the cache-safe mode boundary)
  activates/deactivates on mode change. Boot straight into it via `assistant_modes.active_on_startup:
  "senior"`, or say "switch to senior citizen mode".

---

## 6. `tools_and_config/config.json` Reference

| Block | Key points |
|---|---|
| `llm` | Foreground speaking provider/model, local endpoint, tools, prompt, and cache controls. |
| `idle_mind` | The only background cognition configuration: provider/model/fallback/thinking level, state/journal paths, scheduling limits, and tool budgets. |
| `action_agent` | The `complex_query` multi-step agent (§5.2c): provider (`cerebras` default / `groq` fallback), per-provider model and context caps, turn/tool budgets, and the wall-clock deadline. Cloud-only; never uses the local slot. |
| `always_listen_config` | Capture-only buffer path and transcript size limits. |
| `cloud_limits` | Global and active-category caps; `idle_mind` has its own row. Zero means unlimited. |
| `knowledge_base` | Durable-memory path and startup context limits. |
| `agent` | Conversation summaries, token threshold, time anchors, and proactive-vision intervals. |
| `prompts` | Speaking, vision, greeting, summarization, and context wrappers. Unified Idle Mind builds its own protocol prompt in code. |
| `workers` | Independent scheduled/event background workers and their agent-loop limits. |
| `use_local_llm` | Shared non-speaking vision/summary/reasoning router; not used by Unified Idle Mind. |
| `vision_injection`, `peeping` | Scene capture and periodic local ambient capture behavior. |
| `self_extend` | Skill and MCP directories plus startup skill injection. |
| `whatsapp` | Async bridge/MCP lifecycle, timeouts/logs, and local idle-message polling/debounce/lookback limits. `contacts` is the manual name→number override map (§5.2d); the real address book is read automatically from the bridge store. |
| `logging` | Tee file, rotation, and debug verbosity. |

## 7. Data Files

| File | Writer | Reader |
|---|---|---|
| `knowledge_base.json` | Explicit memory tools and Unified Idle Mind | Startup context and `recall_memory` |
| `thinking_journal.json` | `save_background_research` and open-question tools | Unified prompt and `recall_memory` |
| `idle_mind_state.json` | Unified Idle Mind | Scheduler, anti-repeat policy, queued actions, one next-turn note, WhatsApp high-water cursor |
| `ambient_listen_buffer.json` | AmbientListeningManager | Unified Idle Mind snapshot/consume |
| `workers.json` | WorkerManager | WorkerManager |
| `conversation_summary.txt`, `conversations/*.txt` | Summary manager | Startup memory and `recall_memory` |
| `logs/kiki.log` (+`.1`) | Logger tee | Humans and diagnostics |
| `logs/whatsapp-bridge.log`, `logs/whatsapp-mcp.log` | WhatsApp bridge/MCP subprocesses | Humans and diagnostics |

**None of these are tracked in git, and that is deliberate.** They are written by the
running robot, so their on-disk copy drifts away from whatever git last committed. On
2026-07-30 an under-voltage brownout left commit `013fb88` written but nine of its blobs
never flushed; every *source* file was recoverable by re-hashing the worktree copy
(`git hash-object -w <path>`), but the tracked `__pycache__/main.cpython-313.pyc` had been
regenerated in the meantime and was gone for good — it had to be dropped and the commit
rebuilt. Machine-written files therefore stay untracked (`.gitignore`), so a power cut can
only ever cost data that is recoverable from disk.

**Repo durability settings** (these live in `.git/config` + `~/.gitconfig`, so a fresh
clone on a new machine must re-apply them):

```
git config core.fsync all          # fsync loose objects, packs, index AND refs
git config core.fsyncMethod fsync  # real fsync, not writeout-only
git config transfer.fsckObjects true
```

The default is `core.fsync=committed`, which omits the index and refs. `all` costs ~5 ms
per object write on this SD card — immaterial. Note this hardens git's own write ordering
but cannot compensate for an SD card that lies about flushing; the durable fix for the
brownouts is the power supply, not git.

## 8. Important Things to Keep in Mind (gotchas & invariants)

1. **All four KV-cache rules in §4.** They are the difference between 1–2s and 40–60s replies.
2. **Never block the hotword thread.** Everything in the wake handler after `stt.unmute()`
   must be non-blocking (daemon threads / `run_coroutine_threadsafe`). A blocking call there
   eats the user's first words.
3. **`preempt_background()` is non-blocking and safe to call anytime** — but only kills
   *background* requests. The speaking request itself is never preempted.
4. **`thinking_budget_tokens: 0` must be sent on every speaking/background request**, and
   the server must run WITHOUT `--reasoning-budget`, or gemma thinks aloud on voice turns /
   per-request budgets are ignored.
5. **`message_history` is shared by reference** with WorkerManager and UnifiedIdleMindManager.
   Replace its contents with `message_history[:] = new` (in-place), never rebind the name.
6. **gemma's empty-response trap**: with reasoning on, it can burn the whole token budget in
   the thought channel. `generate_background` salvages the thinking text; distill steps must
   run `reasoning=False`; budgets are additive (answer + thinking).
7. **TTS tags**: the local voice model knows exactly 13 bracket tags (`SUPPORTED_TAGS` in
   `core/tts.py` — keep `tests/streaming_tts.py` and the system-prompt note in sync).
   Movement tags use `<angle/dist>` syntax and must be stripped before TTS *and* before the
   history append.
8. **The model sees only the curated `llm.main_tools` set on the speaking path**; the full
   catalog is only for workers and Unified Idle Mind agent loops. Tool results are capped
   (1500 chars speaking / 3000 chars agent loops) and agent conversations are compacted to
   `max_prompt_chars` — the box has 7k ctx total.
9. **The hot-conversation window (180s)** makes `generate_background` return `None`. Every
   caller must treat `None` as "fall back to cloud or skip", never as an error/empty answer.
10. **Most latency-critical config is cached at import** — restart after editing modes/prompts.
    Spoken mode, voice, volume, and follow-up controls are runtime state and apply immediately.
11. **Anti-repetition is enforced in code, not prompts**: journal duplicate gate + banned
    topics + KB `current_ongoing` aging. Fresh Gmail/Notion reads are the exception:
    identical reads may run in later sessions so new mail/page edits are visible, while
    exact duplicate calls within one agent session remain suppressed.
12. **Secrets**: Exa key is hardcoded in `tools.py`; an Inworld key default is hardcoded in
    `tts.py`; GitHub/Smithery/Gemini keys were committed historically — rotation still pending.
13. **STT mute = silence frames, not disconnect.** Don't "optimize" by closing the socket;
    the zero-latency unmute depends on it. The watchdog reconnects on 5s of Deepgram silence.
14. **Filler audio**: `ThinkingSoundPlayer` plays ONE random wav from
    `sound_effects/soundeffects/fillers/` and is stopped by `first_play_event` from the TTS
    streamer. Regenerate fillers with `python3 tests/streaming_tts.py --generate-fillers`.
15. **`play_music`/`dance` set `should_skip_followup`** — after them the mic goes straight
    back to hotword mode (no follow-up listening over the music).
16. **Periodic spoken questions** require `run_vision_update(force_trigger=True,
    force_qa=True)` AND a timer reset at the call site — the bare call returns immediately
    with `traditional_context_enabled: false`.
17. **An agent must never let a non-action sound like a completed one.** The whole
    `complex_query` guard stack (§5.2c: first-JSON truncation, placeholder-argument
    refusal, `min_tool_calls`, meaningful-summary check, explicit "do not claim it
    succeeded" failure text) exists because every one of those failures was observed
    live — a model describing WhatsApp messages that did not exist, or reporting a send
    that never happened. Latency regressions are recoverable; a confident lie about
    Alex's messages is not. Keep the guards when touching that path.
18. **WhatsApp group names only exist in `list_chats`.** `search_contacts` filters
    `@g.us` out in SQL, so any group lookup that goes only through contacts silently
    finds nothing (§5.2c).
19. **`messages.db` knows no names, and files DMs under `@lid`.** Person names come from
    the bridge's `whatsmeow_contacts`; sender ids come from `whatsmeow_lid_map` (§5.2d).
    A direct chat is *addressed* by phone but *stored* under its `@lid`, so any lookup by
    the resolved phone JID can come back empty for a conversation that plainly exists —
    always go through `jid_variants` before reporting "no messages".
20. **Always-listen context must enter history only between turns.** A cloud flush may finish
    during a reply; keep its result queued until `turn_active` and summarization are false,
    then append and re-warm once. Never let the wake thread wait for cloud processing.

---

## 9. Known Pending / Watch List

- Rotate leaked credentials (GitHub, Smithery, Gemini — in git history) and move the Exa key
  to `.env`.
- Generation speed on the box occasionally drops to ~7 tok/s (draft-MTP acceptance dips +
  106 MiB checkpoint writes mid-generation) → TTS underruns. Server-side tuning question.
- `robot/movement.py execute_movements` still uses the legacy direct-GPIO path; the `move`
  tool uses the ZMQ motor server. Consolidation candidate.
- `dance()` has a hardcoded 20s wait after music start before choreography begins.
