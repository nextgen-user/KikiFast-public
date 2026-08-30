# Kiki Senior Care — Canonical End-to-End Plan

> Status date: 2026-08-29 (Asia/Kolkata)  
> SIH Statement ID: 26181  
> Product focus: Kiki's voice-first senior-care experience. The caregiver app is a setup,
> monitoring, and communication surface—not the primary product.

This file is the persistent implementation roadmap and progress checklist. Read it with
`docs/ARCHITECTURE.md` before doing senior-care work. Update its checkboxes and evidence after
each phase, but mark a phase **locked** only after automated tests pass, Alex tests the
feature on the real Kiki, the observed result is documented, and the checkpoint is committed
and pushed.

## Status legend and lock rule

- `[x]` means that exact item is implemented or evidenced.
- `[ ]` means it remains incomplete, unverified, or failed its acceptance test.
- **Implemented is not the same as locked.** A phase is locked only after a real Kiki test.
- A failed real test reopens the relevant item even when unit tests pass.
- Work proceeds one phase at a time. At each gate, Codex gives Alex exact spoken actions;
  Alex interacts with Kiki; then logs, persisted state, UI state, and delivery receipts are
  inspected before moving on.

## Non-negotiable product and architecture decisions

- [x] Keep Kiki as the main product; the PWA exists primarily to configure and observe Kiki.
- [x] Preserve the existing latency-critical microphone → agent → streaming TTS pipeline.
- [x] Use the cloud Cerebras/Gemma complex care agent for care-plan formulation and guided
  care delivery. Do not pass its finished response through a second conversational LLM.
- [x] Timing workers only wake the foreground care session. Workers do not author care speech,
  simulate user replies, or conduct an exercise themselves.
- [x] A care event stores a rich goal/context/session brief, not hardcoded dialogue or an
  executable list of canned instructions. The care agent decides how to conduct the session.
- [x] Guided care remains a normal multi-turn voice conversation. The user can ask questions,
  pause, repeat, skip, change direction, report discomfort, or stop at any point.
- [x] When continuous vision is enabled, a fresh camera image is attached directly to the same
  Cerebras/Gemma Chat Completions request on every real care turn.
- [x] Continuous vision can be enabled per care event and turned on/off for the live session.
- [x] Unified Idle Mind remains Kiki's single background/proactive brain. Do not add another
  proactive-QA loop.
- [x] Unified Idle Mind may propose/formulate care-plan adaptations from concrete routine
  evidence through the complex agent; it must not casually mutate the plan or dispatch an
  emergency itself.
- [x] Machine-format or tool failures must be corrected and retried by the complex agent.
  Kiki must never claim a plan edit, schedule, measurement, or notification succeeded without
  a verified tool result.
- [x] The care plan represents the person's whole daily routine: medicines, meals, hydration,
  movement, exercise, sleep, appointments, tasks, wellbeing, memory aids, and family events.
- [x] Care formulation is general. The model decides what relevant personal context is missing
  and asks naturally; code must not hardcode an exercise questionnaire or domain dialogue.
- [x] No diagnosis of dementia or other disease. Memory support and games are assistance only.
- [x] Wearable fall detection, real step counting, continuous vitals, body temperature, and
  sleep sensing are deferred until after the internal hackathon. The software API is built now.
- [x] Privacy hardening is not an internal-hackathon blocker, but false medical, adherence,
  sensor, schedule, or message-delivery claims are never acceptable.

## Target end-to-end experience

```text
Caregiver PWA / senior voice request / Unified Idle Mind evidence
                         │
                         ▼
       Complex care agent reads full current care context
       asks naturally if essential information is missing
                         │
                         ▼
      Validated, versioned whole-day care plan is persisted
                         │
                         ▼
        Exact schedule is materialized and verified by worker
                         │
                         ▼
     Due event enters Kiki's normal foreground microphone loop
                         │
                         ▼
 Cerebras/Gemma conducts an adaptive voice session end to end
       + optional fresh camera image on every user turn
       + structured tools such as MAX30102 measurement
                         │
                         ▼
  Outcome, acknowledgement, observations, and trusted readings
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
 Caregiver timeline/PWA       Verified WhatsApp/email alert
            │
            ▼
 Compact care/environment state informs Kiki and Idle Mind
```

## Phase A — Existing senior-care audit and reliable foundation

### Implemented foundation

- [x] Dedicated senior assistant mode and warm senior-care persona.
- [x] Local structured care-plan persistence and care log.
- [x] Rich `routine_events` with schedule, session brief, adaptation metadata, and vision flag.
- [x] Complex-agent-only care-plan read/write/verification route.
- [x] Natural clarification when essential context is missing; save nothing until resolved.
- [x] Hindi/Devanagari tool input is accepted as content; machine JSON shape is validated by
  the tool/agent rather than transliterating health content merely to make saving work.
- [x] Exact runtime schedule receipt is required before Kiki promises success.
- [x] Care-plan edits resynchronize timing workers.
- [x] Duplicate senior workers and the infinite failed-worker retry storm have regression fixes.
- [x] A busy active care session defers another care event without consuming its retry budget.
- [x] Stale active sessions have timeout/maintenance handling in code and regression tests.
- [x] Completion-claim guard prevents an agent from reporting success when its required tool did
  not execute or failed.
- [x] Normal care sessions use real microphone input and record only real user/assistant turns.

### Real acceptance evidence

- [x] A one-shot neck event saved at 19:05 for 19:07 fired once at 19:07:04 and entered the
  foreground care loop.
- [x] The 19:07 neck session was genuinely multi-turn and responded to deviations/questions.
- [ ] The earlier daily neck event targeting 18:40 passed acceptance. It did **not**: the event
  was persisted at 18:54, after that day's target, and Kiki neither started it nor clearly asked
  whether the user meant tomorrow or wanted to start immediately.
- [ ] Define and validate past-time semantics: when a requested daily/one-shot time has already
  passed, the agent must explicitly resolve today/tomorrow/start-now and verify the resulting
  next trigger.
- [ ] Prove create, update, delete, cancel, and read-back for routines by voice without any false
  success statement.
- [ ] Prove restart recovery and exactly-once firing for a due event.
- [ ] Prove overlapping due events queue/defer cleanly rather than being lost.
- [x] Prove a completed/cancelled/declined session clears `active_session`. Session ending is now
  deterministic underneath the model: a spoken stop (`user_asked_to_stop`, EN+HI), a hard
  `max_session_turns` ceiling, and the idle timeout each close it, and a finished session is stripped
  of its frozen plan copy and archived to `session_history`. The live stuck "Surya Namaskar" session
  was cleared through this path; the plan file went 43 kB -> 23 kB. **Real-Kiki retest still owed.**
- [ ] Remove or migrate obsolete test routines and the old hydration event from the live care
  plan after Alex approves cleanup.
- [ ] **Phase A locked on real Kiki.**

## Phase B — Adaptive guided-care voice sessions and vision

### Implemented and observed

- [x] Timing worker queues one foreground care start and produces no dialogue itself.
- [x] Care agent receives the frozen full care plan, event brief, transcript, and current speech.
- [x] No action index, hardcoded session script, or generated participant response drives the
  interaction.
- [x] STT is muted during generation/TTS and reopened only after playback, preventing Kiki from
  answering her own speech.
- [x] Direct multimodal Cerebras/Gemma image input works through Chat Completions.
- [x] Fresh frames were observed on consecutive real turns.
- [x] Real neck-session vision correctly distinguished centered, incorrectly posed, and tilted
  head positions and adapted the spoken response.
- [x] Warm care-agent generation was roughly 1.2–1.6 seconds in the observed physical session;
  first audible speech was roughly 2.2–3.6 seconds on warm turns.

### Remaining acceptance and quality work

- [ ] Validate spoken pause, repeat, skip, substitute, vision-off, vision-on, pain, dizziness,
  cancel, and resume paths on the real Kiki.
- [ ] Ensure a guided session is substantive and outcome-oriented rather than a few shallow
  movements, while remaining personalized to known conditions and symptoms.
- [ ] Review exercise safety behavior. The tested session suggested full circular neck rotations;
  guided exercise needs appropriate conservative safety constraints without turning the session
  into hardcoded dialogue.
- [x] Make session completion deterministic enough that a finished activity does not remain
  indefinitely active or capture unrelated later conversation. Three closers, all under the model's
  wording; a forced end also clears `expect_reply` and applies even when the care turn failed.
- [ ] Confirm ordinary chat does not get trapped in care-agent routing after session completion.
- [ ] Confirm a new due care event can start after the previous one completes.
- [ ] Document the accepted real interaction in SIH test evidence.
- [ ] **Phase B locked on real Kiki.**

## Phase C — MAX30102 heart-rate companion

### Implemented

- [x] `core/hardware/max30102.py` exposes structured prepare and capture behavior while retaining its CLI.
- [x] Thread-safe care tool supports prepare, capture, status, cancel, retry, and quality reasons.
- [x] The same complex care agent dynamically walks the user through placement, stillness,
  retries, and results; sensor code contains no spoken script.
- [x] The heart-rate tool can be included in a scheduled routine event, including once-daily use.
- [x] Only GOOD/FAIR quality BPM readings are eligible for trusted history and trend tracking.
- [x] Poor/no-contact attempts are logged separately and are not treated as readings.
- [x] Uncalibrated MAX30102 SpO2 is not presented as a trusted measurement.
- [x] Completion-claim protection exists to prevent a fabricated BPM after a failed capture.

### Remaining acceptance

- [ ] Complete one physical quality-approved MAX30102 measurement through Kiki voice and confirm
  the spoken BPM exactly matches the structured sensor result.
- [ ] Confirm no-contact, motion/noise, retry, cancel, and timeout paths through real speech.
- [ ] Confirm Kiki never invents a BPM. An earlier session falsely said `78 BPM` even though no
  trusted reading existed; the regression fix must be physically re-tested.
- [ ] Confirm the trusted reading is stored once and visible in seven-day trend/history.
- [ ] Schedule a heart-rate session, let it fire, complete it, and verify its care-log outcome.
- [ ] Validate cautious wellness guidance and escalation wording without diagnosis or medicine
  changes.
- [ ] **Phase C locked on real Kiki.**

## Phase D — Reliable medicine, routine, and adherence orchestration

- [ ] Add versioned/validated structured records for profile, waking hours, medicines, meals,
  hydration, walks, exercise, sleep routines, appointments, one-time tasks, contacts, memory aids,
  companionship preferences, and notification policies while preserving legacy plan reads.
- [ ] Medicine occurrence state machine: `due → spoken → taken | snoozed | skipped | unanswered`.
- [ ] Let natural AI conversation understand Hindi/English acknowledgement and deviations; keep
  deterministic occurrence state and escalation underneath it.
- [ ] Default demo policy: `later` snoozes five minutes; unanswered retries once after five
  minutes; still unanswered alerts after ten minutes; a medicine skip alerts immediately unless
  that medicine's policy disables it.
- [ ] Lifestyle routines can request confirmation but do not escalate by default.
- [ ] Make every occurrence idempotent using event ID plus scheduled occurrence time.
- [ ] Generate caregiver summaries only from confirmed states—not inferred tablet consumption.
- [ ] Test medicine taken, later, skip, unanswered, duplicate/restart, and plan-edit cases by voice.
- [ ] **Phase D locked on real Kiki.**

## Phase E — WhatsApp, Gmail, and emergency-contact reliability

- [ ] Validate/update the WhatsApp bridge and protocol dependency against the current service.
- [ ] Expose bridge health, authenticated/logged-out state, QR pairing, reconnect state, and test
  sending.
- [ ] Distinguish accepted, delivered (when a receipt exists), and failed; never treat prose as
  delivery proof.
- [ ] Complete Gmail MCP OAuth with send scope, discover/verify the actual send tool and argument
  mapping, and remove/ignore dead connections.
- [ ] Record independent structured results for WhatsApp and email with one bounded transient retry.
- [ ] Ensure caregiver contacts and per-event channel policies come from the care plan.
- [ ] Add a deterministic voice fast path for “call for help/contact my family” that does not wait
  for the ordinary conversational LLM before dispatching configured alerts.
- [ ] Keep the user informed truthfully when one or all channels fail.
- [ ] Receive one identifiable WhatsApp test and one email test on real caregiver devices.
- [ ] Test missed-medicine, caregiver summary, concerning-health, and help-call messages end to end.
- [ ] **Phase E locked on real devices.**

## Phase F — Caregiver progressive web/mobile app

- [ ] Build a separate bilingual React + TypeScript + Vite PWA under its own frontend directory.
- [ ] Serve its production build at `/care/` from Kiki's existing Flask service with same-origin,
  versioned `/api/care/v1` APIs.
- [ ] Use Tailscale HTTPS/tailnet access for the internal-hackathon deployment.
- [ ] Home: Kiki status, next event, today's care/adherence, active risks, connector health.
- [ ] Care plan: senior profile, health context, medicines, meals, hydration, exercise, sleep,
  appointments, routines, and one-time tasks.
- [ ] Contacts: caregiver recipients, per-channel escalation preferences, and test actions.
- [ ] Memory and companionship: family events, object locations, later tasks, interests, family
  notes, approved content, and games.
- [ ] Live care: environment values, simulated/future wearable readings, trends, and alerts.
- [ ] Timeline: scheduled, spoken, acknowledged, snoozed, skipped, missed, completed, and channel
  outcomes.
- [ ] Setup: English/Hindi UI, Kiki language, location, WhatsApp QR, and Gmail OAuth.
- [ ] Cache app shell and last read-only state; queue offline edits visibly in IndexedDB and sync
  them safely using care-plan revisions.
- [ ] Mutations return the updated resource plus the new care-plan revision.
- [ ] Install on the real phone; create/edit/delete a medicine and routine; verify immediate Kiki
  synchronization; verify one offline edit synchronizes after reconnect.
- [ ] **Phase F locked on caregiver phone and Kiki.**

## Phase G — Compact environment and future-wearable context

- [ ] Add stable telemetry ingestion compatible with a simulator now and the future wearable later.
- [ ] Include timestamp, source, signal quality/confidence, device ID, and measured/inferred/user-
  reported provenance.
- [ ] Support heart rate, future trusted SpO2, body temperature, steps, activity, sleep, and last
  activity fields without requiring all fields at once.
- [x] Poll weather/AQI for configured home coordinates; retain latest-good value with age/source.
  `core/health/environment.py`, Open-Meteo forecast + air-quality, one daemon thread, no API key.
- [x] Track temperature, apparent temperature, humidity, AQI, PM2.5, and PM10.
  AQI is computed on India's **CPCB** scale (max PM2.5/PM10 sub-index) because Open-Meteo returns
  only US/European AQI, which is not the number an Indian advisory means. Documented as an estimate:
  official CPCB uses 24h averages, this uses the current hourly PM.
- [x] Mark environment readings stale after 30 minutes and unavailable after two hours. An
  unavailable snapshot carries no values at all, and a failed poll does not reset the timestamp.
- [ ] Create deterministic normal, attention, and urgent demo telemetry scenarios.
- [x] Build one ephemeral compact `CARE NOW` snapshot, target maximum 80 tokens, containing only
  noteworthy environment/vitals, next due item, medicine state, and active advisory.
  `core/health/care_snapshot.py`; measured at 68 characters live.
- [x] Inject the snapshot like the current time anchor immediately before a user turn; do not
  accumulate repeated telemetry in conversation history. Injected only when the rendered line
  CHANGES, and append-only — retracting it would invalidate the warm prefix.
- [x] Give the same compact snapshot and active senior mode to Unified Idle Mind, which previously
  had no mode awareness at all.
- [x] Use deterministic risk bands/cooldowns underneath AI wording so unchanged readings do not
  nag or grow the prompt. Bands in `environment.py`, change-gate + cooldown in `care_snapshot.py`.
- [ ] Demonstrate normal, heat, poor-AQI, and simulated-vital cases; verify personalized speech,
  expected alerts, cooldown, stale-data behavior, and no prompt growth.
- [ ] **Phase G locked on real Kiki.**

### Planned telemetry contract

```json
{
  "device_id": "demo-wearable",
  "recorded_at": "ISO-8601",
  "source": "simulator",
  "metrics": {
    "heart_rate_bpm": 78,
    "spo2_pct": 97,
    "body_temperature_c": 36.8,
    "steps_today": 1200,
    "last_activity_at": "ISO-8601"
  },
  "quality": {
    "valid": true,
    "confidence": 0.95
  }
}
```

## Phase H0 — Assistant modes: companion, senior, and the SIH health companion

The SIH statement asks for a **personal health companion**, not only an elderly-care assistant.
Kiki now ships three relevant modes rather than one.

| Mode | Persona | Capabilities | Purpose |
|---|---|---|---|
| `default` | Kiki herself — friend, wit, curiosity | *(none)* | Everyday companion. Already reaches WhatsApp, Gmail, Notion, music, memory, and vision through `complex_query`; nothing needed to be added. |
| `senior` | Hindi-locked caregiver | `care` | Elderly care. Behaviour unchanged. |
| `health_sih` | Kiki's own personality + health layer, bilingual | `care`, `environment`, `companion` | The SIH health companion: default's companionship **and** senior's whole care stack. |

- [x] Replace the five hardcoded `get_active_mode() == "senior"` gates with a declared capability
  (`assistant_modes.modes.<mode>.capabilities`) read through
  `core/runtime_controls.py::mode_has_capability()`. Nothing under `core/senior/` ever checked the
  mode, so those five gates were the entire coupling; naming the capability lets `health_sih`
  inherit the whole care stack with no duplicated wiring and no change to `senior`.
- [x] `mode_has_capability()` fails **closed**, unlike `context_enabled()`: a capability starts
  subsystems that speak on a schedule and email families.
- [x] Add the `health_sih` mode with senior's `main_tools` (already a superset of the global list)
  and a prompt that keeps Kiki's identity, answers in the language it is spoken to, and forbids
  diagnosis, invented readings, and treating an empty camera frame as a medical event.
- [x] `tests/test_mode_capabilities.py` pins the contract, the fail-closed behaviour, the shipped
  config, and the spoken names ("health", "health mode", "health companion").
- [ ] **Phase H0 locked on real Kiki**: boot `health_sih`, confirm care workers materialize, confirm
  music/WhatsApp/memory still work, confirm `senior` is unchanged, confirm `default` schedules nothing.

## Phase H — Senior companion intelligence and winning experiences

### Implementation approach

Two hard prerequisites, in this order:

1. **Session completion must be deterministic** (Phase B, `docs/SENIOR_CARE_ROADMAP.md` line ~126: `active_session` did
   not clear after the 19:07 session). Phase H adds at least two more scheduled sessions per day
   plus check-ins; sessions that do not close collide and block each other. This is a **harder**
   prerequisite for Phase H than the environment work.
2. **Environment ingestion must exist** (Phase G). There is currently *no* weather, AQI, air-quality,
   or location code anywhere in the repo, so the morning briefing, the lifestyle follow-ups, and the
   wellness indicators below cannot be built truthfully — and the SIH heat/flood/pollution claim has
   nothing behind it. Build `core/health/environment.py` (Open-Meteo forecast + air-quality: free,
   no key, gives apparent temperature, PM2.5, PM10) with latest-good retention, stale at 30 min,
   unavailable at 2 h, plus the compact `CARE NOW` snapshot injected like the time anchor.

Then most of Phase H is **data, not new code paths.** Per the architecture decisions above, a care
event stores a rich goal/context/session brief and the care agent decides how to conduct it. So the
morning briefing, evening reflection, and the food/hydration/walk/sleep follow-ups are **seeded
`routine_events` with well-written `session_brief` text**, conducted by the existing
`care_voice_agent` — no new session engine, scheduler, or speech path. Only three items need genuinely
new mechanism: Idle Mind care awareness, the memory-aid sections, and the dashboard.

- [x] Morning briefing: greeting, weather/AQI, medicine/routine overview, appointments, and a
  relevant family event. Seeded `routine_event`; the care agent reads live conditions from the
  `CURRENT OUTSIDE CONDITIONS` block and is told never to estimate a missing reading.
- [x] Evening reflection: confirmed care outcomes, gentle missed-item follow-up, and tomorrow's
  appointments. Reads the care log and `session_history`; explicitly forbidden from inferring that
  a scheduled medicine was taken.
- [x] Food, hydration, walking, lifestyle, and sleep follow-ups based on the care plan and compact
  state—supportive and non-shaming. `hydration_checkin`, `movement_checkin`, `sleep_winddown`;
  the movement brief tells the agent to advise staying IN on a bad-heat or bad-air day.
- [ ] Inactivity check-ins based on time, available activity evidence, and personal baseline; never
  equate “not visible to camera” with a medical event. *(Unified Idle Mind)*
- [ ] Mood check-ins and optional caregiver-summary inclusion. Mood is covered inside the evening
  reflection; the standalone Idle-Mind check-in and caregiver-summary inclusion remain open.
- [ ] Boredom cure: approved jokes, word games, reminiscence prompts, music, and simple memory/care
  games chosen naturally by Unified Idle Mind. *(NEW MECHANISM: `unified_idle_mind.py` currently has
  no mode awareness and no care context at all. Needs the compact snapshot, the active mode, and a
  companionship action type. Keep `update_care_plan`/`alert_family` blocked to it — that is correct.)*
- [ ] Structured non-diagnostic memory assistance for appointments, medication schedules, family
  events, object locations explicitly told to Kiki, later tasks, and life stories. *(NEW: care_plan
  sections + tools, following the existing `add_to_list`/`_edit_item` patterns.)*
- [ ] Caregiver family notes delivered naturally once or at a scheduled time.
- [ ] Personal wellness dashboard with trends and explainable heat, respiratory, and cardiovascular
  wellness indicators based only on available data. *(NEW: extend the existing Flask webui on :8090
  with a care view rather than blocking on the Phase F PWA. Needs Phase G for the indicators.)*
- [ ] Rehearse a full day-in-the-life with Alex as the senior and a phone as caregiver, in
  `health_sih` mode.
- [ ] **Phase H locked on real Kiki.**

## Phase I — Reliability, SIH documentation, and final video

- [ ] Run an accelerated multi-hour soak with overlapping events, restart recovery, active-session
  timeout, connector disconnect/reconnect, API outage, stale telemetry, and duplicate prevention.
- [ ] Maintain focused backend, frontend, contract, fake-clock, multimodal, and hardware-boundary
  tests without regressing the normal Kiki voice experience.
- [ ] Keep `docs/ARCHITECTURE.md` synchronized with implemented architecture.
- [ ] Maintain `docs/SIH_SENIOR_CARE_FEATURES.md` using verified claims only.
- [ ] Add `docs/sih/TEST_EVIDENCE.md`: phase, date, commit, exact interaction, expected result,
  observed result, logs, and user acceptance.
- [ ] Add `docs/sih/DEMO_RUNBOOK.md`: setup, reset, fallback, and exact 3–5 minute shot list.
- [ ] Add an SIH requirement-to-feature matrix and clearly label deferred wearable features.
- [ ] Complete the planned video sequence twice without manual JSON/config edits.
- [ ] **Phase I and internal-hackathon software scope locked.**

### Intended final video sequence

1. Caregiver opens/installs the PWA and creates a medicine or routine.
2. Kiki immediately reads back and understands the updated whole-day plan.
3. A scheduled guided session starts on time and adapts through voice and live vision.
4. A scheduled MAX30102 spot-check guides the user and records a trusted BPM.
5. The senior says “later,” then confirms a medicine; the timeline updates truthfully.
6. Poor-AQI/heat or simulated-vital data produces a personalized spoken advisory.
7. A missed medicine or help request produces verified WhatsApp and email outcomes.
8. Kiki recalls an appointment/object/family event and starts a companionship activity.
9. The caregiver sees the timeline, trends, and daily summary.
10. End with the latency-critical architecture and the future wearable story.

## Phase J — Post-internal-hackathon wearable work (explicitly deferred)

- [ ] Wearable step counter and activity classification.
- [ ] Continuous heart-rate and validated SpO2 integration.
- [ ] Body-temperature and sleep sensing.
- [ ] Wearable fall detection and distress confirmation.
- [ ] Location-enabled emergency assistance when the user permits it.
- [ ] On-device/offline anomaly and disaster-risk processing.
- [ ] Strong privacy controls: consent, encryption, retention, export/delete, audit trail, and
  offline emergency fallbacks.

## Current “what is left” summary

The hardest conversational foundation is built: senior mode, complex care-plan formulation,
verified worker materialization, foreground care-agent sessions, microphone ownership, direct
Gemma image feedback, and the MAX30102 software tool. The real 19:07 guided neck session proved
that the core voice-and-vision concept works.

The immediate unfinished gate is **reliability acceptance**, not the PWA: resolve past-time
scheduling semantics, prove edit/delete/restart/overlap behavior, close sessions cleanly, and
physically validate a trusted heart-rate reading.

The second gap is **environment data**. The SIH statement is about heat waves, floods, pollution
events, and early warning, and there is currently no weather, AQI, air-quality, or location code
anywhere in the repository. Phase G is therefore not an optional enrichment phase — it is the
difference between a care assistant and the health companion the statement asks for, and Phase H's
briefing, lifestyle follow-ups, and wellness indicators all depend on it. Phase H0 (mode
capabilities and `health_sih`) is done and unblocks the rest without touching senior mode. After those are locked, proceed in order through
medicine occurrence state, WhatsApp/Gmail, caregiver PWA, compact environment/telemetry context,
senior companion experiences, and SIH hardening. Wearable-only sensing and fall detection remain
deferred.

## Evidence log

| Date | Commit/state | Feature | Result | Lock impact |
|---|---|---|---|---|
| 2026-08-29 | `a6bd5ba` | Direct multimodal care agent | Fresh image on consecutive turns; one Cerebras/Gemma request per care turn | Implementation proven; physical gate still required |
| 2026-08-29 | runtime | 18:40 daily neck event | Event persisted at 18:54; did not trigger or clarify past-time intent | Scheduling gate remains open |
| 2026-08-29 | runtime | 19:07 one-shot neck event | Fired once at 19:07:04; adaptive visual voice session ran for eight turns | Upcoming one-shot path accepted; session completion still open |
| 2026-08-29 | `d0fe638` | Care reliability checkpoint | Current scheduler/session/claim fixes and tests checkpointed and pushed | Requires targeted real-device retest |
| 2026-08-29 | `1b69ef0` | Mode capabilities + `health_sih` | Five `== "senior"` gates replaced by `mode_has_capability`; `health_sih` added; 11 new tests pass, suite 655 passed / 9 pre-existing failures unchanged | Phase H0 implemented; real-boot gate still required |
| 2026-08-29 | `138fb13` | Environment provider (Phase G-lite) | Live Delhi read: 29.2C / feels 34.8C / 75% RH, PM2.5 120.6, PM10 338.1 -> CPCB 301 "very poor" (US AQI 376 for comparison). 33 new tests; suite 688 passed / 9 pre-existing failures unchanged | Weather+AQI implemented; snapshot injection and real-boot gate still required |

## Next exact gate

1. End/cancel the currently active neck session and verify `active_session` clears.
2. Ask Kiki to create a new routine two minutes in the future; read back its verified next trigger.
3. Let it fire exactly once, complete/cancel it by voice, and restart Kiki to confirm it does not
   fire again.
4. Ask for a time already passed today and verify Kiki explicitly asks tomorrow vs start-now.
5. Test update and delete by voice and verify plan plus workers agree.
6. Run one real MAX30102 prepare/capture/retry/result session and compare spoken BPM to persisted
   structured data.
7. Record results here and in SIH evidence. Only then lock Phases A–C and begin medicine state.
