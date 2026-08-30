# Kiki Senior Care — SIH Software Feature Record

Target: SIH Statement 26181. This file distinguishes code that exists now from
wearable/PWA work planned after the internal hackathon.

## Implemented in Kiki software

- Voice-first Hindi/English senior companion using Kiki's existing low-latency
  speaking, listening, memory, camera, worker scheduler, and idle cognition.
- A persistent whole-day care plan. A routine event stores its schedule plus a
  rich `session_brief`: the intended outcome and known person/caregiver context.
  It is not an executable action list, fixed dialogue, generic alarm, or
  proactive-QA loop. The cloud care model owns the actual interaction.
- The complex care agent exclusively owns plan reads/edits. It reads the current
  plan first, resolves context, retries correctable JSON/format failures, avoids
  duplicate events, and asks the person only for genuine ambiguity such as an
  unknown AM/PM time or medication details.
- Transactional schedule verification: every new/edited routine returns a
  machine-readable item receipt and is checked against the live worker manager.
  Kiki promises a trigger only when that exact worker is active, including the
  next trigger time in Asia/Kolkata.
- Multi-turn guided sessions freeze the complete care-plan snapshot at their
  start and persist only real exchanges: heard user speech, delivered assistant
  speech, tool use, and grounded visual observations. There is no action index
  and Kiki never creates the user's reply. The care model produces one spoken
  turn, listens, and naturally handles repeat/skip/slow down/change/pause/stop.
- All scheduled care items use this same path. Canonical routine events and
  legacy reminder/exercise records only open a foreground session; the worker
  cannot write dialogue, conduct steps, or speak through WorkerBrain.
- Per-event continuous vision switch. With `continuous_vision=true`, Kiki pulls
  a fresh clean camera snapshot before every care-model turn and attaches the
  JPEG directly to the same Cerebras Chat Completions request as the care plan,
  transcript and latest speech. Gemma sees the image and produces the spoken
  turn in one pass; no preliminary Gemini call or second conversational LLM is
  involved. Other events leave it off, and the person can override it for only
  the live session.
- Camera capture uses the Hailo server's bounded `/clean` snapshot endpoint,
  samples four 1280x720 frames, and picks the sharpest. It does not consume the
  MJPEG stream, which can block for 30 seconds during pipeline restarts.
- Unified Idle Mind can propose/formulate routine changes through the same
  complex care agent, but only from concrete repeated-routine evidence. It
  cannot directly mutate the plan, infer medication/dosage, create generic care
  workers, or duplicate a recently handled intent.
- Legacy medicine/hydration/meal reminders, guided exercises, family contacts,
  emergency family alerts, caregiver email summaries, approved music/topics,
  and care logs remain available behind the richer routine model.
- MAX30102 heart-rate checks can be requested by voice or scheduled once daily
  as a `vitals` routine. The complex agent dynamically conducts clear-sensor,
  placement, stillness, capture, retry and explanation steps; the sensor tool
  emits no dialogue. Only signal-quality-approved BPM values enter the persistent
  seven-day personal trend. The uncalibrated SpO2 estimate is intentionally not
  presented or trended.
- When a routine becomes due, its timing-only worker queues `main.py`. The
  existing foreground lifecycle mutes STT, pauses the wake recognizer, calls the
  care model, sends its exact answer directly to TTS, and only then reopens the
  microphone. This prevents Kiki from hearing and answering its own care prompt.
  The local speaking LLM is bypassed, eliminating a redundant generation and
  preventing verified BPM or trigger times from being rewritten.

## Automated and live integration evidence (2026-08-29)

- Focused care/health/agent/worker suite: 161 passed.
- Full repository `tests/` suite: 534 passed; nine existing unrelated failures remain in
  history-view marker handling and GPIO/startup mocks.
- Direct multimodal camera/API chain passed with an isolated temporary care
  plan: fresh Hailo frame + frozen plan/transcript/current speech -> one
  Cerebras Gemma response -> persisted spoken turn and visual observation. No
  Gemini/Vertex call occurred. The cold first turn took 5.49 s end-to-end
  (1.25 s Cerebras); the next warm turn took 1.27 s (0.91 s Cerebras).
- Two consecutive turns each sampled four clean 1280x720 snapshots. Their
  selected JPEG hashes were distinct, and Gemma returned two grounded visual
  observations consistent with the corresponding live frames.
- The real microphone/TTS acceptance run still requires the configured Bluetooth
  Bluetooth speaker to be powered and connected; boot intentionally treats it as
  required hardware.

## Planned or requires real integration validation

- Caregiver progressive web/mobile setup UI and sync API.
- Real WhatsApp and Gmail caregiver delivery validation with the intended SIH
  contacts/accounts.
- Wearable sensor ingestion: steps, calibrated SpO2, temperature, activity,
  sleep and later fall detection. The current wired MAX30102 BPM path is implemented.
- Medical/environmental context injection (AQI, heat, humidity and wearable
  vitals) after the data sources and compact prompt contract are finalized.

## Phase-1 physical demo acceptance test

1. Restart Kiki into senior mode.
2. Say in Hindi: “किकी, रोज़ शाम सात बजकर बीस मिनट पर मेरी पूरी पीठ की
   एक्सरसाइज़ करवाना। शुरू करने से पहले दर्द पूछना, फिर एक-एक स्टेप करवाना।”
3. Kiki must create a substantive `session_brief` through the complex agent and
   say the verified next trigger time. She must not merely claim it was saved.
4. Ask: “मेरी आज की पूरी दिनचर्या क्या है?” Kiki must read back the event.
5. Start the event immediately for testing. Kiki should use the full care context
   to begin naturally, speak once, wait for the real answer, and continue from
   what was actually said—not deliver both sides or the whole session as one
   monologue. No fixed readiness/pain questionnaire is required.
6. For a vision-enabled exercise, move posture between two replies. Each care
   turn's persisted transcript must contain a distinct fresh Gemma observation,
   and Kiki's next instruction must remain consistent with visible evidence.
7. Say “आज दर्द है, रोक दो।” The session must end/decline, record the response,
   and continuous care vision must switch off.

Do not record the SIH final video until this acceptance test passes on the real
camera, microphone, TTS, scheduler and selected cloud credentials.
