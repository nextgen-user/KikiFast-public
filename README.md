# KikiFast

KikiFast is a low-latency, multimodal companion-robot runtime for Raspberry Pi.
It combines wake-word detection, streaming speech recognition, local or cloud
language models, speech synthesis, vision, memory, scheduled tasks, robot
controls, a local dashboard, and optional senior-care workflows.

The repository is designed to keep source code separate from credentials,
personal data, generated media, and device-specific runtime state.

> KikiFast is an experimental robotics project, not a medical device. Heart-rate
> and senior-care features must not be used for diagnosis, emergency monitoring,
> or as a substitute for professional care.

## How it fits together

```text
microphone → wake word / STT → main orchestration loop → LLM
                                      │                ├─ tools and workers
camera / sensors ─────────────────────┤                ├─ memory and care flows
                                      │                └─ TTS → speaker
                                      └─ local dashboard and observability
```

Foreground speech has priority over background model work. When a local
llama.cpp-compatible server is used, KikiFast can preempt background generation
and rewarm the conversation prefix to reduce response latency.

## Highlights

- Streaming local or cloud speech pipeline with interruption handling.
- OpenWakeWord activation and whisper.cpp-compatible transcription.
- Local llama.cpp, Cerebras, Groq, OpenRouter, Gemini, and Vertex AI routing.
- Camera snapshots, face events, gesture controls, and ZMQ robot control.
- Long-term memory, conversation summaries, and background workers.
- Senior-care plans, guided sessions, environment context, and MAX30102 support.
- Flask-based local dashboard and structured observability events.
- Optional MCP, skills, WhatsApp, and WebRTC integrations.

## Repository layout

| Path | Purpose |
|---|---|
| `main.py` | Primary voice-loop entry point |
| `kiki_boot.py` | Raspberry Pi boot and service orchestration |
| `core/` | Runtime-facing health, vision, hardware, and device modules |
| `tools_and_config/` | Safe configuration template, logging, paths, and tool dispatch |
| `robot/`, `hotwords/` | Robot-facing and wake-word adapters |
| `webui/` | Local operations dashboard |
| `integrations/` | Optional standalone integrations |
| `scripts/` | Diagnostics, demos, and operational utilities |
| `tests/` | Automated checks for the included components |
| `docs/` | High-level architecture, handoff, roadmap, and feature documentation |

## Requirements

- Python 3.11 or newer.
- Linux; Raspberry Pi OS is the primary deployment target.
- PortAudio and ALSA utilities for microphone/speaker access.
- `mpv`, NetworkManager (`nmcli`), and Bluetooth utilities for full hardware use.
- Optional local services: a llama.cpp-compatible server, TTS server,
  whisper.cpp server, camera service, and Hailo/controller process.

Some Python packages require native system libraries. On Debian/Raspberry Pi OS,
install the relevant PortAudio, BLAS, camera, GPIO, and audio development packages
for your hardware before installing `requirements.txt`.

## Quick start

```bash
git clone <your-repository-url> KikiFast
cd KikiFast

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cp .env.example .env
cp tools_and_config/config.example.json tools_and_config/config.json
```

Edit `.env` for providers you use and adjust `tools_and_config/config.json` for
your services and hardware. Both files are excluded from Git.

Run the primary process from the repository root:

```bash
python main.py
```

On a configured Raspberry Pi deployment, the boot orchestrator can be exercised
manually with:

```bash
python kiki_boot.py --force
```

## Configuration and credentials

The checked-in `tools_and_config/config.example.json` contains safe localhost
defaults. If `config.json` does not exist, regular module imports use this
template read-only; saving settings creates the local `config.json`.

Secrets are read from environment variables. Never place API keys in Python,
JSON configuration, tests, or documentation. Vertex AI uses an external service
account file:

```bash
GOOGLE_APPLICATION_CREDENTIALS=/secure/path/service-account.json
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global
```

Store that JSON outside the repository. The supported variables and optional
deployment overrides are documented in `.env.example`.

Writable state defaults to `runtime/`, including logs, recordings, care plans,
conversation memory, and worker state. Override it with
`KIKIFAST_RUNTIME_DIR` when deployments need a separate data volume.

## Optional external integrations

KikiFast does not vendor downloaded toolchains, third-party repositories, or
WhatsApp session databases.

- Install `whisper-cli` and its model separately, then set
  `KIKIFAST_WHISPER_CLI` and `KIKIFAST_WHISPER_MODEL`.
- Install a compatible WhatsApp MCP checkout separately under
  `integrations/whatsapp-mcp` or set `KIKIFAST_WHATSAPP_ROOT`. Keep the feature
  disabled in configuration until it is installed and authenticated.
- The WebRTC bridge has its own instructions in
  `integrations/webrtc/README.md`.

## Tests

Install development dependencies and run the maintained suite from the root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Tests isolate writable care state with temporary directories. Tests requiring
an unversioned Hailo `care_gate.py` skip automatically unless
`KIKIFAST_CARE_GATE_PATH` is set.

## Security and privacy

- Keep `.env`, service-account files, device identifiers, session databases,
  face images, recordings, conversations, and care data out of Git.
- Bind dashboards and hardware-control services to trusted interfaces only.
- Review tool permissions before enabling shell, Python, messaging, or
  self-extension capabilities.
- See [SECURITY.md](SECURITY.md) for reporting and credential-response guidance.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes should include focused tests,
must not depend on live personal state, and must pass the credential scan and
test suite before review.

## License

KikiFast is available under the [MIT License](LICENSE).
