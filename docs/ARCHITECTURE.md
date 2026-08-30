# KikiFast architecture

## Overview

KikiFast is a modular companion-robot platform for Raspberry Pi. It combines
voice interaction, vision, hardware control, health-oriented features, and a
local operations dashboard.

## System areas

- `main.py` coordinates the interactive runtime.
- `kiki_boot.py` prepares device services and hardware.
- `core/` contains the included device, health, vision, and runtime modules.
- `robot/` and `hotwords/` contain robot-facing adapters.
- `tools_and_config/` provides configuration and tool dispatch.
- `webui/` provides local monitoring and control.
- `integrations/` contains optional standalone integrations.

## Runtime flow

At a high level, microphone, camera, and sensor inputs enter the runtime, which
selects an appropriate response or device action. Audio, display, movement, and
dashboard outputs are coordinated around the foreground interaction.

Hardware and external services are optional and configured independently.
Writable state belongs under `runtime/` and is excluded from version control.

## Design principles

- Keep credentials and personal data outside the repository.
- Keep hardware adapters isolated from application orchestration.
- Prefer explicit configuration over machine-specific paths.
- Degrade safely when optional hardware or services are unavailable.
- Treat health features as assistive, not diagnostic.

Detailed runtime implementations and internal operational design are maintained
separately.
