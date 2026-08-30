# Contributing to KikiFast

Thank you for helping improve KikiFast.

## Development setup

1. Create and activate a Python virtual environment.
2. Install `requirements-dev.txt`.
3. Copy the example environment and configuration files locally.
4. Keep hardware services disabled unless a test explicitly requires them.

## Pull requests

- Keep changes focused and explain user-visible behavior.
- Add or update automated tests for behavior changes.
- Run `python -m pytest` and `python -m compileall -q .`.
- Do not commit credentials, personal data, runtime state, generated recordings,
  downloaded models/toolchains, or messaging session databases.
- Document new environment variables in `.env.example` and new configuration
  keys in `tools_and_config/config.example.json`.

Hardware-only changes should include the device, operating-system version, and
manual verification performed. Never make network-dependent tests part of the
default unit suite.
