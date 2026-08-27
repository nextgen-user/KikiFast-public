#!/usr/bin/env bash
# Start the WebRTC bridge and wire up the USB link to the phone.
set -e
cd "$(dirname "$0")"

PORT=8080

# If a phone is connected via adb, forward its localhost:PORT -> this Pi.
# This makes http://localhost:PORT a "secure context" on the phone, so the
# camera / mic / motion sensors are allowed without any HTTPS certificate.
if command -v adb >/dev/null && adb get-state >/dev/null 2>&1; then
  adb reverse tcp:$PORT tcp:$PORT && echo "adb reverse set: phone localhost:$PORT -> Pi:$PORT"
else
  echo "No adb device. Connect the phone via USB and enable USB debugging."
fi

exec python3 server.py --port $PORT
