#!/bin/bash
#
# KIKI Smart Robot - Full Stack Startup Script
# Starts all KIKI processes with proper virtual environments
#

set -e

# =============================================================================
# Audio Environment Setup (required for PulseAudio speaker access)
# =============================================================================
export XDG_RUNTIME_DIR=/run/user/1000
export PULSE_SERVER=unix:/run/user/1000/pulse/native
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export DISPLAY=:0
export XAUTHORITY=~/.Xauthority

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Store all PIDs for cleanup
PIDS=()

# Cleanup function for graceful shutdown
cleanup() {
    log "Received shutdown signal, stopping all processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log "Stopping process $pid"
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    
    # Wait a moment for graceful shutdown
    sleep 2
    
    # Force kill any remaining
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log "Force killing process $pid"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    
    log "All processes stopped"
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup SIGTERM SIGINT SIGHUP

log "=========================================="
log "Starting KIKI Smart Robot Services"
log "=========================================="

# -----------------------------------------------------------------------------
# Process 0: Hailo Follower (Face Recognition with Webcam)
# -----------------------------------------------------------------------------
log "Starting Hailo Follower (webcam face recognition)..."
(
    source /srv/hailo-apps/venv_hailo_apps/bin/activate
    cd /srv/kikifast-navigation
    exec python hailo_follower_webcam_only.py --input usb --show-fps
) &
PIDS+=($!)
log "Hailo Follower started with PID ${PIDS[-1]}"

# Wait for Hailo to initialize
sleep 5

# -----------------------------------------------------------------------------
# Process 1: LiveKit Server
# -----------------------------------------------------------------------------
log "Starting LiveKit Server..."
(
    export LIVEKIT_PLI_THROTTLE_MS=50
    export LIVEKIT_CONGESTION_PROBE_MS=50
    exec livekit-server --dev
) &
PIDS+=($!)
log "LiveKit Server started with PID ${PIDS[-1]}"

# Wait for LiveKit server to be ready
sleep 10

# -----------------------------------------------------------------------------
# KIKI-SMART Scripts (using kiki venv)
# -----------------------------------------------------------------------------
log "Activating KIKI venv and starting KIKI-SMART scripts..."

# Process 2: Agent.py
log "Starting Agent.py..."
(
    source /srv/kikifast/.venv/bin/activate
    cd /srv/kikifast
    exec python agent.py dev
) &
PIDS+=($!)
log "Agent.py started with PID ${PIDS[-1]}"

sleep 3

# Process 3: Mint Token
log "Starting Mint Token..."
(
    source /srv/kikifast/.venv/bin/activate
    cd /srv/kikifast
    exec python mint_token.py
) &
PIDS+=($!)
log "Mint Token started with PID ${PIDS[-1]}"

sleep 10

# Process 4: LiveKit Client
log "Starting LiveKit Client..."
(
    source /srv/kikifast/.venv/bin/activate
    cd /srv/kikifast
    exec python livekit_clc.py
) &
PIDS+=($!)
log "LiveKit Client started with PID ${PIDS[-1]}"

# -----------------------------------------------------------------------------
# Keep script running and monitor processes
# -----------------------------------------------------------------------------
log "=========================================="
log "All services started successfully!"
log "PIDs: ${PIDS[*]}"
log "=========================================="

# Wait for any process to exit
wait -n "${PIDS[@]}" 2>/dev/null || true

# If we get here, a process exited - log and cleanup
log "A process exited unexpectedly, shutting down all services..."
cleanup
