#!/usr/bin/env python3
"""
WebRTC phone <-> RPi bridge.

The PHONE (Android Chrome) is the WebRTC *sender*: it captures camera + mic and
streams them to the RPi in real time. The RPi (this script) is the *receiver*.
The RPi can also push audio back to the phone speaker (e.g. TTS).

Why this is low-CPU on the RPi:
  * Audio echo-cancellation / noise-suppression is done in the phone browser
    (getUserMedia constraints) -- the RPi never runs an AEC filter.
  * Video frames are decoded only at a throttled rate (VISION_FPS) for the
    vision system; the phone is asked to send a small resolution.
  * Sensor data (gyro / accel / magnetometer) travels over a DataChannel, not
    a media track, so it is nearly free.

Access from the phone over USB-C:
    adb reverse tcp:8080 tcp:8080
    # then open http://localhost:8080 in Chrome on the phone
localhost is a "secure context", so camera/mic/motion sensors work with no HTTPS.
"""

import argparse
import asyncio
import fractions
import json
import logging
import os
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
from aiohttp import web
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError
from av import AudioFrame, AudioResampler

try:
    import cv2  # video preview on the Pi
except ImportError:
    cv2 = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("webrtc")

ROOT = Path(__file__).parent
STATIC = ROOT / "static"
REC_DIR = ROOT / "recordings"
REC_DIR.mkdir(exist_ok=True)

# How often (per second) we hand a decoded camera frame to the vision system.
# Keep this low on a Pi -- decoding every frame is what burns CPU.
VISION_FPS = 5.0

# Audio sent to the phone speaker is 48 kHz mono, 20 ms frames (WebRTC native).
SPK_RATE = 48000
SPK_FRAME_SAMPLES = SPK_RATE // 50  # 960 samples = 20 ms

# Microphone capture / processing.
MIC_RATE = 48000
RECORD_SECONDS = 5
PITCH_FACTOR = 1.4  # >1 raises pitch ("chipmunk"); 1.0 = unchanged


# --------------------------------------------------------------------------- #
#  Shared application state (the rest of KikiFast can read/write this)
# --------------------------------------------------------------------------- #
class State:
    def __init__(self):
        self.latest_frame = None        # most recent camera frame (numpy BGR)
        self.latest_frame_ts = 0.0
        self.sensors = {}               # last sensor payload from the phone
        self.events = []                # button presses etc.
        self.speaker = None             # SpeakerTrack instance (set on connect)
        self.last_sensor_print = 0.0    # throttle for console sensor printout


STATE = State()


# --------------------------------------------------------------------------- #
#  Speaker: RPi -> phone audio. Push WAV files or raw PCM into a queue.
# --------------------------------------------------------------------------- #
class SpeakerTrack(AudioStreamTrack):
    """An outgoing audio track. Plays whatever PCM is queued, silence otherwise."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._timestamp = 0
        self._leftover = b""

    async def play_wav(self, path: str):
        """Queue a WAV file for playback to the phone (resampled to 48k mono)."""
        loop = asyncio.get_event_loop()
        frames = await loop.run_in_executor(None, self._load_wav, path)
        for f in frames:
            await self._queue.put(f)

    async def play_pcm(self, pcm: bytes):
        """Queue raw s16 mono 48 kHz PCM for playback to the phone."""
        await self._queue.put(pcm)

    @staticmethod
    def _load_wav(path: str):
        with wave.open(path, "rb") as wf:
            n, ch, rate = wf.getnframes(), wf.getnchannels(), wf.getframerate()
            raw = wf.readframes(n)
        layout = "mono" if ch == 1 else "stereo"
        src = AudioFrame(format="s16", layout=layout, samples=n)
        src.sample_rate = rate
        src.planes[0].update(raw)
        resampler = AudioResampler(format="s16", layout="mono", rate=SPK_RATE)
        out = []
        for rf in resampler.resample(src):
            out.append(bytes(rf.planes[0]))
        return out

    async def recv(self):
        # 20 ms cadence
        if hasattr(self, "_start"):
            self._timestamp += SPK_FRAME_SAMPLES
            wait = self._start + self._timestamp / SPK_RATE - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = time.time()

        data = self._leftover
        need = SPK_FRAME_SAMPLES * 2  # bytes (s16 mono)
        while len(data) < need and not self._queue.empty():
            data += self._queue.get_nowait()
        if len(data) >= need:
            chunk, self._leftover = data[:need], data[need:]
        else:
            self._leftover = b""
            chunk = data + b"\x00" * (need - len(data))  # pad with silence

        frame = AudioFrame(format="s16", layout="mono", samples=SPK_FRAME_SAMPLES)
        frame.sample_rate = SPK_RATE
        frame.planes[0].update(chunk)
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, SPK_RATE)
        return frame


# --------------------------------------------------------------------------- #
#  Consumers for the incoming phone tracks
# --------------------------------------------------------------------------- #
async def consume_video(track):
    """Decode camera frames at VISION_FPS and stash the latest one."""
    interval = 1.0 / VISION_FPS
    next_t = 0.0
    while True:
        try:
            frame = await track.recv()
        except MediaStreamError:
            break
        now = time.time()
        if now < next_t:
            continue  # drop frame without converting -> cheap
        next_t = now + interval
        # to_ndarray is the expensive part; only done VISION_FPS times/sec
        STATE.latest_frame = frame.to_ndarray(format="bgr24")
        STATE.latest_frame_ts = now
    log.info("video track ended")


def pitch_shift(pcm: bytes, factor: float) -> bytes:
    """Raise pitch by `factor` via resampling (classic 'chipmunk' effect).

    Cheap and dependency-light (numpy only). Resampling to fewer samples and
    replaying at the original rate shifts the pitch up; it also shortens the
    clip slightly, which is fine for this demo.
    """
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n = len(audio)
    if n == 0:
        return pcm
    new_n = max(1, int(n / factor))
    idx = np.linspace(0, n - 1, new_n)
    shifted = np.interp(idx, np.arange(n), audio)
    return shifted.astype(np.int16).tobytes()


def save_wav(path: Path, pcm: bytes, rate: int = MIC_RATE):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def process_chunk(pcm: bytes):
    """Runs in a thread: pitch-shift the 5 s mic clip and save both WAVs."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_wav(REC_DIR / f"{stamp}_raw.wav", pcm)
    shifted = pitch_shift(pcm, PITCH_FACTOR)
    out_path = REC_DIR / f"{stamp}_pitch.wav"
    save_wav(out_path, shifted)
    return shifted, out_path


async def consume_audio(track):
    """Capture mic audio in 5 s chunks, pitch-shift it, save a WAV, and play it
    back out to the phone speaker."""
    resampler = AudioResampler(format="s16", layout="mono", rate=MIC_RATE)
    needed = MIC_RATE * RECORD_SECONDS * 2  # bytes for one chunk (s16 mono)
    buf = bytearray()
    loop = asyncio.get_event_loop()

    while True:
        try:
            frame = await track.recv()
        except MediaStreamError:
            break
        for rf in resampler.resample(frame):
            buf += bytes(rf.planes[0])

        if len(buf) >= needed:
            chunk = bytes(buf[:needed])
            del buf[:needed]
            shifted, out_path = await loop.run_in_executor(None, process_chunk, chunk)
            log.info("recorded 5 s -> pitch-shifted -> %s (-> phone speaker)",
                     out_path.name)
            if STATE.speaker:
                await STATE.speaker.play_pcm(shifted)
    log.info("audio track ended")


async def display_loop():
    """Show the phone camera feed in a window on the Pi (~15 fps)."""
    if cv2 is None:
        log.warning("opencv not installed -- no video window. pip install opencv-python")
        return
    if not os.environ.get("DISPLAY"):
        log.warning("no DISPLAY set -- skipping video window (run on the Pi desktop)")
        return
    win = "KikiFast phone cam"
    last_ts = 0.0
    while True:
        await asyncio.sleep(1 / 15)
        f = STATE.latest_frame
        if f is not None and STATE.latest_frame_ts != last_ts:
            last_ts = STATE.latest_frame_ts
            cv2.imshow(win, f)
        cv2.waitKey(1)


# --------------------------------------------------------------------------- #
#  Signaling / HTTP
# --------------------------------------------------------------------------- #
pcs = set()


async def offer(request):
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # No STUN/TURN needed -- phone and Pi are on the same USB link via localhost.
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    pcs.add(pc)

    speaker = SpeakerTrack()
    STATE.speaker = speaker
    pc.addTrack(speaker)  # RPi -> phone speaker

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(msg):
            try:
                data = json.loads(msg)
            except (ValueError, TypeError):
                return
            kind = data.get("t")
            if kind == "sensor":
                STATE.sensors = data
                now = time.time()
                if now - STATE.last_sensor_print >= 0.5:  # throttle console output
                    STATE.last_sensor_print = now
                    h = data.get("heading")
                    log.info(
                        "gyro=(%6.1f,%6.1f,%6.1f) deg/s | "
                        "accel=(%5.1f,%5.1f,%5.1f) m/s^2 | compass=%s",
                        data.get("gx", 0), data.get("gy", 0), data.get("gz", 0),
                        data.get("ax", 0), data.get("ay", 0), data.get("az", 0),
                        f"{h:.0f}deg" if h is not None else "n/a",
                    )
            elif kind == "event":
                STATE.events.append(data)
                log.info("phone event: %s", data.get("name"))

    @pc.on("track")
    def on_track(track):
        log.info("track received: %s", track.kind)
        if track.kind == "video":
            asyncio.ensure_future(consume_video(track))
        elif track.kind == "audio":
            asyncio.ensure_future(consume_audio(track))

    @pc.on("connectionstatechange")
    async def on_state():
        log.info("connection state: %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


async def api_speak(request):
    """POST a JSON {"wav": "/path/to/file.wav"} to play audio on the phone."""
    data = await request.json()
    if STATE.speaker and data.get("wav"):
        await STATE.speaker.play_wav(data["wav"])
        return web.json_response({"ok": True})
    return web.json_response({"ok": False, "error": "no speaker / no wav"}, status=400)


async def api_sensors(request):
    return web.json_response(STATE.sensors)


async def index(request):
    return web.FileResponse(STATIC / "index.html")


async def on_startup(app):
    app["display"] = asyncio.ensure_future(display_loop())


async def on_shutdown(app):
    app["display"].cancel()
    if cv2 is not None:
        cv2.destroyAllWindows()
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()


def build_app():
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_post("/speak", api_speak)
    app.router.add_get("/sensors", api_sensors)
    app.router.add_static("/static/", STATIC)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    log.info("Serving on http://%s:%d  (adb reverse tcp:%d tcp:%d)",
             args.host, args.port, args.port, args.port)
    web.run_app(build_app(), host=args.host, port=args.port, access_log=None)
