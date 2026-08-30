# KikiFast WebRTC phone link

Real-time, low-latency camera + microphone stream from an **Android phone** to a
**Raspberry Pi 5**, with optional audio back to the phone speaker, sensor
telemetry, and a full-screen always-on web UI. Connected over **USB-C + adb**.

## Why this design is low-CPU and low-latency

| Concern | How it's handled |
|---|---|
| Echo / noise cancellation | Done in the **phone browser** (`echoCancellation`, `noiseSuppression`, `autoGainControl`). The Pi runs **no** audio DSP. |
| Video CPU on the Pi | Phone sends **640×480 @ 20 fps**; the Pi only `to_ndarray()`-decodes at `VISION_FPS` (default 5/s). Other frames are dropped before conversion. |
| Latency | Direct WebRTC over the USB link (`localhost`), **no STUN/TURN**, no transcoding. |
| Sensors | Sent over a WebRTC **DataChannel**, not a media track. |

## Install (on the Pi)

PyAV needs ffmpeg libraries:

```bash
sudo apt update
sudo apt install -y python3-pip ffmpeg libavdevice-dev pkg-config
pip install -r requirements.txt
```

## Run

```bash
./run.sh
```

`run.sh` calls `adb reverse tcp:8080 tcp:8080` so the phone's `localhost:8080`
maps to the Pi. Then on the phone open **Chrome → http://localhost:8080** and tap
**Start Stream**.

> Using `http://localhost` is the trick that makes the camera, mic, and motion
> sensors work **without HTTPS** — `localhost` is a browser "secure context".

## What the phone does

- Streams **camera + mic** to the Pi (echo-cancelled, noise-suppressed).
- **Full-screen**, **screen stays awake** (Wake Lock API).
- **Flip camera** front/back, **mute** mic, **fullscreen** toggle.
- Mock buttons + "Hello World" text (button presses are sent to the Pi).
- Streams **gyroscope, accelerometer, magnetometer/compass heading**.

## Hooking into KikiFast

`server.py` exposes a shared `STATE`:

```python
STATE.latest_frame      # newest camera frame as a numpy BGR array (for vision)
STATE.latest_frame_ts   # timestamp of that frame
STATE.sensors           # latest {heading, gx,gy,gz, ax,ay,az, mx,my,mz}
STATE.events            # list of button events from the phone
STATE.speaker           # SpeakerTrack -> await STATE.speaker.play_wav(path)
```

- **Vision**: read `STATE.latest_frame` from `core/vision/vision_handler.py`.
- **Speaker (TTS to phone)**: `await STATE.speaker.play_wav("reply.wav")`, or
  `POST /speak {"wav": "/abs/path/reply.wav"}`.
- **Mic → STT**: add your recognizer inside `consume_audio()`.

## Tuning

- `VISION_FPS` in `server.py` — frames/sec handed to vision (raise for smoother
  vision, lower for less CPU).
- Resolution / frame rate — `getMedia()` constraints in `static/app.js`.
