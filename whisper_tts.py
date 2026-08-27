#!/usr/bin/env python3
"""
whisper_tts.py — low-latency realtime ASR client for whisper.cpp server.

Design (see plan): three decoupled stages so audio capture never blocks on VAD
or the network.

  1. Capture        : sounddevice pushes fixed 32ms (512-sample @16k) frames.
  2. Streaming VAD  : Silero model scored per-frame (O(1)); a small state machine
                      tracks utterance start / trailing silence with a pre-roll
                      ring buffer so onsets aren't clipped.
  3. ASR worker     : a thread pool with a keep-alive requests.Session sends the
                      utterance to /inference with short-clip-tuned params.

Speculative finalization: the instant trailing silence begins (spec_silence_ms),
we fire the final ASR *in the background* and keep listening. When silence crosses
endpoint_ms we commit using that already-in-flight result. The critical path
becomes max(endpoint_ms, asr_time) ~= 200-250ms instead of silence_timeout + ASR.

Run live (on the Pi):   python3 whisper_tts.py -u http://HOST:5555/inference
Benchmark a WAV file:   python3 whisper_tts.py --wav samples/jfk.wav
"""

import argparse
import io
import sys
import time
import wave
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

try:
    import torch
    from silero_vad import load_silero_vad
    torch.set_num_threads(1)  # avoid CPU thread contention for the tiny VAD model
except ImportError:
    print("Error: 'silero-vad' or 'torch' not installed. Run: pip install torch silero-vad",
          file=sys.stderr)
    sys.exit(1)

SAMPLE_RATE = 16000
FRAME = 512                       # Silero's required window @16k == 32ms
FRAME_MS = 1000.0 * FRAME / SAMPLE_RATE


def parse_args():
    p = argparse.ArgumentParser(description="Low-latency realtime mic ASR client (Silero VAD + whisper.cpp).")
    p.add_argument("-u", "--url", default="http://192.0.2.10:5555/inference", help="Whisper server inference URL")
    p.add_argument("-d", "--device", type=str, default=None, help="Audio input device index or name")
    p.add_argument("-lang", "--language", type=str, default="en", help="Language code (default: en)")
    p.add_argument("-vt", "--vad-threshold", type=float, default=0.5, help="Silero speech probability threshold")
    p.add_argument("-ep", "--endpoint-ms", type=int, default=200,
                   help="Trailing silence (ms) that commits the utterance (default: 200)")
    p.add_argument("-sp", "--spec-silence-ms", type=int, default=80,
                   help="Trailing silence (ms) after which the speculative final ASR is fired (default: 80)")
    p.add_argument("-ms", "--min-speech-ms", type=int, default=150,
                   help="Discard utterances with less than this much speech (default: 150)")
    p.add_argument("-pr", "--preroll-ms", type=int, default=250,
                   help="Audio kept before speech onset so the first phoneme isn't clipped (default: 250)")
    p.add_argument("-mt", "--max-tokens", type=int, default=128, help="Server max_tokens cap (default: 128)")
    p.add_argument("--wav", type=str, default=None,
                   help="Non-interactive: feed this WAV file through the pipeline instead of the mic")
    p.add_argument("--no-realtime", action="store_true",
                   help="In --wav mode, feed frames as fast as possible (skews latency; default feeds at realtime pace)")
    return p.parse_args()


def audio_to_wav_bytes(audio_f32, sample_rate=SAMPLE_RATE):
    pcm = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def compute_audio_ctx(n_samples):
    """Size the encoder context to the clip. Core asserts n_mel_frames <= audio_ctx*2,
    i.e. audio_ctx >= duration_s*50. Add 30% margin, round to 64, clamp [256, 1500]."""
    duration_s = n_samples / SAMPLE_RATE
    ctx = int(np.ceil(duration_s * 50 * 1.3 / 64.0)) * 64
    return max(256, min(1500, ctx))


class AsrClient:
    """Fires transcription requests on a small thread pool over a keep-alive Session."""

    def __init__(self, url, language, max_tokens, workers=2):
        self.url = url
        self.language = language
        self.max_tokens = max_tokens
        self.session = requests.Session()
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def _do(self, wav_bytes, audio_ctx):
        payload = {
            "response_format": "json",
            "temperature": "1",
            "language": "hi",
            "vad": "false",            # client already did VAD; skip the server's loop
            "no_timestamps": "true",   # also disables token_timestamps server-side
            "single_segment": "true",
            # "best_of": "1",
            "max_tokens": str(self.max_tokens),
            # "audio_ctx": str(audio_ctx),
        }
        files = {"file": ("speech.wav", wav_bytes, "audio/wav")}
        t0 = time.time()
        try:
            r = self.session.post(self.url, data=payload, files=files, timeout=30)
            if r.status_code == 200:
                return r.json().get("text", "").strip(), time.time() - t0
        except requests.exceptions.RequestException:
            pass
        return None, time.time() - t0

    def submit(self, audio_f32):
        """Fire a transcription in the background; returns a Future of (text, asr_seconds)."""
        wav_bytes = audio_to_wav_bytes(audio_f32)
        audio_ctx = compute_audio_ctx(len(audio_f32))
        return self.pool.submit(self._do, wav_bytes, audio_ctx)


class Endpointer:
    """Per-frame Silero VAD state machine with pre-roll and speculative finalization."""

    def __init__(self, model, args, asr, on_commit):
        self.model = model
        self.args = args
        self.asr = asr
        self.on_commit = on_commit

        self.preroll_frames = max(1, int(args.preroll_ms / FRAME_MS))
        self.preroll = []                  # ring buffer of recent frames while idle
        self.in_speech = False
        self.utt = []                      # list of frames for the active utterance
        self.speech_ms = 0.0
        self.silence_ms = 0.0
        self.spec_future = None            # in-flight speculative ASR future
        self.spec_at = None                # wall-clock when speculative fire happened
        self.last_voice_at = None          # wall-clock of last voiced frame (== speech end)

    def _fire_speculative(self):
        audio = np.concatenate(self.utt) if self.utt else np.zeros(0, np.float32)
        self.spec_future = self.asr.submit(audio)
        self.spec_at = time.time()

    def _commit(self):
        end_at = self.last_voice_at or time.time()
        # If speech was too short, drop it (likely noise) and reset silently.
        if self.speech_ms < self.args.min_speech_ms:
            self._reset()
            return
        if self.spec_future is None:           # no speculative fired (very short silence); fire now
            self._fire_speculative()
        text, asr_s = self.spec_future.result()
        e2e = time.time() - end_at             # speech-end -> transcript on screen
        self.on_commit(text, asr_s, e2e, self.speech_ms / 1000.0)
        self._reset()

    def _reset(self):
        self.model.reset_states()
        self.in_speech = False
        self.utt = []
        self.preroll = []
        self.speech_ms = 0.0
        self.silence_ms = 0.0
        self.spec_future = None
        self.spec_at = None
        self.last_voice_at = None

    def process(self, frame, now):
        """Feed one 512-sample float32 frame."""
        prob = self.model(torch.from_numpy(frame), SAMPLE_RATE).item()
        voiced = prob >= self.args.vad_threshold

        if not self.in_speech:
            self.preroll.append(frame)
            if len(self.preroll) > self.preroll_frames:
                self.preroll.pop(0)
            if voiced:
                self.in_speech = True
                self.utt = list(self.preroll) + [frame]
                self.preroll = []
                self.speech_ms = FRAME_MS
                self.silence_ms = 0.0
                self.last_voice_at = now
            return

        # in_speech
        self.utt.append(frame)
        if voiced:
            self.speech_ms += FRAME_MS
            self.silence_ms = 0.0
            self.last_voice_at = now
            if self.spec_future is not None:   # speech resumed -> discard stale speculation
                self.spec_future = None
                self.spec_at = None
        else:
            self.silence_ms += FRAME_MS
            if self.spec_future is None and self.silence_ms >= self.args.spec_silence_ms:
                self._fire_speculative()
            if self.silence_ms >= self.args.endpoint_ms:
                self._commit()


# ---- ENTER benchmark (live mode): timestamp when the user presses ENTER ----
_enter_time = None
_enter_evt = threading.Event()


def _input_thread():
    global _enter_time
    while True:
        try:
            input()
            _enter_time = time.time()
            _enter_evt.set()
        except EOFError:
            break


def make_on_commit():
    def on_commit(text, asr_s, e2e_s, dur_s):
        global _enter_time
        if text:
            sys.stdout.write(f"\r\033[K🗣️  You: {text}\n")
        line = (f"[Query Complete] ASR: {asr_s*1000:.0f}ms | end→text: {e2e_s*1000:.0f}ms "
                f"| utt: {dur_s:.2f}s")
        if _enter_evt.is_set() and _enter_time is not None:
            line += f" | since ENTER: {(time.time()-_enter_time)*1000:.0f}ms"
            _enter_evt.clear()
            _enter_time = None
        print(line)
        print(">>> Listening for next query... (Press ENTER right after speaking to benchmark)")
        sys.stdout.flush()
    return on_commit


def run_live(args, model, asr):
    try:
        import sounddevice as sd
    except ImportError:
        print("Error: 'sounddevice' not installed (needed for live mic mode).", file=sys.stderr)
        sys.exit(1)

    q = queue.Queue()

    def cb(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    ep = Endpointer(model, args, asr, make_on_commit())

    print("-" * 60)
    print(f"Whisper Server: {args.url}")
    print(f"Endpoint silence: {args.endpoint_ms}ms | Speculative at: {args.spec_silence_ms}ms")
    print("Press Ctrl+C to stop.")
    print("-" * 60)
    threading.Thread(target=_input_thread, daemon=True).start()

    stream = sd.InputStream(device=args.device, channels=1, samplerate=SAMPLE_RATE,
                            blocksize=FRAME, callback=cb, dtype="float32")
    with stream:
        print("\n>>> Listening... Speak into your microphone. (Press ENTER right after speaking to benchmark) <<<\n")
        try:
            while True:
                try:
                    frame = q.get(timeout=1.0)
                except queue.Empty:
                    continue
                if len(frame) != FRAME:        # pad/truncate the last partial block
                    frame = np.resize(frame, FRAME)
                ep.process(frame.astype(np.float32), time.time())
        except KeyboardInterrupt:
            print("\nStopping Voice Assistant...")


def run_wav(args, model, asr):
    """Feed a WAV through the exact same pipeline for reproducible benchmarking."""
    with wave.open(args.wav, "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE, "WAV must be 16kHz mono"
        n = wf.getnframes()
        pcm = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0

    results = []

    def on_commit(text, asr_s, e2e_s, dur_s):
        results.append((text, asr_s, e2e_s, dur_s))
        print(f"  utt {dur_s:.2f}s | ASR {asr_s*1000:.0f}ms | end→text {e2e_s*1000:.0f}ms | {text!r}")

    ep = Endpointer(model, args, asr, on_commit)
    realtime = not args.no_realtime
    print(f"Feeding {args.wav} ({len(pcm)/SAMPLE_RATE:.1f}s) {'in realtime' if realtime else 'at full speed'}...")
    for i in range(0, len(pcm) - FRAME + 1, FRAME):
        t = time.time()
        ep.process(pcm[i:i + FRAME].copy(), t)
        if realtime:
            time.sleep(max(0, FRAME_MS / 1000.0 - (time.time() - t)))
    # flush a final commit if we ended mid-speech
    if ep.in_speech and ep.speech_ms >= args.min_speech_ms:
        ep.last_voice_at = ep.last_voice_at or time.time()
        ep._commit()

    if results:
        e2e = sorted(r[2] * 1000 for r in results)
        asr_ms = sorted(r[1] * 1000 for r in results)
        print(f"\n{len(results)} utterance(s) | end→text p50={e2e[len(e2e)//2]:.0f}ms "
              f"max={e2e[-1]:.0f}ms | ASR p50={asr_ms[len(asr_ms)//2]:.0f}ms")


def main():
    args = parse_args()
    print("Loading Silero VAD model (local CPU)...")
    model = load_silero_vad()
    asr = AsrClient(args.url, args.language, args.max_tokens)
    if args.wav:
        run_wav(args, model, asr)
    else:
        run_live(args, model, asr)


if __name__ == "__main__":
    main()
