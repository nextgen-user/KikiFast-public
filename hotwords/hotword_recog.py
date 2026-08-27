import os
import sys
import time
import threading
import numpy as np
import pyaudio
import openwakeword
from openwakeword.model import Model

class HotwordRecognizer:
    def __init__(self, device_index=2):
        """
        Initialize the Hotword Recognizer (OpenWakeWord, kiki.onnx).

        :param device_index: PyAudio input device index. Falls back to default if invalid.
        """
        self.device_index = device_index
        self.chunk_size = 1280
        self.threshold = 0.5
        self.vad_threshold = 0.5
        self.inference_framework = "onnx"
        self.cooldown_seconds = 4.5
        self.last_detection_time = 0.0

        # When set, detection is suspended (e.g. while the AI is speaking) so the
        # bot's own TTS audio can't trigger its wake word and loop on itself.
        self._paused = threading.Event()
        # When set, we WANT to resume but only once the room is actually quiet
        # (i.e. the AI's speech + the speaker's audio tail has died down). This
        # detects real end-of-speech instead of using a hardcoded delay.
        self._resume_when_quiet = threading.Event()
        self._pause_started = 0.0
        # Energy gate for "is the bot done talking": frames below this RMS count
        # as silence. ~3 consecutive quiet frames (~240ms) → safe to resume.
        self.resume_rms_threshold = 500.0
        self.resume_quiet_frames = 3
        self.resume_max_wait = 2.5   # hard cap so a noisy room still resumes
        
        # Load the kiki.onnx model
        self.owwModel = Model(
            wakeword_models=["/srv/kikifast/kiki.onnx"],
            vad_threshold=self.vad_threshold,
            inference_framework=self.inference_framework,
        )
        self.model_key = next(iter(self.owwModel.models.keys()))
        print(f"[Hotword] Loaded openwakeword model key: {self.model_key}")
        
        self.pa = pyaudio.PyAudio()
        
        # PyAudio configuration
        stream_kwargs = {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": 16000,
            "input": True,
            "frames_per_buffer": self.chunk_size,
        }
        
        if self.device_index is not None:
            try:
                device_info = self.pa.get_device_info_by_index(self.device_index)
                stream_kwargs["input_device_index"] = self.device_index
                print(f"[Hotword] Using specified device index {self.device_index}: {device_info.get('name')}")
            except Exception as e:
                print(f"[Hotword] Warning: Specified device_index {self.device_index} is invalid ({e}). Falling back to default device.")
        
        self.audio_stream = self.pa.open(**stream_kwargs)

    def pause(self):
        """Suspend wake-word detection (call when the AI starts speaking)."""
        self._resume_when_quiet.clear()
        self._pause_started = time.time()
        self._paused.set()

    def request_resume(self):
        """
        Ask to resume, but only once the AI's audio has ACTUALLY stopped.

        Instead of a hardcoded delay, the listen loop watches mic energy and
        un-pauses the moment the room goes quiet (bot finished + speaker tail
        drained) — so you can say "kiki" the instant Kiki stops talking.
        """
        self._pause_started = time.time()
        self._resume_when_quiet.set()

    def resume(self):
        """Resume wake-word detection immediately (no quiet-wait)."""
        self._do_resume()

    def _do_resume(self):
        # Clear features accumulated from the AI's own audio so the next frames
        # don't carry a stale, near-threshold activation.
        try:
            self.owwModel.reset()
        except Exception:
            pass
        # Tiny debounce only — NOT the full cooldown, so the user isn't locked out.
        self.last_detection_time = time.time() - (self.cooldown_seconds - 0.3)
        self._resume_when_quiet.clear()
        self._paused.clear()

    @property
    def is_paused(self):
        return self._paused.is_set()

    def listen(self):
        """
        Generator that yields detected hotword names.
        """
        print(f"Listening for hotword: {self.model_key}")
        quiet_streak = 0
        try:
            while True:
                # Read audio frame
                data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)

                # While the AI is speaking, keep draining the mic (so the buffer
                # doesn't overflow) but DO NOT run inference — this is what stops
                # the bot's own TTS from triggering its wake word.
                if self._paused.is_set():
                    if self._resume_when_quiet.is_set():
                        # Real end-of-speech detection: resume once the mic has
                        # been quiet for a few frames, or after a hard timeout.
                        frame = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                        rms = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
                        if rms < self.resume_rms_threshold:
                            quiet_streak += 1
                        else:
                            quiet_streak = 0
                        if (quiet_streak >= self.resume_quiet_frames or
                                time.time() - self._pause_started >= self.resume_max_wait):
                            quiet_streak = 0
                            self._do_resume()
                            print("[Hotword] Resumed (speech ended).")
                    continue

                quiet_streak = 0
                audio_frame = np.frombuffer(data, dtype=np.int16)

                prediction = self.owwModel.predict(audio_frame)
                score = float(prediction[self.model_key])
                
                if score >= self.threshold:
                    current_time = time.time()
                    if current_time - self.last_detection_time >= self.cooldown_seconds:
                        print(f"\n[Hotword] Detected: {self.model_key} with score {score:.4f}")
                        self.last_detection_time = current_time
                        # Map to "heyy" since main.py expects "heyy" to wake up
                        yield "heyy"
                    
        except KeyboardInterrupt:
            print("\nStopping Hotword Recognizer...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'audio_stream') and self.audio_stream is not None:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except Exception as e:
                print(f"Error closing stream: {e}")
        if hasattr(self, 'pa') and self.pa is not None:
            try:
                self.pa.terminate()
            except Exception as e:
                print(f"Error terminating pyaudio: {e}")

def main():
    recognizer = HotwordRecognizer(device_index=None)
    try:
        for hotword in recognizer.listen():
            print(f"Action: Wake word '{hotword}' detected!")
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
