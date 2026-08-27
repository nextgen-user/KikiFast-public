import unittest

import numpy as np

from core.noise_suppression import (
    RNNOISE_FRAME,
    RNNoiseSuppressor,
    RNNoiseUnavailable,
    StreamingDecimator3,
)


class StreamingDecimator3Tests(unittest.TestCase):
    def test_each_rnnoise_frame_becomes_exactly_160_samples(self):
        decimator = StreamingDecimator3()
        for _ in range(20):
            output = decimator.process(np.zeros(RNNOISE_FRAME, dtype=np.float32))
            self.assertEqual(output.shape, (160,))
            self.assertEqual(output.dtype, np.float32)

    def test_passband_speech_tone_is_preserved(self):
        decimator = StreamingDecimator3()
        sample_index = np.arange(RNNOISE_FRAME * 20)
        input_audio = np.sin(2 * np.pi * 1_000 * sample_index / 48_000).astype(
            np.float32
        )
        output = np.concatenate(
            [
                decimator.process(input_audio[start : start + RNNOISE_FRAME])
                for start in range(0, input_audio.size, RNNOISE_FRAME)
            ]
        )
        # Ignore causal-filter startup. RMS should remain close to the source.
        output_rms = np.sqrt(np.mean(output[320:] ** 2))
        self.assertGreater(output_rms, 0.65)
        self.assertLess(output_rms, 0.75)

    def test_above_nyquist_noise_is_strongly_attenuated(self):
        decimator = StreamingDecimator3()
        sample_index = np.arange(RNNOISE_FRAME * 20)
        input_audio = np.sin(2 * np.pi * 12_000 * sample_index / 48_000).astype(
            np.float32
        )
        output = np.concatenate(
            [
                decimator.process(input_audio[start : start + RNNOISE_FRAME])
                for start in range(0, input_audio.size, RNNOISE_FRAME)
            ]
        )
        output_rms = np.sqrt(np.mean(output[320:] ** 2))
        self.assertLess(output_rms, 0.01)


class RNNoiseRuntimeTests(unittest.TestCase):
    def test_native_runtime_processes_one_frame(self):
        try:
            suppressor = RNNoiseSuppressor()
        except RNNoiseUnavailable as exc:
            self.skipTest(str(exc))
        try:
            output, probability = suppressor.process(
                np.zeros(RNNOISE_FRAME, dtype=np.int16)
            )
            self.assertEqual(output.shape, (RNNOISE_FRAME,))
            self.assertEqual(output.dtype, np.float32)
            self.assertIsInstance(probability, float)
            self.assertTrue(suppressor.active)
        finally:
            suppressor.close()


if __name__ == "__main__":
    unittest.main()
