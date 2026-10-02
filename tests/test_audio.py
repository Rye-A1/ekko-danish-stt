from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from ekko.audio import SAMPLE_RATE, load_audio, peak_normalize
from ekko.runtime import EkkoError


class AudioTest(unittest.TestCase):
    def test_downmixes_and_resamples(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "stereo.wav"
        samples = np.column_stack(
            (np.full(2205, 0.25, dtype=np.float32), np.full(2205, -0.25, dtype=np.float32))
        )
        sf.write(path, samples, 22_050, subtype="PCM_16")

        decoded = load_audio(path)

        self.assertEqual(decoded.samples.dtype, np.float32)
        self.assertEqual(decoded.samples.ndim, 1)
        self.assertAlmostEqual(len(decoded.samples) / SAMPLE_RATE, 0.1, places=3)
        self.assertLess(float(np.abs(decoded.samples).max()), 0.001)

    def test_rejects_audio_over_profile_limit(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "long.wav"
        sf.write(path, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

        with self.assertRaisesRegex(EkkoError, "accepts at most"):
            load_audio(path, max_seconds=0.5)

    def test_rejects_empty_audio(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "empty.wav"
        sf.write(path, np.array([], dtype=np.float32), SAMPLE_RATE)

        with self.assertRaisesRegex(EkkoError, "no samples"):
            load_audio(path)

    def test_rejects_non_finite_audio(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "nan.wav"
        sf.write(
            path,
            np.array([0.0, np.nan], dtype=np.float32),
            SAMPLE_RATE,
            subtype="FLOAT",
        )

        with self.assertRaisesRegex(EkkoError, "non-finite"):
            load_audio(path)

    def test_peak_normalization_suppresses_numerical_noise(self) -> None:
        samples = np.array([1e-8, -1e-8], dtype=np.float32)

        normalized = peak_normalize(samples)

        np.testing.assert_array_equal(normalized, np.zeros_like(samples))

    def test_peak_normalization_rejects_non_finite_samples(self) -> None:
        with self.assertRaisesRegex(EkkoError, "non-finite"):
            peak_normalize(np.array([np.inf], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()