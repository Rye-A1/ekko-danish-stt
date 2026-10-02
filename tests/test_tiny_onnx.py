from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ekko import EkkoError, TinyOnnxBackend
from ekko.audio import AudioData, peak_normalize
from ekko.tiny_onnx import sherpa_words


class FakeStream:
    def __init__(self) -> None:
        self.waveform = None
        self.result = SimpleNamespace(
            text="hej verden",
            tokens=["▁hej", "▁ver", "den"],
            timestamps=[0.08, 0.4, 0.56],
            durations=[],
        )

    def accept_waveform(self, sample_rate: int, samples: object) -> None:
        self.waveform = (sample_rate, samples)


class FakeRecognizer:
    def __init__(self) -> None:
        self.streams: list[FakeStream] = []
        self.decode_calls = 0

    def create_stream(self) -> FakeStream:
        stream = FakeStream()
        self.streams.append(stream)
        return stream

    def decode_stream(self, _stream: FakeStream) -> None:
        self.decode_calls += 1


class FakeOfflineRecognizer:
    calls: list[dict] = []
    recognizer = FakeRecognizer()

    @classmethod
    def from_transducer(cls, **kwargs):
        cls.calls.append(kwargs)
        return cls.recognizer


class TinyOnnxBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.model_dir = Path(temporary.name)
        for name in (
            "encoder.int8.onnx",
            "decoder.int8.onnx",
            "joiner.int8.onnx",
            "tokens.txt",
        ):
            (self.model_dir / name).touch()
        FakeOfflineRecognizer.calls = []
        FakeOfflineRecognizer.recognizer = FakeRecognizer()
        self.runtime = SimpleNamespace(OfflineRecognizer=FakeOfflineRecognizer)

    @patch("ekko.tiny_onnx.load_audio")
    def test_loads_once_normalizes_and_preserves_timestamps(self, load_audio) -> None:
        load_audio.return_value = AudioData(
            samples=np.array([0.0, 0.5, -1.0], dtype=np.float32),
            duration=0.72,
        )
        with TinyOnnxBackend(
            self.model_dir,
            _runtime=self.runtime,
            _runtime_version="1.13.6",
        ) as backend:
            first = backend.transcribe("first.wav")
            second = backend.transcribe("second.wav")

        self.assertEqual(len(FakeOfflineRecognizer.calls), 1)
        self.assertEqual(FakeOfflineRecognizer.recognizer.decode_calls, 2)
        sample_rate, samples = FakeOfflineRecognizer.recognizer.streams[0].waveform
        self.assertEqual(sample_rate, 16000)
        self.assertAlmostEqual(float(np.abs(samples).max()), 0.95)
        self.assertEqual(first.text, "hej verden")
        self.assertEqual(first.words[1].start, 0.4)
        self.assertEqual(first.words[1].end, 0.64)
        self.assertEqual(first.timestamp_source, "rnnt-native")
        self.assertEqual(first.runtime_version, "1.13.6")
        self.assertEqual(second.backend, "sherpa-onnx")

    def test_peak_normalization_preserves_silence(self) -> None:
        samples = np.zeros(4, dtype=np.float32)

        normalized = peak_normalize(samples)

        np.testing.assert_array_equal(normalized, samples)

    def test_rejects_missing_precision_files(self) -> None:
        with self.assertRaisesRegex(EkkoError, "missing Tiny ONNX files"):
            TinyOnnxBackend(
                self.model_dir,
                precision="fp32",
                _runtime=self.runtime,
                _runtime_version="test",
            )

    def test_rejects_partial_token_durations(self) -> None:
        result = SimpleNamespace(
            tokens=["▁hej", "▁verden"],
            timestamps=[0.1, 0.4],
            durations=[0.2],
        )

        with self.assertRaisesRegex(EkkoError, "token/duration cardinality"):
            sherpa_words(result, audio_duration=1.0)

    @patch("ekko.tiny_onnx.load_audio")
    def test_rejects_transcription_after_close(self, load_audio) -> None:
        load_audio.return_value = AudioData(
            samples=np.zeros(4, dtype=np.float32), duration=0.1
        )
        backend = TinyOnnxBackend(
            self.model_dir,
            _runtime=self.runtime,
            _runtime_version="test",
        )
        backend.close()

        with self.assertRaisesRegex(EkkoError, "closed"):
            backend.transcribe("audio.wav")


if __name__ == "__main__":
    unittest.main()