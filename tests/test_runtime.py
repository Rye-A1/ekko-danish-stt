from __future__ import annotations

import json
import gc
import subprocess
import threading
import unittest
import weakref
from unittest.mock import patch

from ekko import (
    EkkoError,
    TinyBackend,
    TinyCLIBackend,
    Transcript,
    Word,
    validate_transcript,
)
from ekko.audio import AudioData


class FakeParakeetAPI:
    abi_version = 6
    runtime_version = "0.5.0"

    def __init__(self) -> None:
        self.loads = 0
        self.calls = 0
        self.frees = 0

    def load(self, model: str) -> int:
        self.loads += 1
        return 42

    def free(self, context: int) -> None:
        self.frees += 1

    def transcribe_pcm_json(self, context: int, samples: object) -> object:
        self.calls += 1
        return {
            "text": "hej verden",
            "words": [
                {"w": "hej", "start": 0.08, "end": 0.32, "conf": 0.9},
                {"w": "verden", "start": 0.4, "end": 0.88, "conf": 0.8},
            ],
        }


class BlockingParakeetAPI(FakeParakeetAPI):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def transcribe_pcm_json(self, context: int, samples: object) -> object:
        self.started.set()
        if not self.release.wait(timeout=2):
            raise RuntimeError("test did not release inference")
        return super().transcribe_pcm_json(context, samples)


class TinyNativeBackendTest(unittest.TestCase):
    @patch("ekko.audio.load_audio")
    def test_loads_once_and_reuses_context(self, load_audio) -> None:
        load_audio.return_value = AudioData(samples=[0.0], duration=1.25)
        api = FakeParakeetAPI()
        with TinyBackend("tiny.gguf", _api=api) as backend:
            first = backend.transcribe("first.wav")
            second = backend.transcribe("second.wav")

        self.assertEqual(first.text, "hej verden")
        self.assertEqual(second.words[1].end, 0.88)
        self.assertEqual(first.runtime_version, "0.5.0")
        self.assertEqual(first.timestamp_source, "rnnt-native")
        self.assertEqual(first.duration, 1.25)
        self.assertEqual((api.loads, api.calls, api.frees), (1, 2, 1))

    def test_rejects_wrong_abi(self) -> None:
        api = FakeParakeetAPI()
        api.abi_version = 5

        with self.assertRaisesRegex(EkkoError, "expected 6"):
            TinyBackend("tiny.gguf", _api=api)

    def test_abandoned_backend_frees_context(self) -> None:
        api = FakeParakeetAPI()
        backend = TinyBackend("tiny.gguf", _api=api)
        reference = weakref.ref(backend)

        del backend
        gc.collect()

        self.assertIsNone(reference())
        self.assertEqual(api.frees, 1)

    @patch("ekko.audio.load_audio")
    def test_close_waits_for_in_flight_transcription(self, load_audio) -> None:
        load_audio.return_value = AudioData(samples=[0.0], duration=1.0)
        api = BlockingParakeetAPI()
        backend = TinyBackend("tiny.gguf", _api=api)
        result: list[object] = []
        transcribe_thread = threading.Thread(
            target=lambda: result.append(backend.transcribe("audio.wav"))
        )
        closed = threading.Event()
        close_thread = threading.Thread(target=lambda: (backend.close(), closed.set()))

        transcribe_thread.start()
        self.assertTrue(api.started.wait(timeout=1))
        close_thread.start()
        self.assertFalse(closed.wait(timeout=0.05))
        api.release.set()
        transcribe_thread.join(timeout=2)
        close_thread.join(timeout=2)

        self.assertTrue(closed.is_set())
        self.assertEqual(len(result), 1)
        self.assertEqual(api.frees, 1)


class TranscriptValidationTest(unittest.TestCase):
    def test_rejects_invalid_timestamp_structures(self) -> None:
        invalid_words = [
            (Word("hej", float("nan"), 0.2),),
            (Word("hej", -0.1, 0.2),),
            (Word("hej", 0.3, 0.2),),
            (Word("hej", 0.1, 1.1),),
            (Word("hej", 0.4, 0.5), Word("verden", 0.3, 0.6)),
            (Word("hej", 0.1, 0.2, confidence=1.1),),
            (Word("", 0.1, 0.2),),
        ]
        for words in invalid_words:
            with self.subTest(words=words), self.assertRaises(EkkoError):
                validate_transcript(
                    Transcript(
                        text="hej verden",
                        backend="test",
                        words=words,
                        duration=1.0,
                    )
                )

    def test_requires_words_for_timestamped_text(self) -> None:
        with self.assertRaisesRegex(EkkoError, "no word timestamps"):
            validate_transcript(
                Transcript(text="hej", backend="test", duration=1.0),
                require_word_timestamps=True,
            )

class TinyCLIBackendTest(unittest.TestCase):
    @patch("ekko.audio.load_audio")
    @patch("ekko.runtime.subprocess.run")
    def test_parses_native_json_with_timestamps(self, run, load_audio) -> None:
        load_audio.return_value = AudioData(samples=[0.0], duration=1.0)
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "text": "hej verden",
                    "words": [
                        {"w": "hej", "start": 0.08, "end": 0.32, "conf": 0.9},
                        {"w": "verden", "start": 0.4, "end": 0.88, "conf": 0.8},
                    ],
                }
            ),
            stderr="",
        )

        result = TinyCLIBackend("tiny.gguf", threads=4).transcribe("audio.wav")

        self.assertEqual(result.text, "hej verden")
        self.assertEqual(result.words[1].start, 0.4)
        run.assert_called_once_with(
            [
                "parakeet-cli",
                "transcribe",
                "--model",
                "tiny.gguf",
                "--input",
                "audio.wav",
                "--timestamps",
                "--json",
                "--threads",
                "4",
            ],
            capture_output=True,
            text=True,
            timeout=7200.0,
        )

    @patch("ekko.audio.load_audio")
    @patch("ekko.runtime.subprocess.run")
    def test_reports_runtime_failure(self, run, load_audio) -> None:
        load_audio.return_value = AudioData(samples=[0.0], duration=1.0)
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=2, stdout="", stderr="model rejected"
        )

        with self.assertRaisesRegex(EkkoError, "model rejected"):
            TinyCLIBackend("tiny.gguf").transcribe("audio.wav")

    @patch("ekko.audio.load_audio")
    @patch("ekko.runtime.subprocess.run")
    def test_reports_runtime_timeout(self, run, load_audio) -> None:
        load_audio.return_value = AudioData(samples=[0.0], duration=1.0)
        run.side_effect = subprocess.TimeoutExpired("parakeet-cli", 2)

        with self.assertRaisesRegex(EkkoError, "exceeded the 2s timeout"):
            TinyCLIBackend("tiny.gguf", timeout_seconds=2).transcribe("audio.wav")

    @patch("ekko.audio.load_audio")
    @patch("ekko.runtime.subprocess.run", side_effect=FileNotFoundError("missing"))
    def test_reports_missing_runtime(self, _run, load_audio) -> None:
        load_audio.return_value = AudioData(samples=[0.0], duration=1.0)
        with self.assertRaisesRegex(EkkoError, "could not start"):
            TinyCLIBackend("tiny.gguf").transcribe("audio.wav")

    def test_is_a_context_manager(self) -> None:
        backend = TinyCLIBackend("tiny.gguf")

        with backend as entered:
            self.assertIs(entered, backend)


if __name__ == "__main__":
    unittest.main()