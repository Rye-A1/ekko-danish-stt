from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ekko import Punctuator, Transcript, Word


class FakeEncoding:
    def __init__(self, words: list[str]) -> None:
        self.ids = [101]
        self.word_ids = [None]
        for index, word in enumerate(words):
            pieces = 2 if word.startswith("long") else 1
            self.ids.extend([index + 1] * pieces)
            self.word_ids.extend([index] * pieces)
        self.ids.append(102)
        self.word_ids.append(None)


class FakeTokenizer:
    def encode(self, words: list[str], *, is_pretokenized: bool) -> FakeEncoding:
        return FakeEncoding(words)


class FakeSession:
    def __init__(self) -> None:
        self.lengths: list[int] = []

    def run(self, _outputs, inputs):
        length = inputs["input_ids"].shape[1]
        self.lengths.append(length)
        logits = np.zeros((1, length, 10), dtype=np.float32)
        logits[:, :, 5] = 1.0
        return [logits]


class PunctuatorTest(unittest.TestCase):
    def make_punctuator(self, *, max_length: int = 8, overlap: int = 2) -> Punctuator:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        labels = {index: "O" for index in range(10)}
        labels[5] = ".|U"
        (directory / "config.json").write_text(
            json.dumps({"id2label": labels}), encoding="utf-8"
        )
        return Punctuator(
            directory,
            model_filename="test.onnx",
            max_length=max_length,
            overlap_words=overlap,
            _session=FakeSession(),
            _tokenizer=FakeTokenizer(),
        )

    def test_processes_every_word_across_overlapping_windows(self) -> None:
        punctuator = self.make_punctuator()
        words = [f"ord{index}" for index in range(17)]

        rendered = punctuator.punctuate_words(words)

        self.assertEqual(len(rendered), len(words))
        self.assertTrue(all(word.startswith("Ord") and word.endswith(".") for word in rendered))
        self.assertEqual(rendered[0], "Ord0.")
        self.assertEqual(rendered[-1], "Ord16.")

    def test_indexes_long_multisubword_input_without_dropping_words(self) -> None:
        punctuator = self.make_punctuator(max_length=12, overlap=2)
        words = [f"long{index}" for index in range(37)]

        rendered = punctuator.punctuate_words(words)

        self.assertEqual(len(rendered), len(words))
        self.assertEqual(punctuator.session.lengths[0], 12)
        self.assertTrue(all(length <= 12 for length in punctuator.session.lengths))
        self.assertEqual(rendered[-1], "Long36.")

    def test_preserves_word_timestamps_and_raw_text(self) -> None:
        punctuator = self.make_punctuator(max_length=16)
        transcript = Transcript(
            text="hej verden",
            backend="parakeet.cpp",
            words=(Word("hej", 0.1, 0.3), Word("verden", 0.4, 0.8)),
        )

        result = punctuator.apply(transcript)

        self.assertEqual(result.text, "Hej. Verden.")
        self.assertEqual(result.raw_text, "hej verden")
        self.assertEqual(result.words[0].start, 0.1)
        self.assertEqual(result.words[1].end, 0.8)

    def test_sentence_case_matches_browser_rule(self) -> None:
        rendered = Punctuator._stabilize_sentence_case(
            [
                "hej.", "hvordan", "går", "det?", "det", "er", "f.eks.",
                "sådan.", "men", "CPR", "virker.", "iPhone",
            ]
        )
        self.assertEqual(
            rendered,
            [
                "Hej.", "Hvordan", "går", "det?", "Det", "er", "f.eks.",
                "sådan.", "Men", "CPR", "virker.", "iPhone",
            ],
        )


if __name__ == "__main__":
    unittest.main()
