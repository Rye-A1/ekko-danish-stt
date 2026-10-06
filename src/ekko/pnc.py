from __future__ import annotations

import importlib
import json
import threading
from bisect import bisect_right
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .runtime import EkkoError, Transcript, validate_transcript


class Punctuator:
    """Reusable int8 ONNX punctuation and capitalization model."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        model_filename: str = "pnc.int8.onnx",
        max_length: int = 128,
        overlap_words: int = 12,
        _session: object | None = None,
        _tokenizer: object | None = None,
    ) -> None:
        if max_length < 8:
            raise ValueError("max_length must be at least 8")
        if overlap_words < 0:
            raise ValueError("overlap_words cannot be negative")
        self.max_length = max_length
        self.overlap_words = overlap_words

        directory = Path(model_dir)
        if _session is None or _tokenizer is None:
            try:
                onnxruntime = importlib.import_module("onnxruntime")
                tokenizers = importlib.import_module("tokenizers")
            except ImportError as error:
                raise EkkoError(
                    "PnC support is missing; run: uv sync --extra pnc"
                ) from error
            model_path = directory / model_filename
            if not model_path.is_file():
                raise EkkoError(f"missing PnC model: {model_path}")
            try:
                _session = onnxruntime.InferenceSession(
                    str(model_path), providers=["CPUExecutionProvider"]
                )
                _tokenizer = tokenizers.Tokenizer.from_file(
                    str(directory / "tokenizer.json")
                )
            except Exception as error:
                raise EkkoError(f"could not load PnC files in {directory}: {error}") from error

        self.session = _session
        self.tokenizer = _tokenizer
        self._lock = threading.Lock()
        try:
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            self.id2label = {int(key): value for key, value in config["id2label"].items()}
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            raise EkkoError(f"invalid PnC config in {directory}: {error}") from error

    def _encoded_length(self, words: Sequence[str]) -> int:
        return len(self.tokenizer.encode(list(words), is_pretokenized=True).ids)

    def _predict_window(self, words: Sequence[str]) -> list[str]:
        numpy = importlib.import_module("numpy")
        encoding = self.tokenizer.encode(list(words), is_pretokenized=True)
        ids = encoding.ids[: self.max_length]
        word_ids = encoding.word_ids[: self.max_length]
        try:
            logits = self.session.run(
                None,
                {
                    "input_ids": numpy.array([ids], dtype=numpy.int64),
                    "attention_mask": numpy.array([[1] * len(ids)], dtype=numpy.int64),
                },
            )[0]
        except Exception as error:
            raise EkkoError(f"PnC inference failed: {error}") from error
        predictions = logits[0].argmax(-1)

        rendered = list(words)
        seen: set[int] = set()
        for index, word_index in enumerate(word_ids):
            if word_index is None or word_index in seen:
                continue
            seen.add(word_index)
            label = self.id2label[int(predictions[index])]
            upper = label.endswith("|U")
            mark = label[:-2] if upper else label
            word = words[word_index]
            rendered[word_index] = (
                (self._capitalize_initial_preserving_case(word) if upper else word)
                + ("" if mark == "O" else mark)
            )
        return rendered

    def punctuate_words(self, words: Sequence[str]) -> list[str]:
        if not words:
            return []

        with self._lock:
            return self._stabilize_sentence_case(self._punctuate_words_locked(words))

    @staticmethod
    def _capitalize_initial_preserving_case(word: str) -> str:
        if any(character.isupper() or character.istitle() for character in word):
            return word
        for index, character in enumerate(word):
            if character.isalpha():
                return word[:index] + character.title() + word[index + 1 :]
        return word

    @staticmethod
    def _stabilize_sentence_case(words: list[str]) -> list[str]:
        def is_abbreviation(word: str) -> bool:
            return any(
                word[index - 1].isalpha() and word[index + 1].isalpha()
                for index in range(1, len(word) - 1)
                if word[index] == "."
            )

        if not words:
            return words
        rendered = words.copy()
        rendered[0] = Punctuator._capitalize_initial_preserving_case(rendered[0])
        for index, word in enumerate(rendered[:-1]):
            if not word.endswith((".", "!", "?")) or is_abbreviation(word):
                continue
            rendered[index + 1] = Punctuator._capitalize_initial_preserving_case(
                rendered[index + 1]
            )
        return rendered

    def _punctuate_words_locked(self, words: Sequence[str]) -> list[str]:
        encoding = self.tokenizer.encode(list(words), is_pretokenized=True)
        token_counts = [0] * len(words)
        special_tokens = 0
        for word_id in encoding.word_ids:
            if word_id is None:
                special_tokens += 1
            else:
                token_counts[word_id] += 1
        prefix = [0]
        for count in token_counts:
            prefix.append(prefix[-1] + count)

        def max_window_end(start: int) -> int:
            token_budget = prefix[start] + self.max_length - special_tokens
            return min(
                len(words),
                start + self.max_length,
                max(start + 1, bisect_right(prefix, token_budget) - 1),
            )

        output: list[str] = []
        next_word = 0
        while next_word < len(words):
            context_start = max(0, next_word - self.overlap_words)
            window_end = max_window_end(context_start)
            if window_end <= next_word:
                context_start = next_word
                window_end = max_window_end(context_start)
            rendered = self._predict_window(words[context_start:window_end])
            final_window = window_end == len(words)
            emit_end = (
                window_end
                if final_window
                else max(next_word + 1, window_end - self.overlap_words)
            )
            output.extend(
                rendered[next_word - context_start : emit_end - context_start]
            )
            next_word = emit_end

        if len(output) != len(words):
            raise EkkoError(
                f"PnC changed word cardinality: {len(words)} -> {len(output)}"
            )
        return output

    def __call__(self, text: str) -> str:
        return " ".join(self.punctuate_words(text.split()))

    def apply(self, transcript: Transcript) -> Transcript:
        if transcript.words:
            rendered = self.punctuate_words([word.text for word in transcript.words])
            words = tuple(
                replace(word, text=text)
                for word, text in zip(transcript.words, rendered)
            )
            text = " ".join(rendered)
        else:
            words = transcript.words
            text = self(transcript.text)
        return validate_transcript(
            replace(transcript, text=text, raw_text=transcript.text, words=words),
            require_word_timestamps=bool(transcript.words and text),
        )
