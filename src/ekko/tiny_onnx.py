from __future__ import annotations

import importlib
import threading
from importlib.metadata import version
from pathlib import Path

from .audio import SAMPLE_RATE, load_audio, peak_normalize
from .runtime import EkkoError, Transcript, Word, validate_transcript


def sherpa_words(
    result: object, *, audio_duration: float | None = None
) -> tuple[Word, ...]:
    tokens = list(getattr(result, "tokens", []))
    timestamps = list(getattr(result, "timestamps", []))
    if len(tokens) != len(timestamps):
        raise EkkoError("sherpa token/timestamp cardinality mismatch")
    durations = list(getattr(result, "durations", []))
    if durations and len(durations) != len(tokens):
        raise EkkoError("sherpa token/duration cardinality mismatch")
    has_durations = len(durations) == len(tokens)
    words: list[Word] = []
    current = ""
    start: float | None = None
    token_indices: list[int] = []

    def finish(end_hint: float) -> None:
        nonlocal current, start, token_indices
        if not current or start is None:
            return
        end = end_hint
        if has_durations and token_indices:
            end = max(
                float(timestamps[index]) + float(durations[index])
                for index in token_indices
            )
        if audio_duration is not None:
            end = min(end, audio_duration)
        words.append(Word(current, start, max(start, end)))

    for index, (token, timestamp) in enumerate(zip(tokens, timestamps)):
        token = str(token)
        if token == " " or token.startswith("▁"):
            finish(float(timestamp))
            current = token.strip().lstrip("▁")
            start = float(timestamp)
            token_indices = [index]
        else:
            if start is None:
                start = float(timestamp)
            current += token
            token_indices.append(index)
    if current and start is not None:
        fallback_end = float(timestamps[-1]) + 0.08 if timestamps else start
        finish(
            min(audio_duration, fallback_end)
            if audio_duration is not None
            else fallback_end
        )
    return tuple(words)


class TinyOnnxBackend:
    """Load a Tiny sherpa-onnx transducer once with validated preprocessing."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        precision: str = "int8",
        threads: int = 4,
        max_seconds: float = 600.0,
        _runtime: object | None = None,
        _runtime_version: str | None = None,
    ) -> None:
        if precision not in {"fp32", "int8"}:
            raise EkkoError(f"unsupported Tiny ONNX precision: {precision}")
        if threads < 1:
            raise EkkoError("Tiny ONNX threads must be at least 1")
        directory = Path(model_dir)
        suffix = ".int8.onnx" if precision == "int8" else ".onnx"
        files = {
            name: directory / f"{name}{suffix}"
            for name in ("encoder", "decoder", "joiner")
        }
        files["tokens"] = directory / "tokens.txt"
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise EkkoError("missing Tiny ONNX files: " + ", ".join(missing))
        if _runtime is None:
            try:
                _runtime = importlib.import_module("sherpa_onnx")
            except ImportError as error:
                raise EkkoError(
                    "Tiny ONNX support is missing; run: "
                    "uv sync --extra tiny-onnx"
                ) from error
        try:
            recognizer = _runtime.OfflineRecognizer.from_transducer(
                encoder=str(files["encoder"]),
                decoder=str(files["decoder"]),
                joiner=str(files["joiner"]),
                tokens=str(files["tokens"]),
                model_type="nemo_transducer",
                num_threads=threads,
                decoding_method="greedy_search",
            )
        except Exception as error:
            raise EkkoError(f"could not load Tiny ONNX model: {error}") from error
        self._recognizer = recognizer
        self._lock = threading.Lock()
        self.model_path = str(directory)
        self.precision = precision
        self.max_seconds = max_seconds
        self.runtime_version = _runtime_version or version("sherpa-onnx")

    def transcribe(self, audio: str | Path) -> Transcript:
        decoded = load_audio(audio, max_seconds=self.max_seconds)
        samples = peak_normalize(decoded.samples)
        try:
            with self._lock:
                if self._recognizer is None:
                    raise EkkoError("Tiny ONNX backend is closed")
                stream = self._recognizer.create_stream()
                stream.accept_waveform(SAMPLE_RATE, samples)
                self._recognizer.decode_stream(stream)
                result = stream.result
                text = str(result.text).strip()
                words = sherpa_words(result, audio_duration=decoded.duration)
        except EkkoError:
            raise
        except Exception as error:
            raise EkkoError(f"Tiny ONNX transcription failed: {error}") from error
        return validate_transcript(
            Transcript(
                text=text,
                backend="sherpa-onnx",
                words=words,
                duration=decoded.duration,
                model=self.model_path,
                runtime_version=self.runtime_version,
                device="cpu",
                timestamp_source="rnnt-native",
            ),
            require_word_timestamps=bool(text),
        )

    def close(self) -> None:
        with self._lock:
            self._recognizer = None

    def __enter__(self) -> "TinyOnnxBackend":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()