from __future__ import annotations

import ctypes
import json
import math
import subprocess
import threading
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


class EkkoError(RuntimeError):
    """A local inference backend failed or returned invalid output."""


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    confidence: float | None = None


@dataclass(frozen=True)
class Transcript:
    text: str
    backend: str
    words: tuple[Word, ...] = ()
    raw_text: str | None = None
    duration: float | None = None
    model: str | None = None
    runtime_version: str | None = None
    device: str | None = None
    timestamp_source: str | None = None
    truncated: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "backend": self.backend,
            "words": [asdict(word) for word in self.words],
            "raw_text": self.raw_text,
            "duration": self.duration,
            "model": self.model,
            "runtime_version": self.runtime_version,
            "device": self.device,
            "timestamp_source": self.timestamp_source,
            "truncated": self.truncated,
        }


def validate_transcript(
    transcript: Transcript, *, require_word_timestamps: bool = False
) -> Transcript:
    if transcript.duration is not None and (
        not math.isfinite(transcript.duration) or transcript.duration < 0
    ):
        raise EkkoError("transcript has an invalid audio duration")
    if transcript.words and not transcript.text:
        raise EkkoError("transcript has word timestamps but no text")
    if require_word_timestamps and transcript.text and not transcript.words:
        raise EkkoError("transcript text has no word timestamps")
    previous_start = 0.0
    previous_end = 0.0
    for index, word in enumerate(transcript.words):
        if not word.text.strip():
            raise EkkoError(f"transcript word {index} is empty")
        if not math.isfinite(word.start) or not math.isfinite(word.end):
            raise EkkoError(f"transcript word {index} has non-finite timestamps")
        if not 0 <= word.start <= word.end:
            raise EkkoError(f"transcript word {index} has an invalid span")
        if transcript.duration is not None and word.end > transcript.duration + 1e-6:
            raise EkkoError(f"transcript word {index} exceeds the audio duration")
        if word.start < previous_start or word.end < previous_end:
            raise EkkoError(f"transcript word {index} is non-monotonic")
        if word.confidence is not None and (
            not math.isfinite(word.confidence) or not 0 <= word.confidence <= 1
        ):
            raise EkkoError(f"transcript word {index} has invalid confidence")
        previous_start = word.start
        previous_end = word.end
    return transcript


def _tiny_transcript(
    payload: object,
    *,
    model: str | None = None,
    runtime_version: str | None = None,
    duration: float | None = None,
) -> Transcript:
    try:
        if not isinstance(payload, dict):
            raise TypeError("top-level JSON value is not an object")
        text = str(payload["text"]).strip()
        words = tuple(
            Word(
                text=str(word["w"]),
                start=float(word["start"]),
                end=float(word["end"]),
                confidence=float(word["conf"]) if "conf" in word else None,
            )
            for word in payload.get("words", [])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EkkoError(f"invalid parakeet.cpp JSON: {payload!r}") from error
    return validate_transcript(
        Transcript(
            text=text,
            backend="parakeet.cpp",
            words=words,
            duration=duration,
            model=model,
            runtime_version=runtime_version,
            timestamp_source="rnnt-native",
        ),
        require_word_timestamps=bool(text),
    )


class _ParakeetBindings(Protocol):
    abi_version: int
    runtime_version: str

    def load(self, model: str) -> int: ...
    def free(self, context: int) -> None: ...
    def transcribe_pcm_json(self, context: int, samples: object) -> object: ...


class _ParakeetCAPI:
    REQUIRED_ABI = 6

    def __init__(self, library: str | Path) -> None:
        try:
            self._library = ctypes.CDLL(str(library))
        except OSError as error:
            raise EkkoError(
                f"could not load parakeet.cpp library {library}: {error}"
            ) from error
        self._library.parakeet_capi_abi_version.argtypes = []
        self._library.parakeet_capi_abi_version.restype = ctypes.c_int
        self._library.parakeet_version.argtypes = []
        self._library.parakeet_version.restype = ctypes.c_char_p
        self._library.parakeet_capi_load.argtypes = [ctypes.c_char_p]
        self._library.parakeet_capi_load.restype = ctypes.c_void_p
        self._library.parakeet_capi_free.argtypes = [ctypes.c_void_p]
        self._library.parakeet_capi_free.restype = None
        self._library.parakeet_capi_transcribe_path_json.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self._library.parakeet_capi_transcribe_path_json.restype = ctypes.c_void_p
        self._library.parakeet_capi_transcribe_pcm_batch_json.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._library.parakeet_capi_transcribe_pcm_batch_json.restype = ctypes.c_void_p
        self._library.parakeet_capi_free_string.argtypes = [ctypes.c_void_p]
        self._library.parakeet_capi_free_string.restype = None
        self._library.parakeet_capi_last_error.argtypes = [ctypes.c_void_p]
        self._library.parakeet_capi_last_error.restype = ctypes.c_char_p
        self.abi_version = int(self._library.parakeet_capi_abi_version())
        self.runtime_version = self._library.parakeet_version().decode(
            "ascii", "replace"
        )

    def load(self, model: str) -> int:
        return int(self._library.parakeet_capi_load(model.encode()) or 0)

    def free(self, context: int) -> None:
        self._library.parakeet_capi_free(context)

    def transcribe_pcm_json(self, context: int, samples: object) -> object:
        lengths = (ctypes.c_int * 1)(len(samples))
        pointer = self._library.parakeet_capi_transcribe_pcm_batch_json(
            context,
            samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            lengths,
            1,
            16_000,
            0,
        )
        if not pointer:
            detail = self._library.parakeet_capi_last_error(context)
            message = detail.decode("utf-8", "replace") if detail else "unknown error"
            raise EkkoError(f"parakeet.cpp transcription failed: {message}")
        try:
            payload = json.loads(ctypes.string_at(pointer))
        finally:
            self._library.parakeet_capi_free_string(pointer)
        if not isinstance(payload, list) or len(payload) != 1:
            raise EkkoError(f"invalid parakeet.cpp batch JSON: {payload!r}")
        return payload[0]


class TinyBackend:
    """Load a parakeet.cpp GGUF once through its stable C API."""

    def __init__(
        self,
        model: str | Path,
        *,
        library: str | Path | None = None,
        offline: bool = False,
        max_seconds: float = 3600.0,
        _api: _ParakeetBindings | None = None,
    ) -> None:
        if _api is None:
            from .artifacts import ensure_parakeet_library

            _api = _ParakeetCAPI(
                library or ensure_parakeet_library(offline=offline)
            )
        if _api.abi_version != _ParakeetCAPI.REQUIRED_ABI:
            raise EkkoError(
                f"parakeet.cpp ABI {_api.abi_version} is unsupported; "
                f"expected {_ParakeetCAPI.REQUIRED_ABI}"
            )
        self._api = _api
        self.model_path = str(model)
        self._context = self._api.load(self.model_path)
        if not self._context:
            raise EkkoError(f"parakeet.cpp could not load model: {model}")
        self._finalizer = weakref.finalize(
            self, self._api.free, self._context
        )
        self.max_seconds = max_seconds
        self._lock = threading.Lock()

    def transcribe(self, audio: str | Path) -> Transcript:
        from .audio import load_audio

        decoded = load_audio(audio, max_seconds=self.max_seconds)
        with self._lock:
            if not self._context:
                raise EkkoError("Tiny backend is closed")
            payload = self._api.transcribe_pcm_json(
                self._context, decoded.samples
            )
        return _tiny_transcript(
            payload,
            model=self.model_path,
            runtime_version=self._api.runtime_version,
            duration=decoded.duration,
        )

    def close(self) -> None:
        with self._lock:
            if self._context:
                self._context = 0
                self._finalizer()

    def __enter__(self) -> "TinyBackend":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class TinyCLIBackend:
    """Compatibility fallback using the parakeet.cpp command-line binary."""

    def __init__(
        self,
        model: str | Path,
        *,
        cli: str | Path = "parakeet-cli",
        threads: int = 0,
        max_seconds: float = 3600.0,
        timeout_seconds: float = 7200.0,
    ) -> None:
        if threads < 0:
            raise EkkoError("Tiny CLI threads cannot be negative")
        if timeout_seconds <= 0:
            raise EkkoError("Tiny CLI timeout must be positive")
        self.model = str(model)
        self.cli = str(cli)
        self.threads = threads
        self.max_seconds = max_seconds
        self.timeout_seconds = timeout_seconds

    def transcribe(self, audio: str | Path) -> Transcript:
        from .audio import load_audio

        decoded = load_audio(audio, max_seconds=self.max_seconds)
        command = [
            self.cli,
            "transcribe",
            "--model",
            self.model,
            "--input",
            str(audio),
            "--timestamps",
            "--json",
        ]
        if self.threads:
            command.extend(["--threads", str(self.threads)])

        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise EkkoError(
                f"{self.cli} exceeded the {self.timeout_seconds:.0f}s timeout"
            ) from error
        except OSError as error:
            raise EkkoError(f"could not start {self.cli}: {error}") from error
        if process.returncode != 0:
            detail = process.stderr.strip() or "no error detail"
            raise EkkoError(
                f"{self.cli} failed ({process.returncode}): {detail[:500]}"
            )

        try:
            payload = json.loads(process.stdout)
        except json.JSONDecodeError as error:
            raise EkkoError(
                f"invalid JSON from {self.cli}: {process.stdout[:500]}"
            ) from error
        return _tiny_transcript(
            payload,
            model=self.model,
            runtime_version=None,
            duration=decoded.duration,
        )

    def __enter__(self) -> "TinyCLIBackend":
        return self

    def __exit__(self, *_: object) -> None:
        pass