from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from .runtime import EkkoError


SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AudioData:
    samples: object
    duration: float


def audio_duration(path: str | Path) -> float:
    try:
        soundfile = importlib.import_module("soundfile")
        info = soundfile.info(str(path))
    except (ImportError, OSError, RuntimeError) as error:
        raise EkkoError(f"could not inspect audio {path}: {error}") from error
    if info.samplerate <= 0:
        raise EkkoError(f"audio has invalid sample rate: {path}")
    return float(info.frames) / float(info.samplerate)


def load_audio(path: str | Path, *, max_seconds: float = 400.0) -> AudioData:
    duration = audio_duration(path)
    if duration > max_seconds:
        raise EkkoError(
            f"audio is {duration:.2f}s; this profile accepts at most "
            f"{max_seconds:.2f}s per call"
        )

    try:
        numpy = importlib.import_module("numpy")
        soundfile = importlib.import_module("soundfile")
        samples, sample_rate = soundfile.read(
            str(path), dtype="float32", always_2d=True
        )
    except (ImportError, OSError, RuntimeError) as error:
        raise EkkoError(f"could not decode audio {path}: {error}") from error

    mono = samples.mean(axis=1, dtype=numpy.float32)
    if sample_rate != SAMPLE_RATE:
        try:
            soxr = importlib.import_module("soxr")
        except ImportError as error:
            raise EkkoError(
                "resampling requires soxr; reinstall ekko-stt"
            ) from error
        mono = soxr.resample(mono, sample_rate, SAMPLE_RATE, quality="HQ")
    mono = numpy.ascontiguousarray(mono, dtype=numpy.float32)
    if not mono.size:
        raise EkkoError(f"audio contains no samples: {path}")
    if not bool(numpy.isfinite(mono).all()):
        raise EkkoError(f"audio contains non-finite samples: {path}")
    if mono.size > int(max_seconds * SAMPLE_RATE) + 1:
        raise EkkoError(f"decoded audio exceeds {max_seconds:.2f}s")
    return AudioData(samples=mono, duration=float(mono.size) / SAMPLE_RATE)


def peak_normalize(
    samples: object, *, target: float = 0.95, silence_floor: float = 1e-6
) -> object:
    if not 0 < target <= 1:
        raise ValueError("peak-normalization target must be in (0, 1]")
    if silence_floor < 0:
        raise ValueError("silence floor must not be negative")
    numpy = importlib.import_module("numpy")
    values = numpy.asarray(samples, dtype=numpy.float32)
    if not values.size:
        return numpy.ascontiguousarray(values)
    if not bool(numpy.isfinite(values).all()):
        raise EkkoError("cannot peak-normalize non-finite samples")
    peak = float(numpy.abs(values).max())
    if peak <= silence_floor:
        return numpy.zeros_like(values, dtype=numpy.float32)
    return numpy.ascontiguousarray(values / peak * target, dtype=numpy.float32)