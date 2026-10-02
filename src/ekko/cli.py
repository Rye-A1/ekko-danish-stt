from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .artifacts import (
    RELEASE_MANIFEST,
    ensure_hub_artifact,
    ensure_parakeet_library,
    ensure_pnc_directory,
    ensure_tiny_onnx_directory,
)
from .pnc import Punctuator
from .runtime import EkkoError, TinyBackend, TinyCLIBackend, Transcript
from .tiny_onnx import TinyOnnxBackend


def _model_path(profile: str, supplied: str | None, offline: bool) -> Path:
    return Path(supplied).expanduser() if supplied else ensure_hub_artifact(
        profile, offline=offline
    )


def _print_result(result: Transcript, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(result.to_dict(), ensure_ascii=False))
    else:
        print(result.text)


def _run_tiny(args: argparse.Namespace) -> None:
    model = _model_path("tiny", args.model, args.offline)
    if args.cli:
        backend = TinyCLIBackend(
            model,
            cli=args.cli,
            threads=args.threads,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        backend = TinyBackend(
            model,
            library=args.library,
            offline=args.offline,
        )
    with backend:
        result = backend.transcribe(args.audio)
    if args.pnc or args.pnc_dir:
        directory = (
            Path(args.pnc_dir).expanduser()
            if args.pnc_dir
            else ensure_pnc_directory(offline=args.offline)
        )
        result = Punctuator(directory).apply(result)
    _print_result(result, args.output_format)


def _run_tiny_onnx(args: argparse.Namespace) -> None:
    directory = (
        Path(args.model_dir).expanduser()
        if args.model_dir
        else ensure_tiny_onnx_directory(args.precision, offline=args.offline)
    )
    with TinyOnnxBackend(
        directory,
        precision=args.precision,
        threads=args.threads,
    ) as runtime:
        result = runtime.transcribe(args.audio)
    if args.pnc or args.pnc_dir:
        pnc_directory = (
            Path(args.pnc_dir).expanduser()
            if args.pnc_dir
            else ensure_pnc_directory(offline=args.offline)
        )
        result = Punctuator(pnc_directory).apply(result)
    _print_result(result, args.output_format)


def _download(args: argparse.Namespace) -> None:
    names = {
        "tiny": ("tiny",),
        "tiny-pnc": ("tiny", "pnc-model", "pnc-tokenizer", "pnc-config"),
        "tiny-onnx-fp32": (),
        "tiny-onnx-int8": (),
        "tiny-onnx-int8-pnc": ("pnc-model", "pnc-tokenizer", "pnc-config"),
        "pnc": ("pnc-model", "pnc-tokenizer", "pnc-config"),
        "all": (
            "tiny",
            "pnc-model",
            "pnc-tokenizer",
            "pnc-config",
        ),
    }[args.profile]
    if args.profile in ("tiny", "tiny-pnc", "all"):
        print(ensure_parakeet_library(offline=args.offline))
    if args.profile in ("tiny-onnx-fp32", "all"):
        print(ensure_tiny_onnx_directory("fp32", offline=args.offline))
    if args.profile in ("tiny-onnx-int8", "tiny-onnx-int8-pnc", "all"):
        print(ensure_tiny_onnx_directory("int8", offline=args.offline))
    for name in names:
        print(ensure_hub_artifact(name, offline=args.offline))


def _info(_args: argparse.Namespace) -> None:
    print(json.dumps(RELEASE_MANIFEST, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ekko", description="Local Danish speech recognition"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tiny = subparsers.add_parser(
        "tiny", help="Tiny Q5_0 with native word timestamps"
    )
    tiny.add_argument("audio")
    tiny.add_argument("--model")
    tiny.add_argument("--pnc", action="store_true", help="apply the int8 PnC model")
    tiny.add_argument("--pnc-dir")
    provider = tiny.add_mutually_exclusive_group()
    provider.add_argument("--library", help="path to a parakeet.cpp shared library")
    provider.add_argument("--cli", help="use this parakeet-cli instead of the C API")
    tiny.add_argument("--threads", type=int, default=0, help="CLI fallback only")
    tiny.add_argument(
        "--timeout-seconds", type=float, default=7200.0, help="CLI fallback only"
    )
    tiny.add_argument("--offline", action="store_true")
    tiny.add_argument("--format", dest="output_format", choices=("text", "json"), default="text")
    tiny.set_defaults(handler=_run_tiny)

    tiny_onnx = subparsers.add_parser(
        "tiny-onnx", help="Tiny sherpa-onnx with native word timestamps"
    )
    tiny_onnx.add_argument("audio")
    tiny_onnx.add_argument("--model-dir")
    tiny_onnx.add_argument("--precision", choices=("fp32", "int8"), default="int8")
    tiny_onnx.add_argument("--threads", type=int, default=4)
    tiny_onnx.add_argument("--pnc", action="store_true", help="apply the int8 PnC model")
    tiny_onnx.add_argument("--pnc-dir")
    tiny_onnx.add_argument("--offline", action="store_true")
    tiny_onnx.add_argument(
        "--format", dest="output_format", choices=("text", "json"), default="text"
    )
    tiny_onnx.set_defaults(handler=_run_tiny_onnx)

    download = subparsers.add_parser(
        "download", help="download and verify a pinned release profile"
    )
    download.add_argument(
        "profile",
        choices=(
            "tiny",
            "tiny-pnc",
            "tiny-onnx-fp32",
            "tiny-onnx-int8",
            "tiny-onnx-int8-pnc",
            "pnc",
            "all",
        ),
    )
    download.add_argument("--offline", action="store_true")
    download.set_defaults(handler=_download)

    info = subparsers.add_parser(
        "info", help="print the machine-readable release and capability manifest"
    )
    info.set_defaults(handler=_info)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except EkkoError as error:
        print(f"ekko: {error}", file=sys.stderr)
        return 1
    return 0
