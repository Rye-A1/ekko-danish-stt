#!/usr/bin/env python3
"""Check or upload the generated Ekko v1 Tiny static Space."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import HfApi, create_repo, hf_hub_download


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
DEFAULT_REPOSITORY = "RyeAI/ekko-tiny-browser"
REQUIRED_FILES = {
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "SHERPA_ONNX_LICENSE",
    "ONNXRUNTIME_LICENSE",
    "index.html",
    "styles.css",
    "app.js",
    "worker.js",
    "pnc-worker.js",
    "capture-worklet.js",
    "model-config.json",
    "model-files/tiny/silero_vad.onnx",
    "ort.wasm.min.js",
    "ort.wasm.min.js.map",
    "ort-wasm-simd-threaded.mjs",
    "ort-wasm-simd-threaded.wasm",
    "sherpa-onnx-asr.js",
    "sherpa-onnx-vad.js",
    "sherpa-onnx-wasm-main-vad-asr.js",
    "sherpa-onnx-wasm-main-vad-asr.wasm",
    "sherpa-onnx-wasm-main-vad-asr.data",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_files() -> dict[str, Path]:
    if not DIST.is_dir():
        raise FileNotFoundError("Space dist/ is missing; run build.sh first")
    files = {
        path.relative_to(DIST).as_posix(): path
        for path in DIST.rglob("*")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(DIST).parts)
    }
    missing = REQUIRED_FILES - set(files)
    if missing:
        raise FileNotFoundError("Space dist/ is incomplete: " + ", ".join(sorted(missing)))
    return files


def check(api: HfApi, repository: str) -> list[str]:
    files = local_files()
    remote_files = set(api.list_repo_files(repository, repo_type="space"))
    issues: list[str] = []
    for relative in sorted(remote_files - set(files) - {".gitattributes"}):
        issues.append(f"unexpected {relative}")
    for relative, local_path in sorted(files.items()):
        if relative not in remote_files:
            issues.append(f"missing {relative}")
            continue
        remote_path = Path(
            hf_hub_download(
                repository,
                relative,
                repo_type="space",
                force_download=True,
            )
        )
        if digest(local_path) != digest(remote_path):
            issues.append(f"differs {relative}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--push", action="store_true", help="upload dist/ before checking")
    parser.add_argument(
        "--create-private",
        action="store_true",
        help="create a missing private static Space before upload",
    )
    args = parser.parse_args()

    api = HfApi()
    if args.create_private:
        create_repo(
            args.repository,
            repo_type="space",
            space_sdk="static",
            private=True,
            exist_ok=True,
        )
    if args.push:
        local_files()
        api.upload_folder(
            repo_id=args.repository,
            repo_type="space",
            folder_path=str(DIST),
            path_in_repo="",
            ignore_patterns=[".*", "**/.*"],
            delete_patterns="*",
            commit_message="Deploy Ekko v1 Tiny + PnC browser demo",
        )
        print(f"uploaded {args.repository}")

    issues = check(api, args.repository)
    if issues:
        for issue in issues:
            print(f"{args.repository}: {issue}")
        return 1
    print(f"verified {args.repository}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
