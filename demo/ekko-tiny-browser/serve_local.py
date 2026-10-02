#!/usr/bin/env python3
"""Serve the built Space using checksum-verified local model artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import ssl
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import snapshot_download

from ekko.artifacts import ensure_tiny_onnx_directory


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
CONFIG = ROOT / "model-config.json"
VAD_NAME = "silero_vad.onnx"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_routes(pnc_dir: Path | None = None) -> tuple[dict, dict[str, Path]]:
    if not DIST.is_dir():
        raise FileNotFoundError("Space dist/ is missing; run build.sh first")
    missing = sorted(name for name in REQUIRED_FILES if not (DIST / name).is_file())
    if missing:
        raise FileNotFoundError("Space dist/ is incomplete: " + ", ".join(missing))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    model_dir = ensure_tiny_onnx_directory("int8", offline=True)
    if pnc_dir is None:
        pnc_dir = Path(snapshot_download(
            repo_id=config["pnc"]["repository"],
            revision=config["pnc"]["revision"],
            allow_patterns=[item["path"] for item in config["pnc"]["files"]],
            local_files_only=True,
        ))
    routes = {
        "nemo-transducer-encoder.onnx": model_dir / "encoder.int8.onnx",
        "nemo-transducer-decoder.onnx": model_dir / "decoder.int8.onnx",
        "nemo-transducer-joiner.onnx": model_dir / "joiner.int8.onnx",
        "tokens.txt": model_dir / "tokens.txt",
        "runtime-config.json": ROOT / "runtime-config.json",
        VAD_NAME: DIST / "model-files" / "tiny" / VAD_NAME,
    }
    vad = next(item for item in config["files"] if item["path"] == VAD_NAME)
    if sha256(routes[VAD_NAME]) != vad["sha256"]:
        raise ValueError("Bundled Silero VAD checksum mismatch")
    for item in config["pnc"]["files"]:
        source = pnc_dir / item["path"]
        if not source.is_file() or source.stat().st_size != item["bytes"] or sha256(source) != item["sha256"]:
            raise ValueError(f"PnC artifact missing or checksum mismatch: {source}")
        routes[f"pnc/{item['path']}"] = source
        item["url"] = f"/__ekko_model__/pnc/{item['path']}?sha256={item['sha256']}"
    for item in config["files"]:
        item["url"] = f"/__ekko_model__/{item['path']}?sha256={item['sha256']}"
    return config, routes


def handler_factory(config: dict, routes: dict[str, Path]):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(DIST), **kwargs)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/model-config.json":
                payload = json.dumps(config).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            prefix = "/__ekko_model__/"
            if path.startswith(prefix):
                name = path.removeprefix(prefix)
                source = routes.get(name)
                if source is None or not source.is_file():
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Cache-Control", "public, max-age=31536000, immutable")
                self.send_header("Content-Length", str(source.stat().st_size))
                self.end_headers()
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, self.wfile)
                return
            super().do_GET()

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--tls-cert", type=Path)
    parser.add_argument("--tls-key", type=Path)
    parser.add_argument("--pnc-dir", type=Path, help="local checksum-verified v2 PnC release directory")
    args = parser.parse_args()
    if (args.tls_cert is None) != (args.tls_key is None):
        parser.error("--tls-cert and --tls-key must be provided together")
    config, routes = build_routes(args.pnc_dir.expanduser() if args.pnc_dir else None)
    server = ThreadingHTTPServer((args.host, args.port), handler_factory(config, routes))
    scheme = "http"
    if args.tls_cert is not None and args.tls_key is not None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.tls_cert.expanduser(), args.tls_key.expanduser())
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    print(f"Ekko browser demo: {scheme}://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
