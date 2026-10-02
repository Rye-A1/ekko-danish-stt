---
title: Ekko Tiny + PnC — Danish speech to text in your browser
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
short_description: On-device Danish speech recognition and punctuation
---

# Ekko v1 Tiny browser demo

This Space's code is MIT licensed. The Tiny and PnC model weights have separate
terms in their Hugging Face model repositories.

The [Ekko STT source code](https://github.com/Rye-A1/ekko-danish-stt) is on GitHub.

This static demo transcribes Danish speech locally in the browser with Ekko v1
Tiny int8, sherpa-onnx WebAssembly, and Silero VAD. Ekko PnC loads separately
and adds punctuation and capitalization without changing word timestamps.
Audio and transcripts are never sent to an inference server. The most recent
20 transcript sessions are saved in the browser's local storage; audio is not
saved.

The model URLs and checksums are pinned in [`model-config.json`](model-config.json).
The browser downloads Tiny's runtime configuration and ONNX files, then PnC's
files, directly from their Hugging Face model repos. Verified model files are
kept in browser storage for later visits when storage is available; every load
still verifies their checksums and initializes the WebAssembly runtime. The
small third-party Silero VAD file is bundled with the Space for browser access.
Microphone access requires a secure origin and the browser's permission prompt,
shown after the app's introductory mic dialog. Browser support remains a preview while
physical-device and mobile checks are completed.

The browser and Python CLI pin the same PnC v2 int8 artifact. The Python
downloader also verifies the accompanying legal files from that revision.

## Local preview

From the repository root, use the checksum-verified local model cache:

```bash
uv sync --locked
uv run --no-sync python demo/ekko-tiny-browser/serve_local.py
```

The PnC v2 files must already be in the Hugging Face cache, or pass a
local directory explicitly:

```bash
uv run --no-sync python demo/ekko-tiny-browser/serve_local.py \
  --pnc-dir /path/to/ekko-pnc-v2
```

The preview checks all PnC file sizes and hashes before serving them.

## Build and deploy

```bash
demo/ekko-tiny-browser/build.sh
```

The build pins sherpa-onnx v1.13.6, Emscripten 4.0.23, and ONNX Runtime Web
1.29.0. It writes a static site to `demo/ekko-tiny-browser/dist/`; model
weights are downloaded by the browser and are not committed. Set
`EKKO_WASM_BUILDER=docker` to use the pinned container image instead of a local
Emscripten installation.

The GitHub workflow builds the Space for demo pull requests and pushes to
`main`. On pushes to `main`, it also deploys to `RyeAI/ekko-tiny-browser` when
the repository variable `HF_SPACE_DEPLOY_ENABLED` is `true`. A manual workflow
run can deploy as well. Deployment requires the `HF_TOKEN` secret and
anonymously accessible model URLs. The upload mirrors `dist/`, including
deletions of old generated files.

The JavaScript behavior checks can be run with:

```bash
node tests/browser_live_chunking.mjs
node tests/browser_pnc_sentence_casing.mjs
node tests/browser_space_signed_assets.mjs
node tests/browser_tiny_runtime_config.mjs
node tests/browser_model_cache.mjs
```
