from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import shutil
import subprocess
import unittest
from contextlib import redirect_stdout
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "demo" / "ekko-tiny-browser"


class _Parser(HTMLParser):
    pass


class BrowserSpaceTest(unittest.TestCase):
    def test_space_sync_mirrors_deletions(self) -> None:
        spec = importlib.util.spec_from_file_location("ekko_space_sync", SPACE / "sync_space.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        api = Mock()
        api.list_repo_files.return_value = [".gitattributes", "stale.js"]
        with patch.object(module, "local_files", return_value={}):
            self.assertEqual(module.check(api, "RyeAI/ekko-tiny-browser"), ["unexpected stale.js"])

        with patch.object(module, "HfApi", return_value=api), patch.object(
            module, "local_files", return_value={}
        ), patch.object(module, "check", return_value=[]), patch(
            "sys.argv", ["sync_space.py", "--push"]
        ), redirect_stdout(io.StringIO()):
            self.assertEqual(module.main(), 0)
        self.assertEqual(api.upload_folder.call_args.kwargs["delete_patterns"], "*")

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for browser worker tests")
    def test_pnc_sentence_casing_behavior(self) -> None:
        subprocess.run(
            ["node", str(ROOT / "tests" / "browser_pnc_sentence_casing.mjs")],
            check=True,
            capture_output=True,
            text=True,
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for browser worker tests")
    def test_pnc_indexed_windows(self) -> None:
        subprocess.run(
            ["node", str(ROOT / "tests" / "browser_pnc_windows.mjs")],
            check=True,
            capture_output=True,
            text=True,
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for browser worker tests")
    def test_transcript_integrity(self) -> None:
        subprocess.run(
            ["node", str(ROOT / "tests" / "browser_transcript_integrity.mjs")],
            check=True,
            capture_output=True,
            text=True,
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for browser worker tests")
    def test_model_cache(self) -> None:
        subprocess.run(
            ["node", str(ROOT / "tests" / "browser_model_cache.mjs")],
            check=True,
            capture_output=True,
            text=True,
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for browser worker tests")
    def test_live_chunking_and_capture_drain(self) -> None:
        subprocess.run(
            ["node", str(ROOT / "tests" / "browser_live_chunking.mjs")],
            check=True,
            capture_output=True,
            text=True,
        )

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for browser worker tests")
    def test_tiny_hub_runtime_config(self) -> None:
        subprocess.run(
            ["node", str(ROOT / "tests" / "browser_tiny_runtime_config.mjs")],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_model_manifest_matches_release_artifacts(self) -> None:
        config = json.loads((SPACE / "model-config.json").read_text(encoding="utf-8"))
        release = json.loads(
            (ROOT / "src" / "ekko" / "release_manifest.json").read_text(encoding="utf-8")
        )
        expected = {
            "nemo-transducer-encoder.onnx": "tiny-onnx-int8-encoder",
            "nemo-transducer-decoder.onnx": "tiny-onnx-int8-decoder",
            "nemo-transducer-joiner.onnx": "tiny-onnx-int8-joiner",
            "tokens.txt": "tiny-onnx-tokens",
        }
        self.assertEqual(config["revision"], release["models"]["tiny"]["snapshot_revision"])
        files = {item["path"]: item for item in config["files"]}
        self.assertTrue(files["silero_vad.onnx"]["url"].startswith("model-files/tiny/"))
        runtime_file = files["runtime-config.json"]
        runtime_path = SPACE / "runtime-config.json"
        self.assertEqual(runtime_file["sha256"], hashlib.sha256(runtime_path.read_bytes()).hexdigest())
        self.assertEqual(runtime_file["bytes"], runtime_path.stat().st_size)
        self.assertIn(config["revision"], runtime_file["url"])
        self.assertIn("/onnx-sherpa/runtime-config.json", runtime_file["url"])
        self.assertEqual(json.loads(runtime_path.read_text())["sampleRate"], 16000)
        for browser_name, artifact_name in expected.items():
            artifact = release["artifacts"][artifact_name]
            self.assertEqual(files[browser_name]["sha256"], artifact["sha256"])
            self.assertIn(artifact["revision"], files[browser_name]["url"])
            self.assertTrue(files[browser_name]["url"].startswith("https://"))

        pnc = config["pnc"]
        self.assertEqual(pnc["revision"], release["models"]["pnc"]["snapshot_revision"])
        pnc_files = {item["path"]: item for item in pnc["files"]}
        self.assertEqual(
            pnc_files["pnc.int8.onnx"]["sha256"],
            "46bb979d409731872f10b6cb71298594595f16368d2aec2fcaf94d87b4266527",
        )
        self.assertEqual(pnc_files["pnc.int8.onnx"]["bytes"], 22424741)
        for browser_name, artifact_name in {
            "pnc.int8.onnx": "pnc-model",
            "tokenizer.json": "pnc-tokenizer",
            "config.json": "pnc-config",
        }.items():
            artifact = release["artifacts"][artifact_name]
            self.assertEqual(pnc_files[browser_name]["sha256"], artifact["sha256"])
        for item in pnc_files.values():
            self.assertIn(pnc["revision"], item["url"])
            self.assertTrue(item["url"].startswith("https://"))

    def test_space_is_live_microphone_only(self) -> None:
        html = (SPACE / "src" / "index.html").read_text(encoding="utf-8")
        parser = _Parser()
        parser.feed(html)
        parser.close()
        for identifier in (
            "app",
            "livePanel",
            "recordButton",
            "transcriptPanel",
            "segments",
            "pncStatus",
            "themeButton",
            "historyList",
            "clearHistoryButton",
        ):
            self.assertIn(f'id="{identifier}"', html)
        app = (SPACE / "src" / "app.js").read_text(encoding="utf-8")
        worker = (SPACE / "src" / "worker.js").read_text(encoding="utf-8")
        pnc_worker = (SPACE / "src" / "pnc-worker.js").read_text(encoding="utf-8")
        self.assertNotIn('type="file"', html)
        self.assertNotIn("filePanel", html)
        self.assertNotIn("UPLOAD_MAX_SECONDS", app)
        self.assertNotIn('"file-result"', worker)
        self.assertNotIn('"file-progress"', worker)
        self.assertNotIn('"file-stopped"', worker)
        self.assertNotIn('id="timingButton"', html)
        self.assertNotIn('label.textContent = "Dig"', app)
        self.assertIn("class StreamingLinearResampler", app)
        self.assertNotIn("function resampleLinear", app)
        self.assertIn('state.audioContext.state === "suspended"', app)
        self.assertLess(
            app.index("state.recording = true"),
            app.index("state.sourceNode.connect(state.captureNode)"),
        )
        self.assertIn("LIVE_PRE_ROLL_SECONDS = 0.32", worker)
        self.assertIn("SHORT_UTTERANCE_MIN_SECONDS = 0.12", worker)
        self.assertIn("minSpeechDuration: SHORT_UTTERANCE_MIN_SECONDS", worker)
        self.assertIn("segmentStart - LIVE_PRE_ROLL_SAMPLES", worker)
        self.assertIn("addSegmentPreRoll(segment)", worker)
        self.assertIn("function stabilizeRenderedWords(words)", pnc_worker)
        self.assertIn("output[index + 1] = capitalizeInitial", pnc_worker)
        self.assertNotIn('output[index].replace(/[.!?]+$/u, "")', pnc_worker)
        self.assertIn('HISTORY_STORAGE_KEY = "ekko-tiny-browser:conversations:v1"', app)
        self.assertIn("HISTORY_LIMIT = 20", app)
        self.assertIn("localStorage.setItem(HISTORY_STORAGE_KEY", app)
        self.assertIn("persistCurrentConversation();", app)
        self.assertIn("if (state.pncFailed || state.pncPending", app)
        self.assertIn("Sig det til Ekko", html)
        self.assertIn('<link rel="icon" href="data:," />', html)
        self.assertIn("Ingen lyd, tekst eller brugsdata forlader din browser.", html)
        self.assertNotIn("Ingen lyd, tekst eller brugsdata sendes til Rye AI.", html)
        self.assertNotIn('id="pncToggle"', html)
        self.assertNotIn('id="historyButton"', html)
        self.assertGreater(html.index('id="historyList"'), html.index('id="settingsDialog"'))
        self.assertIn('href="styles.css?v=20260926-9"', html)
        self.assertIn('src="app.js?v=20261001-3"', html)
        self.assertIn('new Worker(signedAsset("worker.js?v=20260927-2"))', app)
        self.assertIn('new Worker(signedAsset("pnc-worker.js?v=20261001-2"))', app)
        self.assertIn('capture-worklet.js?v=20260926-4', app)
        styles = (SPACE / "src" / "styles.css").read_text(encoding="utf-8")
        self.assertIn('class="ring ring-1"', html)
        self.assertIn('class="orb" id="orb"', html)
        self.assertIn('class="model-link" href="https://huggingface.co/RyeAI/ekko-v1-tiny"', html)
        self.assertIn('class="model-link" href="https://huggingface.co/RyeAI/ekko-pnc"', html)
        self.assertEqual(html.count('class="lucide lucide-chevron-right"'), 2)
        self.assertLess(html.index('id="transcriptPanel"'), html.index('class="transcript-heading"'))
        self.assertIn("#version 300 es", app)
        self.assertIn('hexRgb("#27AE60")', app)
        self.assertIn('hexRgb("#8CBFFF")', app)
        self.assertIn('hexRgb("#FFAAAD")', app)
        self.assertIn('classList.toggle("conversation-visible"', app)
        self.assertIn("reveal-word", app)
        self.assertIn(".scene.conversation-visible", styles)
        self.assertIn("cubic-bezier(0.32, 0.72, 0, 1)", styles)
        self.assertIn("height: 2px;", styles[styles.index(".circle-button.sliders span,"):])

    def test_build_is_pinned_and_source_contains_no_runtime_binaries(self) -> None:
        build = (SPACE / "build.sh").read_text(encoding="utf-8")
        sync = (SPACE / "sync_space.py").read_text(encoding="utf-8")
        serve = (SPACE / "serve_local.py").read_text(encoding="utf-8")
        self.assertIn('SHERPA_TAG="v1.13.6"', build)
        self.assertIn('SHERPA_COMMIT="1cb484af5e69d3c7803c1eb0b3b5ab8041e0e911"', build)
        self.assertIn('EMSDK_COMMIT="c0bb220cb6e6f4e0fabb6f6db9efd53390ef5e56"', build)
        self.assertIn('EMSDK_IMAGE="emscripten/emsdk:4.0.23"', build)
        self.assertIn('ORT_WEB_VERSION="1.29.0"', build)
        self.assertIn('ORT_WEB_SHA256="7a934b7811c3b050ecfb7619722e2b4de771ce6da20520e17a2018a440316ef3"', build)
        self.assertIn('ORT_LICENSE_SHA256="2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c"', build)
        self.assertIn("pnc-worker.js", (SPACE / "src" / "app.js").read_text(encoding="utf-8"))
        for runtime_file in (
            "pnc-worker.js",
            "ort.wasm.min.js",
            "ort.wasm.min.js.map",
            "ort-wasm-simd-threaded.mjs",
            "ort-wasm-simd-threaded.wasm",
            "ONNXRUNTIME_LICENSE",
        ):
            self.assertIn(f'"{runtime_file}"', sync)
            self.assertIn(f'"{runtime_file}"', serve)
        self.assertIn('parser.add_argument("--tls-cert", type=Path)', serve)
        self.assertIn('parser.add_argument("--tls-key", type=Path)', serve)
        self.assertIn("ssl.PROTOCOL_TLS_SERVER", serve)
        self.assertIn("?sha256={item['sha256']}", serve)
        self.assertIn('self.send_header("Cache-Control", "no-store")', serve)
        for path in SPACE.rglob("*"):
            if ".build" in path.parts or "dist" in path.parts or not path.is_file():
                continue
            self.assertNotIn(path.suffix, {".onnx", ".wasm", ".data"})

    def test_space_deployment_is_gated_and_pinned(self) -> None:
        workflow = (
            ROOT / ".github" / "workflows" / "deploy-browser-space.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("HF_SPACE_DEPLOY_ENABLED", workflow)
        self.assertIn("pull_request:", workflow)
        self.assertIn("github.ref == 'refs/heads/main'", workflow)
        self.assertIn("github.event_name == 'push'", workflow)
        self.assertIn("workflow_dispatch", workflow)
        self.assertIn("Verify browser model URLs", workflow)
        self.assertIn("curl --head --location", workflow)
        self.assertIn('test -f "$local_path"', workflow)
        self.assertIn("--push", workflow)
        self.assertIn("actions/checkout@11d5960a326750d5838078e36cf38b85af677262", workflow)
        self.assertIn("actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02", workflow)
        self.assertIn("actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093", workflow)


if __name__ == "__main__":
    unittest.main()
