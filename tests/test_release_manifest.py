from __future__ import annotations

import json
import tomllib
import unittest
from pathlib import Path

from ekko import RELEASE_MANIFEST
from ekko.artifacts import HUB_ARTIFACTS


ROOT = Path(__file__).resolve().parents[1]


class ReleaseManifestTest(unittest.TestCase):
    def test_distribution_and_model_pins(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertEqual(RELEASE_MANIFEST["distribution"], project["name"])
        self.assertEqual(RELEASE_MANIFEST["release_version"], project["version"])
        self.assertNotIn("benchmark", project["optional-dependencies"])
        self.assertEqual(set(RELEASE_MANIFEST["artifacts"]), set(HUB_ARTIFACTS))
        self.assertEqual(RELEASE_MANIFEST["native_runtime"]["version"], "0.5.0")
        self.assertNotIn("data_sources", RELEASE_MANIFEST)
        self.assertNotIn("training_recipes", RELEASE_MANIFEST)
        self.assertNotIn("evidence", RELEASE_MANIFEST)

        for name, model in RELEASE_MANIFEST["models"].items():
            self.assertRegex(model["snapshot_revision"], r"^[0-9a-f]{40}$")
            self.assertTrue(model["repo_id"].startswith("RyeAI/"))
            self.assertTrue(model["license_files"])
            for filename, digest in model["license_files"].items():
                self.assertFalse(filename.startswith("/"))
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertEqual(
                {artifact.revision for artifact in HUB_ARTIFACTS.values()
                 if artifact.repo_id == model["repo_id"]},
                {model["snapshot_revision"]},
                name,
            )
        self.assertEqual(
            HUB_ARTIFACTS["pnc-model"].sha256,
            "46bb979d409731872f10b6cb71298594595f16368d2aec2fcaf94d87b4266527",
        )

    def test_browser_artifacts_match_runtime_pins(self) -> None:
        config = json.loads(
            (ROOT / "demo" / "ekko-tiny-browser" / "model-config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["revision"], RELEASE_MANIFEST["models"]["tiny"]["snapshot_revision"])
        self.assertEqual(config["pnc"]["revision"], RELEASE_MANIFEST["models"]["pnc"]["snapshot_revision"])
        expected = {
            "nemo-transducer-encoder.onnx": "tiny-onnx-int8-encoder",
            "nemo-transducer-decoder.onnx": "tiny-onnx-int8-decoder",
            "nemo-transducer-joiner.onnx": "tiny-onnx-int8-joiner",
            "tokens.txt": "tiny-onnx-tokens",
        }
        files = {item["path"]: item for item in config["files"]}
        for filename, artifact_name in expected.items():
            self.assertEqual(files[filename]["sha256"], HUB_ARTIFACTS[artifact_name].sha256)
