from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ekko.artifacts import (
    HUB_ARTIFACTS,
    RELEASE_MANIFEST,
    HubArtifact,
    NativeArtifact,
    ensure_hub_artifact,
    ensure_parakeet_library,
    ensure_tiny_onnx_directory,
    native_artifact,
    sha256_file,
)
from ekko.runtime import EkkoError


class NativeArtifactTest(unittest.TestCase):
    def test_rejects_unsupported_linux_libc(self) -> None:
        with self.assertRaisesRegex(EkkoError, "glibc >= 2.34"):
            native_artifact(
                system="linux", machine="x86_64", libc=("musl", "1.2.5")
            )

    def test_rejects_corrupt_cached_library(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        artifact = NativeArtifact(
            archive="runtime.tar.gz",
            sha256="a" * 64,
            directory="runtime",
            library="libparakeet.so",
            library_sha256="b" * 64,
        )
        library = root / "native" / f"parakeet-0.5.0-{'a' * 12}" / "runtime" / "libparakeet.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"corrupt")

        with patch("ekko.artifacts.cache_root", return_value=root), patch(
            "ekko.artifacts.native_artifact", return_value=artifact
        ):
            with self.assertRaisesRegex(EkkoError, "checksum mismatch"):
                ensure_parakeet_library(offline=True)

    def test_sha256_reads_actual_bytes(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "artifact"
        path.write_bytes(b"ekko")

        self.assertEqual(
            sha256_file(path),
            "a562048f5a252d665d2c3198b7d73928384c94be1e117e8065cabd672002664f",
        )

    def test_hub_cache_is_rehashed(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "artifact"
        path.write_bytes(b"ekko")
        hub = unittest.mock.Mock()
        hub.hf_hub_download.return_value = str(path)
        artifact = HubArtifact(
            repo_id="owner/model",
            revision="a" * 40,
            filename="model.gguf",
            sha256=sha256_file(path),
            dialect="test",
        )

        with patch.dict(HUB_ARTIFACTS, {"test": artifact}), patch(
            "ekko.artifacts.importlib.import_module", return_value=hub
        ):
            self.assertEqual(ensure_hub_artifact("test"), path)
            request = hub.hf_hub_download.call_args.kwargs
            self.assertEqual(request["library_name"], "ekko-stt")
            self.assertEqual(request["library_version"], "0.1.1")
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(EkkoError, "checksum mismatch"):
                ensure_hub_artifact("test", offline=True)

    def test_corrupt_hub_cache_is_repaired_online(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        path = root / "artifact"
        path.write_bytes(b"corrupt")
        artifact = HubArtifact(
            repo_id="owner/model",
            revision="a" * 40,
            filename="model.gguf",
            sha256=hashlib.sha256(b"repaired").hexdigest(),
            dialect="test",
        )
        hub = unittest.mock.Mock()

        def download(**kwargs):
            if kwargs["force_download"]:
                path.write_bytes(b"repaired")
            return str(path)

        hub.hf_hub_download.side_effect = download
        with patch.dict(HUB_ARTIFACTS, {"test": artifact}), patch(
            "ekko.artifacts.importlib.import_module", return_value=hub
        ), patch("ekko.artifacts.cache_root", return_value=root):
            self.assertEqual(ensure_hub_artifact("test"), path)

        self.assertEqual(hub.hf_hub_download.call_count, 2)
        self.assertTrue(hub.hf_hub_download.call_args_list[1].kwargs["force_download"])

    def test_model_download_includes_same_snapshot_legal_files(self) -> None:
        model = RELEASE_MANIFEST["models"]["tiny"]
        model_path = Path("snapshot/model.gguf")
        artifact = HubArtifact(
            repo_id=model["repo_id"],
            revision=model["snapshot_revision"],
            filename="model.gguf",
            sha256="a" * 64,
            dialect="test",
        )
        with patch.dict(HUB_ARTIFACTS, {"test": artifact}), patch(
            "ekko.artifacts._download_verified_hub_file", return_value=model_path
        ) as download, patch(
            "ekko.artifacts.importlib.import_module", return_value=object()
        ):
            self.assertEqual(ensure_hub_artifact("test", offline=True), model_path)

        requested = [call.kwargs["filename"] for call in download.call_args_list]
        self.assertEqual(requested, ["model.gguf", *model["license_files"]])
        self.assertTrue(all(call.kwargs["revision"] == model["snapshot_revision"]
                            for call in download.call_args_list))

    @patch("ekko.artifacts.ensure_hub_artifact")
    def test_tiny_onnx_bundle_uses_one_precision_and_snapshot(self, ensure) -> None:
        root = Path("snapshot/onnx-sherpa")
        ensure.side_effect = [
            root / "encoder.int8.onnx",
            root / "decoder.int8.onnx",
            root / "joiner.int8.onnx",
            root / "tokens.txt",
        ]

        resolved = ensure_tiny_onnx_directory("int8", offline=True)

        self.assertEqual(resolved, root)
        self.assertEqual(
            [call.args[0] for call in ensure.call_args_list],
            [
                "tiny-onnx-int8-encoder",
                "tiny-onnx-int8-decoder",
                "tiny-onnx-int8-joiner",
                "tiny-onnx-tokens",
            ],
        )
        self.assertTrue(all(call.kwargs["offline"] for call in ensure.call_args_list))

    def test_tiny_onnx_bundle_rejects_unknown_precision(self) -> None:
        with self.assertRaisesRegex(EkkoError, "unsupported Tiny ONNX precision"):
            ensure_tiny_onnx_directory("fp16")

    @patch("ekko.artifacts.ensure_hub_artifact")
    def test_tiny_onnx_bundle_rejects_mixed_snapshots(self, ensure) -> None:
        ensure.side_effect = [
            Path("one/encoder.onnx"),
            Path("two/decoder.onnx"),
            Path("one/joiner.onnx"),
            Path("one/tokens.txt"),
        ]

        with self.assertRaisesRegex(EkkoError, "one pinned Hub snapshot"):
            ensure_tiny_onnx_directory("fp32")


if __name__ == "__main__":
    unittest.main()
