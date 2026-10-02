from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

from ekko.cli import main
from ekko.runtime import Transcript, Word


class CommandContext:
    def __init__(self, result: Transcript) -> None:
        self.result = result

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def transcribe(self, _audio: str) -> Transcript:
        return self.result


class CLITest(unittest.TestCase):
    def test_info_returns_machine_readable_release_manifest(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(["info"])

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["schema"], "ekko-release-manifest-v1")
        self.assertEqual(payload["release_version"], "0.1.0")
        self.assertEqual(payload["models"]["tiny"]["word_timestamps"], True)

    @patch("ekko.cli.Punctuator")
    @patch("ekko.cli.TinyBackend")
    @patch("ekko.cli.ensure_pnc_directory", return_value="pnc")
    @patch("ekko.cli.ensure_hub_artifact", return_value="tiny.gguf")
    def test_tiny_pnc_json_route(self, _artifact, _pnc_dir, backend, punctuator) -> None:
        raw = Transcript(
            text="hej",
            backend="parakeet.cpp",
            words=(Word("hej", 0.0, 0.4),),
        )
        final = Transcript(
            text="Hej.",
            raw_text="hej",
            backend="parakeet.cpp",
            words=(Word("Hej.", 0.0, 0.4),),
        )
        backend.return_value = CommandContext(raw)
        punctuator.return_value.apply.return_value = final
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(["tiny", "audio.wav", "--pnc", "--format", "json"])

        self.assertEqual(exit_code, 0)
        self.assertIn('"text": "Hej."', output.getvalue())
        self.assertIn('"start": 0.0', output.getvalue())

    @patch("ekko.cli.Punctuator")
    @patch("ekko.cli.TinyOnnxBackend")
    @patch("ekko.cli.ensure_pnc_directory", return_value="pnc")
    @patch("ekko.cli.ensure_tiny_onnx_directory", return_value="onnx")
    def test_tiny_onnx_pnc_route(
        self, _onnx_dir, _pnc_dir, backend, punctuator
    ) -> None:
        raw = Transcript(
            text="hej",
            backend="sherpa-onnx",
            words=(Word("hej", 0.0, 0.4),),
        )
        final = Transcript(
            text="Hej.",
            raw_text="hej",
            backend="sherpa-onnx",
            words=(Word("Hej.", 0.0, 0.4),),
        )
        backend.return_value = CommandContext(raw)
        punctuator.return_value.apply.return_value = final
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(
                [
                    "tiny-onnx",
                    "audio.wav",
                    "--precision",
                    "fp32",
                    "--threads",
                    "8",
                    "--pnc",
                    "--format",
                    "json",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertIn('"text": "Hej."', output.getvalue())
        backend.assert_called_once_with("onnx", precision="fp32", threads=8)

    def test_all_download_profiles_resolve_exact_artifacts(self) -> None:
        pnc = ["pnc-model", "pnc-tokenizer", "pnc-config"]
        profiles = {
            "tiny": (True, [], ["tiny"]),
            "tiny-pnc": (True, [], ["tiny", *pnc]),
            "tiny-onnx-fp32": (False, ["fp32"], []),
            "tiny-onnx-int8": (False, ["int8"], []),
            "tiny-onnx-int8-pnc": (False, ["int8"], pnc),
            "pnc": (False, [], pnc),
            "all": (
                True,
                ["fp32", "int8"],
                ["tiny", *pnc],
            ),
        }
        for profile, (uses_native, precisions, artifacts) in profiles.items():
            with self.subTest(profile=profile), patch(
                "ekko.cli.ensure_parakeet_library", return_value="native"
            ) as native, patch(
                "ekko.cli.ensure_tiny_onnx_directory", return_value="onnx"
            ) as onnx_dir, patch(
                "ekko.cli.ensure_hub_artifact", return_value="artifact"
            ) as artifact:
                with redirect_stdout(io.StringIO()):
                    exit_code = main(["download", profile, "--offline"])

                self.assertEqual(exit_code, 0)
                self.assertEqual(native.call_count, int(uses_native))
                self.assertEqual(
                    [call.args[0] for call in onnx_dir.call_args_list], precisions
                )
                self.assertEqual(
                    [call.args[0] for call in artifact.call_args_list], artifacts
                )
                self.assertTrue(
                    all(call.kwargs["offline"] for call in onnx_dir.call_args_list)
                )
                self.assertTrue(
                    all(call.kwargs["offline"] for call in artifact.call_args_list)
                )

    @patch("ekko.cli.TinyCLIBackend")
    @patch("ekko.cli.ensure_hub_artifact", return_value="tiny.gguf")
    def test_tiny_cli_fallback_route(self, _artifact, backend) -> None:
        backend.return_value = CommandContext(
            Transcript(text="hej", backend="parakeet.cpp")
        )
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(
                ["tiny", "audio.wav", "--cli", "/opt/parakeet-cli"]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue(), "hej\n")
        backend.assert_called_once_with(
            "tiny.gguf",
            cli="/opt/parakeet-cli",
            threads=0,
            timeout_seconds=7200.0,
        )

    def test_tiny_rejects_conflicting_native_and_cli_providers(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(
                [
                    "tiny",
                    "audio.wav",
                    "--library",
                    "libparakeet.so",
                    "--cli",
                    "parakeet-cli",
                ]
            )


if __name__ == "__main__":
    unittest.main()