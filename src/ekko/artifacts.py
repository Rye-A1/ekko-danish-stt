from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shutil
import stat
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from filelock import FileLock

from .runtime import EkkoError


@dataclass(frozen=True)
class NativeArtifact:
    archive: str
    sha256: str
    directory: str
    library: str
    library_sha256: str


@dataclass(frozen=True)
class HubArtifact:
    repo_id: str
    revision: str
    filename: str
    sha256: str
    dialect: str


RELEASE_MANIFEST = json.loads(
    files("ekko").joinpath("release_manifest.json").read_text(encoding="utf-8")
)
HUB_ARTIFACTS = {
    name: HubArtifact(**metadata)
    for name, metadata in RELEASE_MANIFEST["artifacts"].items()
}
HUB_LIBRARY_NAME = RELEASE_MANIFEST["distribution"]
HUB_LIBRARY_VERSION = RELEASE_MANIFEST["release_version"]
PARAKEET_VERSION = RELEASE_MANIFEST["native_runtime"]["version"]
PARAKEET_RELEASE_URL = RELEASE_MANIFEST["native_runtime"]["release_url"]
_PARAKEET_ARTIFACTS = {
    tuple(key.split("/", 1)): NativeArtifact(**metadata)
    for key, metadata in RELEASE_MANIFEST["native_runtime"]["artifacts"].items()
}


def cache_root() -> Path:
    override = os.environ.get("EKKO_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    current_platform = _platform_name()
    if current_platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ekko"
    if current_platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ekko"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "ekko"


def _platform_name() -> str:
    return sys.platform


def _machine(value: str | None = None) -> str:
    machine = (value or platform.machine()).lower()
    return {"arm64": "arm64", "aarch64": "aarch64", "amd64": "amd64", "x86_64": "x86_64"}.get(
        machine, machine
    )


def native_artifact(
    *,
    system: str | None = None,
    machine: str | None = None,
    libc: tuple[str, str] | None = None,
) -> NativeArtifact:
    current_system = system or _platform_name()
    current_machine = _machine(machine)
    key = (current_system, current_machine)
    if key == ("linux", "arm64"):
        key = ("linux", "aarch64")
    if key == ("win32", "x86_64"):
        key = ("win32", "amd64")
    try:
        artifact = _PARAKEET_ARTIFACTS[key]
    except KeyError as error:
        raise EkkoError(
            f"no prebuilt parakeet.cpp {PARAKEET_VERSION} library for "
            f"{current_system}/{current_machine}"
        ) from error
    if current_system == "linux":
        libc_name, libc_version = libc or platform.libc_ver()
        try:
            version = tuple(int(part) for part in libc_version.split(".")[:2])
        except ValueError:
            version = ()
        if libc_name.lower() != "glibc" or version < (2, 34):
            raise EkkoError(
                "the published parakeet.cpp Linux library requires glibc >= 2.34; "
                f"detected {libc_name or 'unknown'} {libc_version or 'unknown'}"
            )
    return artifact


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_verified_hub_file(
    hub: object,
    *,
    repo_id: str,
    revision: str,
    filename: str,
    expected_sha256: str,
    local_only: bool,
) -> Path:
    lock_root = cache_root() / "locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_root / f"hub-{expected_sha256}.lock"))

    def download(*, force: bool) -> Path:
        try:
            return Path(
                hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    cache_dir=cache_root() / "huggingface",
                    local_files_only=local_only,
                    force_download=force,
                    library_name=HUB_LIBRARY_NAME,
                    library_version=HUB_LIBRARY_VERSION,
                )
            )
        except Exception as error:
            raise EkkoError(
                f"could not download {repo_id}/{filename} at {revision}: {error}"
            ) from error

    def actual_hash(path: Path) -> str:
        try:
            return sha256_file(path)
        except OSError as error:
            raise EkkoError(f"could not hash cached artifact {path}: {error}") from error

    with lock:
        path = download(force=False)
        actual = actual_hash(path)
        if actual == expected_sha256:
            return path
        if not local_only:
            path = download(force=True)
            actual = actual_hash(path)
            if actual == expected_sha256:
                return path
        raise EkkoError(
            f"checksum mismatch for {repo_id}/{filename}: "
            f"expected {expected_sha256}, got {actual}; remove the cached file "
            "and retry"
        )


def _ensure_model_licenses(
    hub: object, *, repo_id: str, revision: str, local_only: bool
) -> None:
    model = next(
        (
            metadata
            for metadata in RELEASE_MANIFEST["models"].values()
            if metadata["repo_id"] == repo_id
        ),
        None,
    )
    if model is None:
        return
    if model["snapshot_revision"] != revision:
        raise EkkoError(f"model and legal files use different revisions for {repo_id}")
    for filename, expected_sha256 in model["license_files"].items():
        _download_verified_hub_file(
            hub,
            repo_id=repo_id,
            revision=revision,
            filename=filename,
            expected_sha256=expected_sha256,
            local_only=local_only,
        )


def ensure_hub_artifact(name: str, *, offline: bool = False) -> Path:
    try:
        artifact = HUB_ARTIFACTS[name]
    except KeyError as error:
        raise EkkoError(f"unknown Ekko artifact: {name}") from error

    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError as error:
        raise EkkoError(
            "model download requires huggingface-hub; install ekko-stt"
        ) from error

    local_only = offline or os.environ.get("EKKO_OFFLINE") == "1"
    path = _download_verified_hub_file(
        hub,
        repo_id=artifact.repo_id,
        revision=artifact.revision,
        filename=artifact.filename,
        expected_sha256=artifact.sha256,
        local_only=local_only,
    )
    _ensure_model_licenses(
        hub,
        repo_id=artifact.repo_id,
        revision=artifact.revision,
        local_only=local_only,
    )
    return path


def ensure_pnc_directory(*, offline: bool = False) -> Path:
    paths = [
        ensure_hub_artifact(name, offline=offline)
        for name in ("pnc-model", "pnc-tokenizer", "pnc-config")
    ]
    parents = {path.parent for path in paths}
    if len(parents) != 1:
        raise EkkoError("PnC files did not resolve to one pinned Hub snapshot")
    return parents.pop()


def ensure_tiny_onnx_directory(
    precision: str, *, offline: bool = False
) -> Path:
    if precision not in {"fp32", "int8"}:
        raise EkkoError(f"unsupported Tiny ONNX precision: {precision}")
    paths = [
        ensure_hub_artifact(
            f"tiny-onnx-{precision}-{component}", offline=offline
        )
        for component in ("encoder", "decoder", "joiner")
    ]
    paths.append(ensure_hub_artifact("tiny-onnx-tokens", offline=offline))
    parents = {path.parent for path in paths}
    if len(parents) != 1:
        raise EkkoError("Tiny ONNX files did not resolve to one pinned Hub snapshot")
    return parents.pop()


def _safe_target(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if target != root.resolve() and root.resolve() not in target.parents:
        raise EkkoError(f"unsafe path in native runtime archive: {name}")
    return target


def _extract_archive(archive: Path, destination: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                _safe_target(destination, member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise EkkoError(
                        f"links are not allowed in native runtime archive: "
                        f"{member.filename}"
                    )
            bundle.extractall(destination)
        return

    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            _safe_target(destination, member.name)
            if member.issym() or member.islnk():
                raise EkkoError(
                    f"links are not allowed in native runtime archive: {member.name}"
                )
        bundle.extractall(destination)


def ensure_parakeet_library(*, offline: bool = False) -> Path:
    override = os.environ.get("EKKO_PARAKEET_LIBRARY")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise EkkoError(f"EKKO_PARAKEET_LIBRARY does not exist: {path}")
        return path

    artifact = native_artifact()
    install_root = (
        cache_root()
        / "native"
        / f"parakeet-{PARAKEET_VERSION}-{artifact.sha256[:12]}"
    )
    library = install_root / artifact.directory / artifact.library
    install_root.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(install_root) + ".lock")
    with lock:
        if library.is_file():
            actual = sha256_file(library)
            if actual != artifact.library_sha256:
                raise EkkoError(
                    f"cached native library checksum mismatch at {library}; "
                    "remove its versioned cache directory and retry"
                )
            return library
        if offline or os.environ.get("EKKO_OFFLINE") == "1":
            raise EkkoError(f"parakeet.cpp runtime is not cached at {library}")

        with tempfile.TemporaryDirectory(
            prefix="parakeet-download-", dir=install_root.parent
        ) as temporary:
            temporary_path = Path(temporary)
            archive = temporary_path / artifact.archive
            request = urllib.request.Request(
                f"{PARAKEET_RELEASE_URL}/{artifact.archive}",
                headers={"User-Agent": "ekko-runtime"},
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    with archive.open("wb") as output:
                        shutil.copyfileobj(response, output)
            except OSError as error:
                raise EkkoError(
                    f"could not download parakeet.cpp runtime: {error}"
                ) from error
            actual = sha256_file(archive)
            if actual != artifact.sha256:
                raise EkkoError(
                    f"parakeet.cpp archive checksum mismatch: expected "
                    f"{artifact.sha256}, got {actual}"
                )
            extracted = temporary_path / "extracted"
            extracted.mkdir()
            _extract_archive(archive, extracted)
            candidate = extracted / artifact.directory / artifact.library
            if not candidate.is_file():
                raise EkkoError(f"native runtime archive lacks {artifact.library}")
            actual_library = sha256_file(candidate)
            if actual_library != artifact.library_sha256:
                raise EkkoError(
                    f"parakeet.cpp library checksum mismatch: expected "
                    f"{artifact.library_sha256}, got {actual_library}"
                )
            try:
                os.replace(extracted, install_root)
            except FileExistsError:
                pass

        if not library.is_file():
            raise EkkoError(
                f"native runtime installation failed: {library}"
            )
        if sha256_file(library) != artifact.library_sha256:
            raise EkkoError(f"installed native library checksum mismatch: {library}")
    return library