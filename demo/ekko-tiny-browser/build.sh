#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHERPA_REPOSITORY="https://github.com/k2-fsa/sherpa-onnx.git"
SHERPA_TAG="v1.13.6"
SHERPA_COMMIT="1cb484af5e69d3c7803c1eb0b3b5ab8041e0e911"
EMSDK_TAG="4.0.23"
EMSDK_COMMIT="c0bb220cb6e6f4e0fabb6f6db9efd53390ef5e56"
EMSDK_IMAGE="emscripten/emsdk:4.0.23"
ORT_WEB_VERSION="1.29.0"
ORT_WEB_SHA256="7a934b7811c3b050ecfb7619722e2b4de771ce6da20520e17a2018a440316ef3"
ORT_LICENSE_SHA256="2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c"
VAD_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
VAD_SHA256="9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"
BUILD_ROOT="${EKKO_SPACE_BUILD_ROOT:-$SCRIPT_DIR/.build}"
EMSDK_ROOT="${EKKO_EMSDK_ROOT:-$HOME/.cache/ekko-browser/emsdk-$EMSDK_TAG}"
SHERPA_DIR="$BUILD_ROOT/sherpa-onnx"
DIST_DIR="$SCRIPT_DIR/dist"

rm -rf "$BUILD_ROOT" "$DIST_DIR"
mkdir -p "$BUILD_ROOT" "$DIST_DIR"
git clone --depth 1 --branch "$SHERPA_TAG" "$SHERPA_REPOSITORY" "$SHERPA_DIR"
test "$(git -C "$SHERPA_DIR" rev-parse HEAD)" = "$SHERPA_COMMIT"

ASSETS="$SHERPA_DIR/wasm/vad-asr/assets"
rm -rf "$ASSETS"
mkdir -p "$ASSETS"
printf 'loaded dynamically by Ekko\n' > "$ASSETS/tokens.txt"
for filename in silero_vad.onnx nemo-transducer-encoder.onnx \
    nemo-transducer-decoder.onnx nemo-transducer-joiner.onnx; do
    printf 'loaded dynamically by Ekko\n' > "$ASSETS/$filename"
done

# Export Emscripten's FS object so the worker can mount checksum-verified model
# downloads before creating the recognizer.
perl -0pi -e "s/'HEAPF64'\]/'HEAPF64','FS'\]/" \
    "$SHERPA_DIR/wasm/vad-asr/CMakeLists.txt"
grep -q "'HEAPF64','FS'" "$SHERPA_DIR/wasm/vad-asr/CMakeLists.txt"

if [[ "${EKKO_WASM_BUILDER:-local}" == "docker" ]]; then
    command -v docker >/dev/null || { echo "docker is required" >&2; exit 1; }
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        --volume "$SHERPA_DIR:/work" \
        --workdir /work \
        "$EMSDK_IMAGE" \
        ./build-wasm-simd-vad-asr.sh
else
    if [[ ! -x "$EMSDK_ROOT/emsdk" ]]; then
        mkdir -p "$(dirname "$EMSDK_ROOT")"
        git clone --depth 1 --branch "$EMSDK_TAG" \
            https://github.com/emscripten-core/emsdk.git "$EMSDK_ROOT"
    fi
    test "$(git -C "$EMSDK_ROOT" rev-parse HEAD)" = "$EMSDK_COMMIT"
    if [[ ! -x "$EMSDK_ROOT/upstream/emscripten/emcc" ]]; then
        "$EMSDK_ROOT/emsdk" install "$EMSDK_TAG"
        "$EMSDK_ROOT/emsdk" activate "$EMSDK_TAG"
    fi
    # shellcheck disable=SC1091
    source "$EMSDK_ROOT/emsdk_env.sh"
    (
        cd "$SHERPA_DIR"
        ./build-wasm-simd-vad-asr.sh
    )
fi

RUNTIME_DIR="$SHERPA_DIR/build-wasm-simd-vad-asr/install/bin/wasm/vad-asr"
for filename in sherpa-onnx-asr.js sherpa-onnx-vad.js \
    sherpa-onnx-wasm-main-vad-asr.js sherpa-onnx-wasm-main-vad-asr.wasm \
    sherpa-onnx-wasm-main-vad-asr.data; do
    test -f "$RUNTIME_DIR/$filename"
    cp "$RUNTIME_DIR/$filename" "$DIST_DIR/$filename"
done

ORT_ARCHIVE="$BUILD_ROOT/onnxruntime-web-$ORT_WEB_VERSION.tgz"
ORT_DIR="$BUILD_ROOT/onnxruntime-web"
curl --fail --location --retry 3 \
    "https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-$ORT_WEB_VERSION.tgz" \
    --output "$ORT_ARCHIVE"
printf '%s  %s\n' "$ORT_WEB_SHA256" "$ORT_ARCHIVE" | shasum -a 256 --check
mkdir -p "$ORT_DIR"
tar -xzf "$ORT_ARCHIVE" -C "$ORT_DIR" \
    --strip-components=2 \
    package/dist/ort.wasm.min.js \
    package/dist/ort.wasm.min.js.map \
    package/dist/ort-wasm-simd-threaded.mjs \
    package/dist/ort-wasm-simd-threaded.wasm
cp "$ORT_DIR"/* "$DIST_DIR/"
curl --fail --location --retry 3 \
    "https://raw.githubusercontent.com/microsoft/onnxruntime/v$ORT_WEB_VERSION/LICENSE" \
    --output "$DIST_DIR/ONNXRUNTIME_LICENSE"
printf '%s  %s\n' "$ORT_LICENSE_SHA256" "$DIST_DIR/ONNXRUNTIME_LICENSE" | \
    shasum -a 256 --check

mkdir -p "$DIST_DIR/model-files/tiny"
curl --fail --location --retry 3 "$VAD_URL" \
    --output "$DIST_DIR/model-files/tiny/silero_vad.onnx"
printf '%s  %s\n' "$VAD_SHA256" "$DIST_DIR/model-files/tiny/silero_vad.onnx" | \
    shasum -a 256 --check

cp "$SCRIPT_DIR"/src/* "$DIST_DIR/"
cp "$SCRIPT_DIR/model-config.json" "$DIST_DIR/model-config.json"
cp "$SCRIPT_DIR/README.md" "$DIST_DIR/README.md"
cp "$SCRIPT_DIR/THIRD_PARTY_NOTICES.md" "$DIST_DIR/THIRD_PARTY_NOTICES.md"
cp "$SHERPA_DIR/LICENSE" "$DIST_DIR/SHERPA_ONNX_LICENSE"

printf 'Built browser Space in %s\n' "$DIST_DIR"
