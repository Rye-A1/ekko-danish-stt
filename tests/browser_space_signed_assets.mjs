import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const base = new URL("../demo/ekko-tiny-browser/src/", import.meta.url);
const signedOrigin = "https://ryeai-ekko-tiny-browser.static.hf.space/";
const signature = "test-signature";

function loadWorker(filename, options = {}) {
  const imports = [];
  const context = {
    URL,
    Float32Array,
    importScripts: (...urls) => imports.push(...urls),
    self: {
      location: {href: `${signedOrigin}${filename}?v=4&__sign=${signature}`},
      postMessage() {},
    },
    ...options,
  };
  const source = readFileSync(new URL(filename, base), "utf8");
  runInNewContext(`${source}\nglobalThis.assetForTest = signedAsset;`, context);
  return {context, imports};
}

for (const filename of ["worker.js", "pnc-worker.js"]) {
  const options = filename === "pnc-worker.js" ? {ort: {env: {wasm: {}}}} : {};
  const {context, imports} = loadWorker(filename, options);
  assert(imports.length, `${filename} should import runtime scripts`);
  for (const path of imports) {
    assert.equal(new URL(path).searchParams.get("__sign"), signature);
  }
  const model = new URL(context.assetForTest("model-files/tiny/encoder.onnx?sha256=abc"));
  assert.equal(model.searchParams.get("sha256"), "abc");
  assert.equal(model.searchParams.get("__sign"), signature);
  assert.equal(
    context.assetForTest("https://example.com/model.onnx"),
    "https://example.com/model.onnx",
    "the private Space signature must not be sent to another origin",
  );
  if (filename === "pnc-worker.js") {
    for (const path of Object.values(context.ort.env.wasm.wasmPaths)) {
      assert.equal(new URL(path).searchParams.get("__sign"), signature);
    }
  }
}

console.log("private Space worker assets retain their access signature");
