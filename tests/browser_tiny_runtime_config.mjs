import assert from "node:assert/strict";
import {createHash, webcrypto} from "node:crypto";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const workerSource = readFileSync(new URL("../demo/ekko-tiny-browser/src/worker.js", import.meta.url), "utf8");
const runtimeBytes = readFileSync(
  new URL("../demo/ekko-tiny-browser/runtime-config.json", import.meta.url),
);
const runtimeDigest = createHash("sha256").update(runtimeBytes).digest("hex");
const options = [];
const writes = [];
const config = {
  schema: "ekko-browser-model-v1",
  name: "Ekko Tiny",
  runtime: "test",
  files: [{
    path: "runtime-config.json",
    url: "https://huggingface.co/RyeAI/ekko-v1-tiny/resolve/test/onnx-sherpa/runtime-config.json",
    sha256: runtimeDigest,
    bytes: runtimeBytes.length,
  }],
};
const context = {
  URL,
  Uint8Array,
  Float32Array,
  TextDecoder,
  crypto: webcrypto,
  importScripts() {},
  OfflineRecognizer: class {
    constructor(value) { options.push(value); }
  },
  createVad() { return {}; },
  self: {
    location: {href: "https://example.static.hf.space/worker.js"},
    postMessage() {},
  },
  fetch: async (url) => new Response(
    url.includes("runtime-config.json") ? runtimeBytes : JSON.stringify(config),
  ),
};
runInNewContext(`${workerSource}\nglobalThis.testInitialize = initialize; globalThis.testModule = Module;`, context);
context.testModule.FS = {writeFile: (...args) => writes.push(args)};
context.testModule.onRuntimeInitialized();
await context.testInitialize("model-config.json");

assert.equal(options.length, 1);
assert.equal(options[0].featConfig.sampleRate, 16000);
assert.equal(options[0].featConfig.featureDim, 80);
assert.equal(options[0].modelConfig.modelType, "nemo_transducer");
assert.equal(writes.length, 0, "runtime metadata is parsed, not mounted as ONNX data");
console.log("Tiny browser uses the checksum-verified Hub runtime configuration");
