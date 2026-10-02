import assert from "node:assert/strict";
import {createHash, webcrypto} from "node:crypto";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const source = new URL("../demo/ekko-tiny-browser/src/", import.meta.url);
const modelBytes = Uint8Array.from(Buffer.from("checksum-verified model bytes"));
const file = {
  path: "model.onnx",
  url: "https://huggingface.co/example/resolve/revision/model.onnx",
  bytes: modelBytes.length,
  sha256: createHash("sha256").update(modelBytes).digest("hex"),
};

for (const name of ["worker.js", "pnc-worker.js"]) {
  const stored = new Map();
  const requests = [];
  const progress = [];
  const cache = {
    async match(url) { return stored.has(url) ? new Response(stored.get(url)) : null; },
    async put(url, response) { stored.set(url, new Uint8Array(await response.arrayBuffer())); },
    async delete(url) { return stored.delete(url); },
  };
  const context = {
    URL, Uint8Array, Float32Array, Response,
    crypto: webcrypto,
    importScripts() {},
    ort: {env: {wasm: {}}},
    self: {
      location: {href: `https://example.static.hf.space/${name}`},
      caches: {open: async () => cache},
      postMessage(message) { progress.push(message); },
    },
    fetch: async (_url, options) => {
      requests.push(options.cache);
      return new Response(modelBytes);
    },
  };
  const worker = readFileSync(new URL(name, source), "utf8");
  runInNewContext(`${worker}\nglobalThis.fetchVerifiedForTest = fetchVerified;`, context);

  const first = await context.fetchVerifiedForTest(file, 0, file.bytes);
  const second = await context.fetchVerifiedForTest(file, 0, file.bytes);
  assert.deepEqual(Array.from(first), Array.from(modelBytes));
  assert.deepEqual(Array.from(second), Array.from(modelBytes));
  assert.deepEqual(requests, ["force-cache"], `${name} should reuse the verified persistent copy`);
  assert(progress.some((message) => message.cached === true),
    `${name} should identify progress from persistent storage`);

  stored.set(file.url, new Uint8Array(modelBytes.length).fill(0));
  const repaired = await context.fetchVerifiedForTest(file, 0, file.bytes);
  assert.deepEqual(Array.from(repaired), Array.from(modelBytes));
  assert.deepEqual(requests, ["force-cache", "reload"],
    `${name} should discard corrupt cache data and fetch a fresh copy`);

  stored.set(file.url, new Uint8Array(modelBytes.length + 1));
  await context.fetchVerifiedForTest(file, 0, file.bytes);
  assert.deepEqual(requests, ["force-cache", "reload", "reload"],
    `${name} should recover from a cached file with the wrong size`);
}
console.log("browser model cache reuse and repair passed");
