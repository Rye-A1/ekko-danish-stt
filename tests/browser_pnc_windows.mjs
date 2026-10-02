import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const worker = readFileSync(new URL("../demo/ekko-tiny-browser/src/pnc-worker.js", import.meta.url), "utf8");
const context = {
  importScripts() {},
  ort: {
    env: {wasm: {}},
    Tensor: class {
      constructor(_type, _data, dims) { this.dims = dims; }
    },
  },
  self: {location: {href: "http://localhost/pnc-worker.js"}},
  URL,
};
runInNewContext(`${worker}
globalThis.probe = async (words, extraVocabulary = {}) => {
  vocabulary = {"[CLS]": 101, "[SEP]": 102, "[UNK]": 100, ...extraVocabulary};
  labels = {"0": "O"};
  const lengths = [];
  session = {run: async ({input_ids}) => {
    const length = input_ids.dims[1];
    lengths.push(length);
    return {logits: {dims: [1, length, 1], data: new Float32Array(length)}};
  }};
  const {prefix, specialTokens} = windowTokenIndex(words);
  return {
    firstEnd: maximumWindowEnd(words, 0, prefix, specialTokens),
    output: await punctuateWords(words),
    lengths,
  };
};`, context);

const ordinary = Array.from({length: 300}, (_, index) => `ord${index}`);
const first = await context.probe(ordinary);
assert.equal(first.firstEnd, 126);
assert.deepEqual(Array.from(first.lengths), [128, 128, 98]);
assert.equal(first.output.length, ordinary.length);
assert.deepEqual(Array.from(first.output), ordinary);

const split = await context.probe(Array(100).fill("aaaa"), {a: 1, "##a": 2});
assert.equal(split.firstEnd, 31);
assert.equal(split.output.length, 100);
assert.ok(split.lengths.every((length) => length <= 128));
console.log("browser PnC indexed windows passed");
