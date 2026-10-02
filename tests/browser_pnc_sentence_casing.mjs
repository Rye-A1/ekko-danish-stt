import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const worker = readFileSync(new URL("../demo/ekko-tiny-browser/src/pnc-worker.js", import.meta.url), "utf8");
const context = {
  importScripts() {},
  ort: {env: {wasm: {}}},
  self: {location: {href: "http://localhost/pnc-worker.js"}},
  URL,
};
runInNewContext(`${worker}\nglobalThis.render = stabilizeRenderedWords;`, context);

function render(words) {
  return Array.from(context.render(words));
}

assert.deepEqual(render([]), []);
assert.deepEqual(render(["hej"]), ["Hej"]);
assert.deepEqual(render(["hej.", "hvordan", "går", "det?"]),
  ["Hej.", "Hvordan", "går", "det?"]);
assert.deepEqual(render(["det", "er", "f.eks.", "svært.", "men", "muligt"]),
  ["Det", "er", "f.eks.", "svært.", "Men", "muligt"]);
assert.deepEqual(render(["ja.", "CPR", "virker.", "iPhone", "også"]),
  ["Ja.", "CPR", "virker.", "iPhone", "også"]);
assert.deepEqual(render(["ja?", "det", "gør", "det!"]),
  ["Ja?", "Det", "gør", "det!"]);
console.log("browser PnC sentence casing passed");
