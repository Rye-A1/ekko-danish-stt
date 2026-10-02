import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const space = new URL("../demo/ekko-tiny-browser/src/", import.meta.url);
const worker = readFileSync(new URL("worker.js", space), "utf8");
const workerContext = {
  importScripts() {},
  self: {location: {href: "http://localhost/worker.js"}},
  URL,
  Float32Array,
};
runInNewContext(`${worker}\nglobalThis.toWordsForTest = toWords;`, workerContext);
const partialResult = {
  text: "første andet tredje fjerde",
  tokens: ["▁første", "▁andet"],
  timestamps: [0.1, 0.5],
};
assert.deepEqual(Array.from(workerContext.toWordsForTest(partialResult, 2)), [],
  "partial timestamp tokens must not override the recognizer's full text");
const completeResult = {...partialResult, text: "første andet"};
assert.equal(workerContext.toWordsForTest(completeResult, 2).length, 2,
  "aligned timestamps must remain available");
assert.deepEqual(Array.from(workerContext.toWordsForTest({
  ...completeResult, text: "første tredje",
}, 2)), [], "timestamps for different words must not be attached to the transcript");

const pncWorker = readFileSync(new URL("pnc-worker.js", space), "utf8");
const pncContext = {
  importScripts() {},
  ort: {env: {wasm: {}}},
  self: {location: {href: "http://localhost/pnc-worker.js"}},
  URL,
};
runInNewContext(`${pncWorker}
  punctuateWords = async (words) => words;
  globalThis.punctuateForTest = punctuateSegments;
`, pncContext);

const longText = Array.from({length: 70}, (_, index) => `ord${index + 1}`).join(" ");
const longSegment = {
  id: "long", text: longText,
  words: [{text: "ord1", start: 0, end: 0.2}, {text: "ord2", start: 0.2, end: 0.4}],
};
const shortSegment = {
  id: "short", text: "kort igen",
  words: [{text: "kort", start: 20, end: 20.3}, {text: "igen", start: 20.3, end: 20.6}],
};
const [longOutput, shortOutput] = await pncContext.punctuateForTest([longSegment, shortSegment]);
assert.equal(longOutput.text.split(/\s+/).length, 70,
  "the long result must survive punctuation after a short result arrives");
assert(longOutput.text.endsWith("ord70"));
assert.equal(longOutput.rawText, longText);
assert.equal(longOutput.words.length, 0,
  "incomplete word timestamps must be discarded, not presented as complete");
assert.equal(shortOutput.text, "kort igen");
assert.equal(shortOutput.words.length, 2,
  "aligned timestamps in a following short result must be preserved");
const [contextOutput] = await pncContext.punctuateForTest([shortSegment], ["tidligere", "sætning"]);
assert.equal(contextOutput.text, "kort igen",
  "context words must not be emitted as part of the new segment");
assert.equal(contextOutput.words.length, 2,
  "context words must not shift aligned timestamps for the new segment");
const [capitalizedContextOutput] = await pncContext.punctuateForTest(
  [shortSegment], ["tidligere", "sætning"], true,
);
assert.equal(capitalizedContextOutput.text, "Kort igen",
  "a committed sentence ending must capitalize the next segment");

const app = readFileSync(new URL("app.js", space), "utf8");
const renderFunctions = app.slice(app.indexOf("function renderWords("), app.indexOf("function renderTranscript("));
const document = {
  createTextNode: (value) => ({textContent: value}),
  createElement: () => ({textContent: "", style: {}}),
};
const appContext = {document};
runInNewContext(`${renderFunctions}\nglobalThis.renderWordsForTest = renderWords;`, appContext);
const paragraph = {
  children: [],
  replaceChildren() { this.children = []; },
  appendChild(child) { this.children.push(child); },
};
appContext.renderWordsForTest(paragraph, longSegment, false);
assert.equal(paragraph.children.map((child) => child.textContent).join(""), longText,
  "the visible transcript must use the full text even when timestamps are partial");

const sentRequests = [];
const queueState = {
  pncFailed: false, pncPending: null, pncRequestId: 0, pncFormattedCount: 0,
  pncCommittedWords: 0,
  nextSegmentId: 0, rawSegments: [], segments: [], previewSegment: null,
  punctuationApplied: false,
};
const queueContext = {
  state: queueState,
  pncWorker: {postMessage: (message) => sentRequests.push(message)},
  elements: {pncStatus: {textContent: ""}},
  ensurePunctuation() {},
  signedAsset: (path) => path,
  setConversationVisible() {},
  renderTranscript() {},
  persistCurrentConversation() {},
  console,
};
const queueFunctions = app.slice(app.indexOf("function wordsOf("),
  app.indexOf("class StreamingLinearResampler"));
const appendFunction = app.slice(app.indexOf("function appendResult("),
  app.indexOf("async function startRecording()"));
runInNewContext(`${queueFunctions}\n${appendFunction}\n` +
  "globalThis.queueTest = {appendResult, requestPunctuation, applyPunctuationResult};", queueContext);

queueContext.queueTest.appendResult({text: longText});
queueContext.queueTest.requestPunctuation();
assert.equal(sentRequests.length, 1);
assert.equal(sentRequests[0].segments.length, 1);
queueContext.queueTest.appendResult({text: "kort igen"});
queueContext.queueTest.requestPunctuation();
assert.equal(sentRequests.length, 1, "a new segment must wait while PnC is working");
assert.equal(queueState.segments.length, 2, "both raw segments stay visible");

const longFormatted = {...sentRequests[0].segments[0], text: longText
  .replace(/^ord1/, "Ord1").replace("ord50 ord51", "ord50. Ord51") + "."};
queueContext.queueTest.applyPunctuationResult({
  requestId: sentRequests[0].requestId, segments: [longFormatted],
});
assert.equal(queueState.segments[0].text, longFormatted.text);
assert.equal(queueState.segments[1].text, "kort igen",
  "formatting the long segment must not replace the later short segment");
assert.equal(sentRequests.length, 2, "the waiting segment is sent after the first finishes");
assert.equal(queueState.pncCommittedWords, 50);
assert.equal(sentRequests[1].segments.length, 2);
assert.equal(sentRequests[1].segments[0].id, queueState.rawSegments[0].id);
assert(sentRequests[1].segments[0].text.startsWith("ord51 "));
assert.equal(sentRequests[1].segments[1].id, queueState.rawSegments[1].id);
assert.equal(sentRequests[1].contextWords.length, 12);
assert.equal(sentRequests[1].contextWords.at(-1), "ord50");
assert.equal(sentRequests[1].capitalizeFirst, true);

queueContext.queueTest.appendResult({text: "tredje segment"});
queueContext.queueTest.requestPunctuation();
const tailFormatted = {...sentRequests[1].segments[0], text: sentRequests[1].segments[0].text
  .replace(/^ord51/, "Ord51").replace(/ord70$/, "ord70.")};
const shortFormatted = {...sentRequests[1].segments[1], text: "Kort igen."};
queueContext.queueTest.applyPunctuationResult({
  requestId: sentRequests[1].requestId, segments: [tailFormatted, shortFormatted],
});
assert.equal(queueState.segments[0].text.split(/\s+/).slice(0, 50).join(" "),
  longFormatted.text.split(/\s+/).slice(0, 50).join(" "),
  "words through the last stable sentence stop must remain unchanged");
assert.equal(queueState.segments[0].text.split(/\s+/).length, 70);
assert.equal(queueState.segments[2].text, "tredje segment",
  "a segment arriving during PnC must remain visible");
assert.equal(sentRequests.length, 3);
assert.equal(queueState.pncCommittedWords, 70);
assert.equal(sentRequests[2].segments[0].id, queueState.rawSegments[1].id);
assert.equal(sentRequests[2].segments[1].id, queueState.rawSegments[2].id);
queueContext.queueTest.applyPunctuationResult({
  requestId: sentRequests[0].requestId, segments: [longFormatted],
});
assert.equal(queueState.segments[2].text, "tredje segment",
  "a stale PnC response must be ignored");
queueContext.queueTest.applyPunctuationResult({
  requestId: sentRequests[2].requestId,
  segments: [
    {...sentRequests[2].segments[0], text: "Kort igen."},
    {...sentRequests[2].segments[1], text: "Tredje segment."},
  ],
});
assert.equal(queueState.segments.length, 3);
assert(queueState.punctuationApplied);
assert.deepEqual(
  queueState.segments.flatMap((segment) => segment.text.split(/\s+/)
    .map((word) => word.replace(/[.!?]$/u, "").toLowerCase())),
  queueState.rawSegments.flatMap((segment) => segment.text.split(/\s+/)),
  "reformatting an open sentence must retain every recognized word in order",
);
console.log("browser transcript integrity passed");
