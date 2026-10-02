import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";

const messages = [];
const worker = readFileSync(new URL("../demo/ekko-tiny-browser/src/worker.js", import.meta.url), "utf8");
const context = {
  importScripts() {},
  self: {location: {href: "http://localhost/worker.js"}, postMessage: (message) => messages.push(message)},
  URL,
  Float32Array,
};
runInNewContext(`${worker}\nglobalThis.exercise = {resetLiveAudio, acceptLiveChunk, drainVad};`, context);
const queue = [];
context.fakeVad = {
  acceptWaveform() {},
  isDetected: () => true,
  isEmpty: () => queue.length === 0,
  front: () => queue[0],
  pop: () => queue.shift(),
};
runInNewContext(`vad = fakeVad; decodeSamples = (samples, source, start) => ({
  source, text: "hej verden", start, end: start + samples.length / SAMPLE_RATE,
  words: [{text: "hej", start, end: start + 0.3}],
});`, context);

context.exercise.resetLiveAudio();
context.exercise.acceptLiveChunk(new Float32Array(16000 * 4));
assert(messages.some((message) => message.type === "live-preview"),
  "continuous speech must produce provisional text before recording stops");
assert(!messages.some((message) => message.type === "live-result"));

queue.push({start: 0, samples: new Float32Array(16000 * 4)});
context.exercise.drainVad();
const relevant = messages.filter((message) => ["live-preview", "live-preview-clear", "live-result"].includes(message.type));
assert.deepEqual(relevant.map((message) => message.type),
  ["live-preview", "live-preview-clear", "live-result"]);
messages.length = 0;
queue.push(
  {start: 16000 * 4, samples: new Float32Array(16000 * 20)},
  {start: 16000 * 24, samples: new Float32Array(16000)},
);
context.exercise.drainVad();
const orderedResults = messages.filter((message) => message.type === "live-result");
assert.equal(orderedResults.length, 2, "a short chunk must not replace a pending long result");
assert(orderedResults[0].result.start < orderedResults[1].result.start);

let Processor;
class AudioWorkletProcessor {
  constructor() { this.port = {postMessage: (message) => messages.push(message)}; }
}
runInNewContext(
  readFileSync(new URL("../demo/ekko-tiny-browser/src/capture-worklet.js", import.meta.url), "utf8"),
  {AudioWorkletProcessor, registerProcessor: (_name, value) => { Processor = value; }, Float32Array},
);
const capture = new Processor();
capture.process([[new Float32Array([1, 2])]]);
capture.port.onmessage({data: {type: "stop"}});
capture.process([[new Float32Array([3, 4])]]);
assert.equal(messages.filter((message) => message instanceof Float32Array).length, 1);
assert(messages.some((message) => message.type === "stopped"));
const recovered = [];
const gapContext = {
  importScripts() {},
  self: {location: {href: "http://localhost/worker.js"}, postMessage: (message) => recovered.push(message)},
  URL,
  Float32Array,
};
runInNewContext(`${worker}\nglobalThis.exerciseGap = {
  recoverSpeechGap, drainVad, retainRecentPreRoll,
  historyLength: () => liveHistory.length,
  historyStart: () => liveHistoryStart,
};`, gapContext);
runInNewContext(`
  lastLiveResultEndSample = 0;
  liveHistory = new Float32Array(SAMPLE_RATE * 8).fill(0.1);
  decodeSamples = (samples, source, start) => ({
    source, text: "manglende tale her", start, end: start + samples.length / SAMPLE_RATE,
  });
`, gapContext);
gapContext.exerciseGap.recoverSpeechGap(16000 * 8);
assert.equal(recovered.length, 1, "energetic speech in a long VAD gap must be recovered");
assert.equal(recovered[0].result.source, "microphone-gap");
recovered.length = 0;
runInNewContext(`
  lastLiveResultEndSample = 0;
  liveHistoryStart = 0;
  liveHistory = new Float32Array(SAMPLE_RATE * 8);
`, gapContext);
gapContext.exerciseGap.recoverSpeechGap(16000 * 8);
assert.equal(recovered.length, 0, "quiet gaps must not produce transcript segments");
runInNewContext(`
  lastLiveResultEndSample = 0;
  liveHistoryStart = 0;
  liveHistory.fill(0.1);
  decodeSamples = (samples, source, start) => ({source, text: "det", start, end: start + samples.length / SAMPLE_RATE});
`, gapContext);
gapContext.exerciseGap.recoverSpeechGap(16000 * 8);
assert.equal(recovered.length, 0, "one-word gap decodes must not become false chunks");
const historyBeforeEmptyVad = gapContext.exerciseGap.historyLength();
runInNewContext(`
  const emptyQueue = [{start: SAMPLE_RATE * 6, samples: new Float32Array(SAMPLE_RATE)}];
  vad = {
    isEmpty: () => emptyQueue.length === 0,
    front: () => emptyQueue[0],
    pop: () => emptyQueue.shift(),
  };
  decodeSamples = (samples, source, start) => ({source, text: "", start, end: start + samples.length / SAMPLE_RATE});
`, gapContext);
gapContext.exerciseGap.drainVad();
assert.equal(gapContext.exerciseGap.historyLength(), historyBeforeEmptyVad,
  "an empty VAD decode must not discard audio needed to recover a later speech gap");
runInNewContext(`
  lastLiveResultEndSample = SAMPLE_RATE * 4;
  liveSamplesAccepted = SAMPLE_RATE * 8;
`, gapContext);
gapContext.exerciseGap.retainRecentPreRoll();
assert(gapContext.exerciseGap.historyStart() < 16000 * 4,
  "delayed VAD results must retain audio following the last finalized segment");

const retryContext = {
  importScripts() {},
  self: {location: {href: "http://localhost/worker.js"}, postMessage() {}},
  URL,
  Float32Array,
};
runInNewContext(`${worker}\nglobalThis.retrySparse = retrySparseSegment;`, retryContext);
runInNewContext(`
  liveHistory = new Float32Array(SAMPLE_RATE * 8).fill(0.1);
  let retryCalls = 0;
  decodeSamples = (_samples, source, start) => {
    retryCalls += 1;
    return {
      source,
      text: start > 1 ? "det er en længere samtale" : "det er lidt",
      start,
      end: start + 5,
      duration: 5,
      words: [{text: "det", start: start + 0.4, end: start + 0.6}],
    };
  };
  globalThis.retryCallCount = () => retryCalls;
`, retryContext);
const sparseSpan = {start: 16000, samples: new Float32Array(16000 * 5)};
const sparseResult = {
  source: "microphone", text: "det", start: 1, end: 6, duration: 5,
  words: [{text: "det", start: 1.4, end: 1.6}],
};
const recoveredSparse = retryContext.retrySparse(sparseSpan, sparseResult);
assert.equal(recoveredSparse.text, "det er en længere samtale");
assert.equal(recoveredSparse.start, 1, "retry must preserve VAD segment timing");
assert.equal(recoveredSparse.end, 6);
assert(Math.abs(recoveredSparse.words[0].start - 1.4) < 1e-6,
  "retry must preserve word timing after shifting the decode window");
assert.equal(retryContext.retryCallCount(), 2);
assert.equal(retryContext.retrySparse(
  {start: 16000, samples: new Float32Array(16000 * 4)}, sparseResult,
), sparseResult, "short segments must not incur retries");
console.log("browser live chunking, sparse retry, gap recovery, and capture drain passed");
