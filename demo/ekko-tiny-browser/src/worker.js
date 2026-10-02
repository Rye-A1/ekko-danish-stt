/* global OfflineRecognizer, createVad */

const SAMPLE_RATE = 16000;
const VAD_WINDOW = 512;
const SHORT_UTTERANCE_MIN_SECONDS = 0.12;
const LIVE_PRE_ROLL_SECONDS = 0.32;
const LIVE_PRE_ROLL_SAMPLES = Math.round(SAMPLE_RATE * LIVE_PRE_ROLL_SECONDS);
const LIVE_HISTORY_LIMIT = SAMPLE_RATE * 90;
const PREVIEW_MIN_SECONDS = 3;
const SPARSE_RETRY_MIN_SECONDS = 5;
const SPARSE_RETRY_WORDS_PER_SECOND = 2.5;
const SPARSE_RETRY_SHIFT_SAMPLES = 64;
const SPARSE_RETRY_MIN_EXTRA_WORDS = 3;

// A private Hugging Face static Space signs its iframe URL. Worker subrequests
// do not inherit that signature, so carry it to runtime scripts and model files.
function signedAsset(path) {
  const base = new URL(self.location.href);
  const url = new URL(path, base);
  const signature = base.searchParams.get("__sign");
  if (signature && url.origin === base.origin) url.searchParams.set("__sign", signature);
  return url.href;
}

let runtimeResolve;
const runtimeReady = new Promise((resolve) => {
  runtimeResolve = resolve;
});

var Module = {
  locateFile(path) {
    return signedAsset(path);
  },
  setStatus(status) {
    self.postMessage({type: "runtime-status", status});
  },
  onRuntimeInitialized() {
    runtimeResolve();
  },
  print(text) {
    self.postMessage({type: "runtime-log", message: String(text)});
  },
  printErr(text) {
    self.postMessage({type: "runtime-log", message: String(text)});
  },
};
self.Module = Module;

importScripts(signedAsset("sherpa-onnx-asr.js"), signedAsset("sherpa-onnx-vad.js"));
importScripts(signedAsset("sherpa-onnx-wasm-main-vad-asr.js"));

let recognizer = null;
let vad = null;
let initialization = null;
let liveCarry = new Float32Array(0);
let liveHistory = new Float32Array(0);
let liveHistoryStart = 0;
let liveSamplesAccepted = 0;
let previewStart = null;
let nextPreviewSample = 0;
let lastLiveResultEndSample = null;

function concatenate(left, right) {
  const output = new Float32Array(left.length + right.length);
  output.set(left);
  output.set(right, left.length);
  return output;
}

function resetLiveAudio() {
  liveCarry = new Float32Array(0);
  liveHistory = new Float32Array(0);
  liveHistoryStart = 0;
  liveSamplesAccepted = 0;
  previewStart = null;
  nextPreviewSample = 0;
  lastLiveResultEndSample = null;
}

function clearLivePreview() {
  if (previewStart === null) return;
  previewStart = null;
  nextPreviewSample = 0;
  self.postMessage({type: "live-preview-clear"});
}

function appendLiveHistory(samples) {
  liveHistory = concatenate(liveHistory, samples);
  liveSamplesAccepted += samples.length;
  if (liveHistory.length > LIVE_HISTORY_LIMIT) {
    const discard = liveHistory.length - LIVE_HISTORY_LIMIT;
    liveHistory = liveHistory.slice(discard);
    liveHistoryStart += discard;
  }
}

function addSegmentPreRoll(segment) {
  const segmentStart = Number(segment.start);
  const prefixStart = Math.max(liveHistoryStart, segmentStart - LIVE_PRE_ROLL_SAMPLES);
  const prefixEnd = Math.min(segmentStart, liveHistoryStart + liveHistory.length);
  if (prefixEnd <= prefixStart) {
    return {samples: segment.samples, start: segmentStart};
  }
  const from = prefixStart - liveHistoryStart;
  const to = prefixEnd - liveHistoryStart;
  return {
    samples: concatenate(liveHistory.slice(from, to), segment.samples),
    start: prefixStart,
  };
}

function retainRecentPreRoll() {
  const keepFrom = Math.max(
    liveHistoryStart,
    (lastLiveResultEndSample ?? liveSamplesAccepted) - LIVE_PRE_ROLL_SAMPLES,
  );
  const discard = keepFrom - liveHistoryStart;
  if (discard > 0) {
    liveHistory = liveHistory.slice(discard);
    liveHistoryStart = keepFrom;
  }
}

function peakNormalize(samples) {
  let peak = 0;
  for (let index = 0; index < samples.length; index += 1) {
    peak = Math.max(peak, Math.abs(samples[index]));
  }
  if (peak === 0 || peak === 0.95) {
    return samples;
  }
  const scale = 0.95 / peak;
  const normalized = new Float32Array(samples.length);
  for (let index = 0; index < samples.length; index += 1) {
    normalized[index] = samples[index] * scale;
  }
  return normalized;
}

async function sha256(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes.buffer);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
}

async function fetchVerified(file, completedBytes, totalBytes) {
  const url = signedAsset(file.url);
  let cache = null;
  try { cache = await self.caches?.open("ekko-browser-models-v1"); } catch { /* Storage is optional. */ }
  for (let attempt = 0; attempt < 2; attempt += 1) {
    let cached = null;
    if (!attempt && cache) {
      try { cached = await cache.match(url); } catch { /* Use the network. */ }
    }
    const response = cached || await fetch(url, {cache: attempt ? "reload" : "force-cache"});
    if (!response.ok || !response.body) {
      throw new Error(`Kunne ikke hente ${file.path} (${response.status})`);
    }
    const reader = response.body.getReader();
    const bytes = new Uint8Array(file.bytes);
    let loaded = 0;
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      if (loaded + value.byteLength > bytes.length) {
        await reader.cancel();
        loaded = bytes.length + 1;
        break;
      }
      bytes.set(value, loaded);
      loaded += value.byteLength;
      self.postMessage({
        type: "model-progress",
        cached: Boolean(cached),
        file: file.path,
        loaded,
        fileBytes: file.bytes,
        overall: Math.min(1, (completedBytes + loaded) / totalBytes),
      });
    }
    if (loaded === file.bytes && await sha256(bytes) === file.sha256) {
      if (!cached && cache) {
        try { await cache.put(url, new Response(bytes)); } catch { /* Keep using the verified bytes. */ }
      }
      return bytes;
    }
    if (cached) {
      try { await cache.delete(url); } catch { /* Retry without this entry. */ }
    }
    if (attempt) throw new Error(`${file.path} fejlede størrelses- eller SHA-256-kontrol`);
  }
}

function createRecognizer(runtimeConfig) {
  recognizer = new OfflineRecognizer(
    {
      featConfig: {
        sampleRate: runtimeConfig.sampleRate,
        featureDim: runtimeConfig.featureDim,
      },
      modelConfig: {
        transducer: {
          encoder: "./nemo-transducer-encoder.onnx",
          decoder: "./nemo-transducer-decoder.onnx",
          joiner: "./nemo-transducer-joiner.onnx",
        },
        tokens: "./tokens.txt",
        modelType: runtimeConfig.modelType,
        provider: "cpu",
        numThreads: 1,
        debug: 0,
      },
      decodingMethod: "greedy_search",
      maxActivePaths: 4,
    },
    Module,
  );
  vad = createVad(Module, {
    sileroVad: {
      model: "./silero_vad.onnx",
      threshold: 0.5,
      minSilenceDuration: 0.5,
      minSpeechDuration: SHORT_UTTERANCE_MIN_SECONDS,
      maxSpeechDuration: 20,
      windowSize: VAD_WINDOW,
    },
    sampleRate: SAMPLE_RATE,
    numThreads: 1,
    provider: "cpu",
    debug: 0,
    bufferSizeInSeconds: 90,
  });
}

async function initialize(configUrl) {
  if (recognizer) return;
  await runtimeReady;
  const response = await fetch(signedAsset(configUrl), {cache: "no-cache"});
  if (!response.ok) throw new Error("Kunne ikke hente modelkonfigurationen");
  const config = await response.json();
  if (config.schema !== "ekko-browser-model-v1") {
    throw new Error("Ukendt modelkonfiguration");
  }
  if (!Module.FS) throw new Error("WASM-filsystemet er ikke eksporteret");

  const totalBytes = config.files.reduce((sum, file) => sum + file.bytes, 0);
  let completedBytes = 0;
  let runtimeConfig = null;
  for (const file of config.files) {
    const bytes = await fetchVerified(file, completedBytes, totalBytes);
    if (file.path === "runtime-config.json") {
      runtimeConfig = JSON.parse(new TextDecoder().decode(bytes));
    } else {
      Module.FS.writeFile(file.path, bytes);
    }
    completedBytes += bytes.byteLength;
  }
  if (
    runtimeConfig?.schema !== "ekko-tiny-onnx-runtime-v1" ||
    runtimeConfig.sampleRate !== SAMPLE_RATE ||
    runtimeConfig.featureDim !== 80 ||
    runtimeConfig.modelType !== "nemo_transducer"
  ) {
    throw new Error("Ukendt Tiny runtimekonfiguration");
  }
  createRecognizer(runtimeConfig);
  self.postMessage({
    type: "ready",
    model: config.name,
    runtime: config.runtime,
    bytes: totalBytes,
  });
}

function toWords(result, duration, timeOffset = 0) {
  const tokens = Array.isArray(result.tokens) ? result.tokens : [];
  const timestamps = Array.isArray(result.timestamps) ? result.timestamps : [];
  const durations = Array.isArray(result.durations) ? result.durations : [];
  if (tokens.length !== timestamps.length) return [];
  const words = [];
  let text = "";
  let start = null;
  let tokenIndices = [];

  const finish = (endHint) => {
    if (!text || start === null) return;
    let end = endHint;
    if (durations.length === tokens.length) {
      for (const index of tokenIndices) {
        end = Math.max(end, Number(timestamps[index]) + Number(durations[index]));
      }
    }
    words.push({
      text,
      start: timeOffset + start,
      end: timeOffset + Math.min(duration, Math.max(start, end)),
    });
  };

  tokens.forEach((rawToken, index) => {
    const token = String(rawToken);
    const timestamp = Number(timestamps[index]);
    if (token === " " || token.startsWith("▁")) {
      finish(timestamp);
      text = token.trim().replace(/^▁/, "");
      start = timestamp;
      tokenIndices = [index];
    } else {
      if (start === null) start = timestamp;
      text += token;
      tokenIndices.push(index);
    }
  });
  if (text && start !== null) {
    const fallback = timestamps.length ? Number(timestamps[timestamps.length - 1]) + 0.08 : start;
    finish(fallback);
  }
  // The recognizer's text is authoritative. A partial token/timestamp list
  // must never become a partial transcript downstream.
  const decodedText = String(result.text || "").trim().replace(/\s+/g, " ");
  if (words.map((word) => word.text).join(" ") !== decodedText) return [];
  return words;
}

function decodeSamples(samples, source, timeOffset = 0) {
  if (!recognizer) throw new Error("Modellen er ikke klar");
  const normalized = peakNormalize(samples);
  const duration = normalized.length / SAMPLE_RATE;
  const stream = recognizer.createStream();
  try {
    stream.acceptWaveform(SAMPLE_RATE, normalized);
    recognizer.decode(stream);
    const result = recognizer.getResult(stream);
    return {
      source,
      text: String(result.text || "").trim(),
      start: timeOffset,
      end: timeOffset + duration,
      duration,
      words: toWords(result, duration, timeOffset),
    };
  } finally {
    stream.free();
  }
}

function wordCount(text) {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

function retrySparseSegment(buffered, result) {
  const duration = buffered.samples.length / SAMPLE_RATE;
  const originalCount = wordCount(result.text);
  if (duration < SPARSE_RETRY_MIN_SECONDS ||
      originalCount >= duration * SPARSE_RETRY_WORDS_PER_SECOND) return result;

  let best = result;
  let bestCount = originalCount;
  for (const shift of [SPARSE_RETRY_SHIFT_SAMPLES, -SPARSE_RETRY_SHIFT_SAMPLES]) {
    const start = buffered.start + shift;
    const end = start + buffered.samples.length;
    if (start < liveHistoryStart || end > liveHistoryStart + liveHistory.length) continue;
    const samples = liveHistory.slice(start - liveHistoryStart, end - liveHistoryStart);
    const candidate = decodeSamples(samples, "microphone", start / SAMPLE_RATE);
    const count = wordCount(candidate.text);
    if (count > bestCount) {
      best = candidate;
      bestCount = count;
    }
  }
  if (bestCount < originalCount + SPARSE_RETRY_MIN_EXTRA_WORDS) return result;

  // Keep the VAD span stable for gap recovery and downstream punctuation.
  const correction = result.start - best.start;
  return {
    ...best,
    start: result.start,
    end: result.end,
    duration: result.duration,
    words: best.words.map((word) => ({
      ...word,
      start: word.start + correction,
      end: word.end + correction,
    })),
  };
}

function recoverSpeechGap(nextStartSample) {
  if (lastLiveResultEndSample === null) return;
  const start = Math.max(lastLiveResultEndSample, liveHistoryStart);
  const end = Math.min(nextStartSample, liveHistoryStart + liveHistory.length);
  if (end - start < SAMPLE_RATE * 2) return;
  const samples = liveHistory.slice(start - liveHistoryStart, end - liveHistoryStart);
  let sumSquares = 0;
  for (const value of samples) sumSquares += value * value;
  if (Math.sqrt(sumSquares / samples.length) < 0.018) return;
  const result = decodeSamples(samples, "microphone-gap", start / SAMPLE_RATE);
  if (result.text.trim().split(/\s+/).length < 3) return;
  self.postMessage({type: "live-result", result});
  lastLiveResultEndSample = end;
  retainRecentPreRoll();
}

function drainVad() {
  while (!vad.isEmpty()) {
    const segment = vad.front();
    vad.pop();
    const buffered = addSegmentPreRoll(segment);
    const result = retrySparseSegment(
      buffered,
      decodeSamples(buffered.samples, "microphone", buffered.start / SAMPLE_RATE),
    );
    recoverSpeechGap(Math.round(result.start * SAMPLE_RATE));
    clearLivePreview();
    if (result.text) {
      self.postMessage({type: "live-result", result});
      lastLiveResultEndSample = Math.round(result.end * SAMPLE_RATE);
      retainRecentPreRoll();
    }
  }
}

function maybePreviewSpeech() {
  if (!vad.isDetected()) {
    clearLivePreview();
    return;
  }
  if (previewStart === null) {
    previewStart = Math.max(liveHistoryStart, liveSamplesAccepted - SAMPLE_RATE / 2);
    nextPreviewSample = previewStart + PREVIEW_MIN_SECONDS * SAMPLE_RATE;
  }
  if (liveSamplesAccepted < nextPreviewSample) return;
  const start = Math.max(previewStart, liveHistoryStart);
  const samples = liveHistory.slice(start - liveHistoryStart);
  const result = decodeSamples(samples, "preview", start / SAMPLE_RATE);
  if (result.text) self.postMessage({type: "live-preview", result});
  const duration = samples.length / SAMPLE_RATE;
  nextPreviewSample = liveSamplesAccepted + Math.max(PREVIEW_MIN_SECONDS, duration / 6) * SAMPLE_RATE;
}

function acceptLiveChunk(samples) {
  liveCarry = concatenate(liveCarry, samples);
  while (liveCarry.length >= VAD_WINDOW) {
    const frame = liveCarry.slice(0, VAD_WINDOW);
    liveCarry = liveCarry.slice(VAD_WINDOW);
    appendLiveHistory(frame);
    vad.acceptWaveform(frame);
    self.postMessage({type: "speech-state", detected: vad.isDetected()});
    drainVad();
    maybePreviewSpeech();
  }
}

function stopLive() {
  if (liveCarry.length) {
    const padded = new Float32Array(VAD_WINDOW);
    padded.set(liveCarry);
    appendLiveHistory(padded);
    vad.acceptWaveform(padded);
  }
  vad.flush();
  drainVad();
  recoverSpeechGap(liveSamplesAccepted);
  clearLivePreview();
  vad.reset();
  resetLiveAudio();
  self.postMessage({type: "live-stopped"});
}

self.onmessage = async (event) => {
  const message = event.data || {};
  try {
    if (message.type === "init") {
      initialization ||= initialize(message.configUrl || "model-config.json");
      await initialization;
    } else if (message.type === "live-start") {
      await initialization;
      vad.reset();
      resetLiveAudio();
    } else if (message.type === "live-chunk") {
      await initialization;
      acceptLiveChunk(new Float32Array(message.samples));
    } else if (message.type === "live-stop") {
      await initialization;
      stopLive();
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      requestId: message.requestId,
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
