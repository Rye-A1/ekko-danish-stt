/* global ort */

function signedAsset(path) {
  const base = new URL(self.location.href);
  const url = new URL(path, base);
  const signature = base.searchParams.get("__sign");
  if (signature && url.origin === base.origin) url.searchParams.set("__sign", signature);
  return url.href;
}

importScripts(signedAsset("ort.wasm.min.js"));

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = {
  mjs: signedAsset("ort-wasm-simd-threaded.mjs"),
  wasm: signedAsset("ort-wasm-simd-threaded.wasm"),
};

let initialization = null;
let session = null;
let vocabulary = null;
let labels = null;
let maximumLength = 128;
const INFERENCE_WINDOW_TOKENS = 128;
let pendingRequest = null;
let processing = false;

async function sha256(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
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
        type: "pnc-progress",
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

async function initialize(configUrl) {
  if (session) return;
  const response = await fetch(signedAsset(configUrl), {cache: "no-cache"});
  if (!response.ok) throw new Error("Kunne ikke hente PnC-konfigurationen");
  const rootConfig = await response.json();
  const config = rootConfig.pnc;
  if (!config?.files?.length) throw new Error("PnC mangler i modelkonfigurationen");

  const totalBytes = config.files.reduce((sum, file) => sum + file.bytes, 0);
  let completedBytes = 0;
  const downloaded = new Map();
  for (const file of config.files) {
    const bytes = await fetchVerified(file, completedBytes, totalBytes);
    downloaded.set(file.path, bytes);
    completedBytes += bytes.byteLength;
  }

  const tokenizer = JSON.parse(new TextDecoder().decode(downloaded.get("tokenizer.json")));
  const modelConfig = JSON.parse(new TextDecoder().decode(downloaded.get("config.json")));
  vocabulary = tokenizer.model.vocab;
  labels = modelConfig.id2label;
  // Shorter context restores sentence stops in long, continuous dictation.
  maximumLength = Math.min(config.maxLength || 128, INFERENCE_WINDOW_TOKENS);
  session = await ort.InferenceSession.create(downloaded.get("pnc.int8.onnx"), {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  self.postMessage({type: "pnc-ready", bytes: totalBytes, model: config.name});
}

function splitBasicToken(word) {
  const parts = [];
  let current = "";
  const flush = () => {
    if (current) parts.push(current);
    current = "";
  };
  for (const character of Array.from(word)) {
    if (/\s/u.test(character)) {
      flush();
    } else if (/[\p{P}\p{S}]/u.test(character)) {
      flush();
      parts.push(character);
    } else if (!/[\p{Cc}\p{Cf}]/u.test(character)) {
      current += character;
    }
  }
  flush();
  return parts;
}

function wordPiece(token) {
  const characters = Array.from(token);
  if (!characters.length) return [];
  if (characters.length > 100) return [vocabulary["[UNK]"]];
  const pieces = [];
  let start = 0;
  while (start < characters.length) {
    let end = characters.length;
    let identifier;
    while (start < end) {
      const candidate = `${start ? "##" : ""}${characters.slice(start, end).join("")}`;
      if (Object.hasOwn(vocabulary, candidate)) {
        identifier = vocabulary[candidate];
        break;
      }
      end -= 1;
    }
    if (identifier === undefined) return [vocabulary["[UNK]"]];
    pieces.push(identifier);
    start = end;
  }
  return pieces;
}

function encodeWords(words) {
  const identifiers = [vocabulary["[CLS]"]];
  const wordIds = [null];
  words.forEach((word, wordIndex) => {
    const basicTokens = splitBasicToken(word);
    const pieces = basicTokens.length
      ? basicTokens.flatMap((token) => wordPiece(token))
      : [vocabulary["[UNK]"]];
    for (const identifier of pieces) {
      identifiers.push(identifier);
      wordIds.push(wordIndex);
    }
  });
  identifiers.push(vocabulary["[SEP]"]);
  wordIds.push(null);
  return {identifiers, wordIds};
}

function windowTokenIndex(words) {
  const {wordIds} = encodeWords(words);
  const counts = new Array(words.length).fill(0);
  let specialTokens = 0;
  for (const wordId of wordIds) {
    if (wordId === null) specialTokens += 1;
    else counts[wordId] += 1;
  }
  const prefix = [0];
  for (const count of counts) prefix.push(prefix.at(-1) + count);
  return {prefix, specialTokens};
}

function maximumWindowEnd(words, start, prefix, specialTokens) {
  const tokenBudget = prefix[start] + maximumLength - specialTokens;
  let low = start + 1;
  let high = Math.min(words.length, start + maximumLength);
  while (low < high) {
    const middle = Math.floor((low + high + 1) / 2);
    if (prefix[middle] <= tokenBudget) {
      low = middle;
    } else {
      high = middle - 1;
    }
  }
  return low;
}

function capitalize(word) {
  const characters = Array.from(word);
  if (!characters.length) return word;
  return characters[0].toLocaleUpperCase("da-DK")
    + characters.slice(1).join("").toLocaleLowerCase("da-DK");
}

function capitalizeInitial(word) {
  const characters = Array.from(word);
  const index = characters.findIndex((character) => /\p{L}/u.test(character));
  if (index < 0) return word;
  characters[index] = characters[index].toLocaleUpperCase("da-DK");
  return characters.join("");
}

function stabilizeRenderedWords(words) {
  if (!words.length) return [];
  const output = [...words];
  output[0] = capitalizeInitial(output[0]);
  for (let index = 0; index < output.length - 1; index += 1) {
    if (!/[.!?]+$/u.test(output[index])) continue;
    if (/\p{L}\.\p{L}/u.test(output[index])) continue; // f.eks.
    if (/\p{Lu}/u.test(output[index + 1])) continue; // CPR, iPhone
    output[index + 1] = capitalizeInitial(output[index + 1]);
  }
  return output;
}

async function predictWindow(words) {
  const {identifiers, wordIds} = encodeWords(words);
  const shape = [1, identifiers.length];
  const inputIds = BigInt64Array.from(identifiers, (value) => BigInt(value));
  const attentionMask = new BigInt64Array(identifiers.length).fill(1n);
  const output = await session.run({
    input_ids: new ort.Tensor("int64", inputIds, shape),
    attention_mask: new ort.Tensor("int64", attentionMask, shape),
  });
  const logits = output.logits;
  const labelCount = logits.dims.at(-1);
  const rendered = [...words];
  const seen = new Set();
  wordIds.forEach((wordIndex, tokenIndex) => {
    if (wordIndex === null || seen.has(wordIndex)) return;
    seen.add(wordIndex);
    let prediction = 0;
    let best = -Infinity;
    for (let labelIndex = 0; labelIndex < labelCount; labelIndex += 1) {
      const value = logits.data[tokenIndex * labelCount + labelIndex];
      if (value > best) {
        best = value;
        prediction = labelIndex;
      }
    }
    const label = labels[String(prediction)];
    const upper = label.endsWith("|U");
    const mark = upper ? label.slice(0, -2) : label;
    rendered[wordIndex] = `${upper ? capitalize(words[wordIndex]) : words[wordIndex]}${mark === "O" ? "" : mark}`;
  });
  return rendered;
}

async function punctuateWords(words) {
  if (!words.length) return [];
  const {prefix, specialTokens} = windowTokenIndex(words);
  const output = [];
  let nextWord = 0;
  const overlapWords = 12;
  while (nextWord < words.length) {
    let contextStart = Math.max(0, nextWord - overlapWords);
    let windowEnd = maximumWindowEnd(words, contextStart, prefix, specialTokens);
    if (windowEnd <= nextWord) {
      contextStart = nextWord;
      windowEnd = maximumWindowEnd(words, contextStart, prefix, specialTokens);
    }
    const rendered = await predictWindow(words.slice(contextStart, windowEnd));
    const emitEnd = windowEnd === words.length
      ? windowEnd
      : Math.max(nextWord + 1, windowEnd - overlapWords);
    output.push(...rendered.slice(nextWord - contextStart, emitEnd - contextStart));
    nextWord = emitEnd;
  }
  if (output.length !== words.length) throw new Error("PnC ændrede antallet af ord");
  return output;
}

async function punctuateSegments(segments, contextWords = [], capitalizeFirst = false) {
  const counts = [];
  const sourceWords = [...contextWords];
  segments.forEach((segment) => {
    const words = String(segment.text || "").trim().split(/\s+/).filter(Boolean);
    counts.push(words.length);
    sourceWords.push(...words);
  });
  const renderedWords = stabilizeRenderedWords(await punctuateWords(sourceWords));
  if (capitalizeFirst && renderedWords.length > contextWords.length) {
    renderedWords[contextWords.length] = capitalizeInitial(renderedWords[contextWords.length]);
  }
  let offset = contextWords.length;
  return segments.map((segment, segmentIndex) => {
    const start = offset;
    const rendered = renderedWords.slice(start, start + counts[segmentIndex]);
    offset += counts[segmentIndex];
    const timedWords = Array.isArray(segment.words) && segment.words.length === rendered.length
      && segment.words.every((word, index) => word.text === sourceWords[start + index]);
    const words = timedWords
      ? segment.words.map((word, wordIndex) => ({...word, text: rendered[wordIndex]}))
      : [];
    return {
      ...segment,
      rawText: segment.rawText || segment.text,
      text: rendered.join(" "),
      words,
    };
  });
}

async function drainRequests() {
  if (processing) return;
  processing = true;
  try {
    while (pendingRequest) {
      const request = pendingRequest;
      pendingRequest = null;
      await initialization;
      const segments = await punctuateSegments(
        request.segments, request.contextWords, request.capitalizeFirst,
      );
      self.postMessage({type: "pnc-result", requestId: request.requestId, segments});
    }
  } catch (error) {
    self.postMessage({
      type: "pnc-error",
      message: error instanceof Error ? error.message : String(error),
    });
  } finally {
    processing = false;
  }
}

self.onmessage = (event) => {
  const message = event.data || {};
  if (message.type === "init") {
    initialization ||= initialize(message.configUrl || "model-config.json");
    initialization.catch((error) => self.postMessage({
      type: "pnc-error",
      message: error instanceof Error ? error.message : String(error),
    }));
  } else if (message.type === "punctuate") {
    initialization ||= initialize(message.configUrl || "model-config.json");
    pendingRequest = message;
    drainRequests();
  }
};
