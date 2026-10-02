const elements = {
  app: document.querySelector("#app"),
  ambient: document.querySelector("#ambient"),
  orb: document.querySelector("#orb"),
  ring: document.querySelector(".ring-1"),
  statusDot: document.querySelector("#statusDot"),
  statusText: document.querySelector("#statusText"),
  statusDetail: document.querySelector("#statusDetail"),
  progressTrack: document.querySelector("#progressTrack"),
  progressBar: document.querySelector("#progressBar"),
  recordButton: document.querySelector("#recordButton"),
  recordLabel: document.querySelector("#recordLabel"),
  emptyTranscript: document.querySelector("#emptyTranscript"),
  transcriptPanel: document.querySelector("#transcriptPanel"),
  segments: document.querySelector("#segments"),
  pncStatus: document.querySelector("#pncStatus"),
  themeButton: document.querySelector("#themeButton"),
  copyButton: document.querySelector("#copyButton"),
  downloadButton: document.querySelector("#downloadButton"),
  clearButton: document.querySelector("#clearButton"),
  infoButton: document.querySelector("#infoButton"),
  settingsButton: document.querySelector("#settingsButton"),
  historyList: document.querySelector("#historyList"),
  historyEmpty: document.querySelector("#historyEmpty"),
  clearHistoryButton: document.querySelector("#clearHistoryButton"),
  microphoneDialog: document.querySelector("#microphoneDialog"),
  microphoneTitle: document.querySelector("#microphoneTitle"),
  microphoneDescription: document.querySelector("#microphoneDescription"),
  microphoneHelp: document.querySelector("#microphoneHelp"),
  microphoneCloseButton: document.querySelector("#microphoneCloseButton"),
  microphoneCancelButton: document.querySelector("#microphoneCancelButton"),
  microphoneContinueButton: document.querySelector("#microphoneContinueButton"),
  infoDialog: document.querySelector("#infoDialog"),
  settingsDialog: document.querySelector("#settingsDialog"),
};

const HISTORY_STORAGE_KEY = "ekko-tiny-browser:conversations:v1";
const THEME_STORAGE_KEY = "ekko-tiny-browser:theme:v1";
const HISTORY_LIMIT = 20;

function signedAsset(path) {
  const url = new URL(path, location.href);
  const signature = new URLSearchParams(location.search).get("__sign");
  if (signature && url.origin === location.origin) url.searchParams.set("__sign", signature);
  return url.href;
}

function setTheme(theme) {
  const dark = theme === "dark";
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  elements.themeButton.setAttribute("aria-pressed", String(dark));
  const label = dark ? "Slå lys tilstand til" : "Slå mørk tilstand til";
  elements.themeButton.setAttribute("aria-label", label);
  elements.themeButton.title = label;
  document.querySelector('meta[name="theme-color"]').content = dark ? "#10151c" : "#f3f4f0";
}

let savedTheme;
try { savedTheme = localStorage.getItem(THEME_STORAGE_KEY); } catch { /* Storage may be disabled. */ }
setTheme(savedTheme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));

const state = {
  ready: false,
  busy: false,
  recording: false,
  stopping: false,
  checkingMicrophonePermission: false,
  startingRecording: false,
  microphonePermissionGranted: false,
  speechDetected: false,
  pncReady: false,
  pncFailed: false,
  pncInitializationRequested: false,
  pncRequestId: 0,
  pncPending: null,
  pncFormattedCount: 0,
  pncCommittedWords: 0,
  nextSegmentId: 0,
  rawSegments: [],
  segments: [],
  previewSegment: null,
  history: [],
  currentConversationId: null,
  currentConversationCreatedAt: null,
  punctuationApplied: false,
  microphone: null,
  audioContext: null,
  sourceNode: null,
  captureNode: null,
  silentGain: null,
  resampler: null,
  captureStopResolve: null,
  captureDrainTimedOut: false,
  level: 0,
};

const worker = new Worker(signedAsset("worker.js?v=20260927-2"));
const pncWorker = new Worker(signedAsset("pnc-worker.js?v=20261001-2"));

function setStatus(text, detail = "", kind = "loading") {
  elements.statusText.textContent = text;
  elements.statusDetail.textContent = detail;
  elements.statusDot.classList.toggle("ready", kind === "ready");
  elements.statusDot.classList.toggle("error", kind === "error");
}

function setProgress(value, complete = false) {
  elements.progressBar.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
  elements.progressTrack.classList.toggle("complete", complete);
}

function formatBytes(value) {
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KiB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MiB`;
}

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(1).padStart(4, "0")}`;
}

function transcriptText() {
  return state.segments.map((segment) => segment.text).filter(Boolean).join(" ").trim();
}

function copySegments(segments) {
  if (!Array.isArray(segments)) return [];
  return segments.map((segment, index) => ({
    id: typeof segment.id === "string" ? segment.id : `segment-${index}`,
    text: String(segment.text || ""),
    rawText: String(segment.rawText || segment.text || ""),
    start: Number.isFinite(segment.start) ? segment.start : 0,
    end: Number.isFinite(segment.end) ? segment.end : 0,
    duration: Number.isFinite(segment.duration) ? segment.duration : 0,
    words: Array.isArray(segment.words)
      ? segment.words.map((word) => ({
        text: String(word.text || ""),
        start: Number.isFinite(word.start) ? word.start : 0,
        end: Number.isFinite(word.end) ? word.end : 0,
      }))
      : [],
  }));
}

function readConversationHistory() {
  try {
    const records = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || "[]");
    if (!Array.isArray(records)) return [];
    return records
      .filter((record) => record && typeof record.id === "string" && Array.isArray(record.segments))
      .slice(0, HISTORY_LIMIT)
      .map((record) => ({
        id: record.id,
        createdAt: String(record.createdAt || new Date().toISOString()),
        updatedAt: String(record.updatedAt || record.createdAt || new Date().toISOString()),
        text: String(record.text || ""),
        rawSegments: copySegments(record.rawSegments || record.segments),
        segments: copySegments(record.segments),
        punctuationApplied: Boolean(record.punctuationApplied),
        formattedCount: Number.isInteger(record.formattedCount) ? record.formattedCount : null,
        committedWords: Number.isInteger(record.committedWords) ? record.committedWords : null,
      }));
  } catch (error) {
    console.warn("Kunne ikke læse lokal samtalehistorik", error);
    return [];
  }
}

function formatHistoryDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Ukendt tidspunkt";
  return new Intl.DateTimeFormat("da-DK", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function historyDuration(record) {
  return record.segments.reduce((maximum, segment) => Math.max(maximum, segment.end || 0), 0);
}

function renderHistory() {
  const hasHistory = state.history.length > 0;
  const historyLocked = state.recording || state.stopping;
  elements.historyList.replaceChildren();
  elements.historyList.hidden = !hasHistory;
  elements.historyEmpty.hidden = hasHistory;
  elements.clearHistoryButton.disabled = !hasHistory || historyLocked;

  for (const record of state.history) {
    const item = document.createElement("li");
    item.className = "history-item";

    const openButton = document.createElement("button");
    openButton.type = "button";
    openButton.className = "history-open";
    openButton.dataset.historyOpen = record.id;
    openButton.disabled = historyLocked;

    const date = document.createElement("time");
    date.className = "history-date";
    date.dateTime = record.createdAt;
    date.textContent = formatHistoryDate(record.createdAt);

    const preview = document.createElement("span");
    preview.className = "history-preview";
    preview.textContent = record.text;

    const meta = document.createElement("span");
    meta.className = "history-meta";
    const wordCount = record.text.split(/\s+/).filter(Boolean).length;
    meta.textContent = `${wordCount} ord · ${formatTime(historyDuration(record))}`;
    openButton.append(date, preview, meta);

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "history-delete";
    deleteButton.dataset.historyDelete = record.id;
    deleteButton.disabled = historyLocked;
    deleteButton.setAttribute("aria-label", `Slet samtale fra ${date.textContent}`);
    deleteButton.title = "Slet samtale";
    deleteButton.textContent = "×";
    item.append(openButton, deleteButton);
    elements.historyList.appendChild(item);
  }
}

function writeConversationHistory() {
  let records = state.history.slice(0, HISTORY_LIMIT);
  while (records.length) {
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(records));
      state.history = records;
      renderHistory();
      return;
    } catch (error) {
      records = records.slice(0, -1);
      if (!records.length) console.warn("Kunne ikke gemme lokal samtalehistorik", error);
    }
  }
  state.history = [];
  try {
    localStorage.removeItem(HISTORY_STORAGE_KEY);
  } catch (error) {
    console.warn("Kunne ikke rydde lokal samtalehistorik", error);
  }
  renderHistory();
}

function persistCurrentConversation() {
  const finalizedSegments = state.segments.filter((segment) => segment.id !== "live-preview");
  const text = finalizedSegments.map((segment) => segment.text).filter(Boolean).join(" ").trim();
  if (!state.currentConversationId || !text) return;
  const timestamp = new Date().toISOString();
  const record = {
    id: state.currentConversationId,
    createdAt: state.currentConversationCreatedAt || timestamp,
    updatedAt: timestamp,
    text,
    rawSegments: copySegments(state.rawSegments),
    segments: copySegments(finalizedSegments),
    punctuationApplied: state.punctuationApplied,
    formattedCount: state.pncFormattedCount,
    committedWords: state.pncCommittedWords,
  };
  state.history = [record, ...state.history.filter((item) => item.id !== record.id)];
  writeConversationHistory();
}

function loadConversation(recordId) {
  const record = state.history.find((item) => item.id === recordId);
  if (!record || state.recording || state.stopping) return;
  state.pncRequestId += 1;
  state.pncPending = null;
  state.currentConversationId = record.id;
  state.currentConversationCreatedAt = record.createdAt;
  state.rawSegments = copySegments(record.rawSegments);
  state.pncFormattedCount = record.formattedCount === null
    ? (record.punctuationApplied ? state.rawSegments.length : 0)
    : Math.max(0, Math.min(record.formattedCount, state.rawSegments.length));
  state.punctuationApplied = state.pncFormattedCount === state.rawSegments.length;
  state.segments = copySegments(state.pncFormattedCount ? record.segments : record.rawSegments)
    .filter((segment) => segment.id !== "live-preview");
  state.pncCommittedWords = record.committedWords === null
    ? stableSentenceBoundary(state.segments, state.pncFormattedCount)
    : Math.min(record.committedWords, countWords(state.rawSegments));
  state.nextSegmentId = state.segments.length;
  elements.app.classList.add("session-active");
  document.body.classList.add("session-active");
  setConversationVisible(true);
  renderTranscript();
  elements.recordLabel.textContent = "Tryk for at optage igen";
  setStatus("Tidligere samtale", formatHistoryDate(record.createdAt), "ready");
  elements.settingsDialog.close();
  if (!record.punctuationApplied) requestPunctuation();
}

function deleteConversation(recordId) {
  if (state.recording || state.stopping) return;
  state.history = state.history.filter((item) => item.id !== recordId);
  if (state.currentConversationId === recordId) state.currentConversationId = null;
  writeConversationHistory();
}

function setConversationVisible(visible) {
  elements.app.classList.toggle("conversation-visible", visible);
  elements.transcriptPanel.classList.toggle("open", visible);
  document.body.classList.toggle("conversation-visible", visible);
  if (visible) {
    requestAnimationFrame(() => {
      elements.transcriptPanel.scrollTop = elements.transcriptPanel.scrollHeight;
    });
  }
}

function renderWords(paragraph, segment, animate) {
  paragraph.replaceChildren();
  const words = String(segment.text || "").split(/\s+/).filter(Boolean);
  words.forEach((word, index) => {
    if (index) paragraph.appendChild(document.createTextNode(" "));
    const span = document.createElement("span");
    span.className = `word${animate ? " reveal-word" : ""}`;
    span.textContent = word;
    if (animate) span.style.animationDelay = `${Math.min(index * 45, 180)}ms`;
    paragraph.appendChild(span);
  });
}

function updateWords(paragraph, segment, animate) {
  const words = String(segment.text || "").split(/\s+/).filter(Boolean);
  const spans = Array.from(paragraph.querySelectorAll(".word"));
  if (!animate && spans.length === words.length) {
    spans.forEach((span, index) => {
      span.textContent = words[index];
    });
    return;
  }
  renderWords(paragraph, segment, animate);
}

function renderTranscript({animateNew = false} = {}) {
  const hasText = state.segments.some((segment) => segment.text);
  elements.emptyTranscript.hidden = hasText;
  elements.segments.hidden = !hasText;
  const existing = new Map(
    Array.from(elements.segments.children, (item) => [item.dataset.segmentId, item]),
  );
  for (const segment of state.segments) {
    if (!segment.text) continue;
    let item = existing.get(segment.id);
    const isNew = !item;
    if (isNew) {
      item = document.createElement("li");
      item.className = "segment message user";
      item.dataset.segmentId = segment.id;
      const paragraph = document.createElement("p");
      paragraph.className = "segment-text message-bubble";
      item.append(paragraph);
      elements.segments.appendChild(item);
    }
    item.classList.toggle("preview", segment.id === "live-preview");
    existing.delete(segment.id);
    const paragraph = item.querySelector(".segment-text");
    if (paragraph.dataset.text !== segment.text) {
      updateWords(paragraph, segment, animateNew && isNew);
      paragraph.dataset.text = segment.text;
    }
  }
  for (const item of existing.values()) item.remove();

  elements.copyButton.disabled = !hasText;
  elements.downloadButton.disabled = !hasText;
  elements.clearButton.disabled = !hasText;
  if (hasText) {
    requestAnimationFrame(() => {
      elements.transcriptPanel.scrollTop = elements.transcriptPanel.scrollHeight;
    });
  }
}

function clearTranscript() {
  state.pncRequestId += 1;
  state.pncPending = null;
  state.pncFormattedCount = 0;
  state.pncCommittedWords = 0;
  state.nextSegmentId = 0;
  state.rawSegments = [];
  state.segments = [];
  state.previewSegment = null;
  state.punctuationApplied = false;
  state.currentConversationId = null;
  state.currentConversationCreatedAt = null;
  elements.segments.replaceChildren();
  renderTranscript();
  setConversationVisible(false);
  if (!state.recording) {
    elements.app.classList.remove("session-active");
    document.body.classList.remove("session-active");
  }
}

function ensurePunctuation() {
  if (state.pncInitializationRequested) return;
  state.pncInitializationRequested = true;
  pncWorker.postMessage({type: "init", configUrl: signedAsset("model-config.json")});
}

function wordsOf(segment) {
  return String(segment?.text || "").trim().split(/\s+/).filter(Boolean);
}

function countWords(segments) {
  return segments.reduce((count, segment) => count + wordsOf(segment).length, 0);
}

function stableSentenceBoundary(segments, formattedCount) {
  const words = segments.slice(0, formattedCount).flatMap(wordsOf);
  // The final word has no right context yet; its sentence stop may change.
  for (let index = words.length - 2; index >= 0; index -= 1) {
    if (/\p{L}\.\p{L}/u.test(words[index])) continue; // f.eks.
    if (/[.!?]["”’']*$/u.test(words[index])) return index + 1;
  }
  return 0;
}

function requestPunctuation() {
  if (state.pncFailed || state.pncPending || state.pncFormattedCount >= state.rawSegments.length) return;
  ensurePunctuation();
  state.pncRequestId += 1;
  const rawWords = state.rawSegments.flatMap(wordsOf);
  const startWord = Math.min(state.pncCommittedWords, rawWords.length);
  const displayedWords = state.segments.filter((segment) => segment.id !== "live-preview").flatMap(wordsOf);
  const capitalizeFirst = startWord > 0 && /[.!?]["”’']*$/u.test(displayedWords[startWord - 1] || "");
  const contextWords = rawWords.slice(Math.max(0, startWord - 12), startWord);
  const entries = [];
  const segments = [];
  let offset = 0;
  state.rawSegments.forEach((segment, index) => {
    const words = wordsOf(segment);
    const skip = Math.max(0, Math.min(words.length, startWord - offset));
    offset += words.length;
    if (skip === words.length) return;
    const suffix = words.slice(skip);
    entries.push({id: segment.id, index, skip, count: suffix.length});
    segments.push({
      ...segment,
      text: suffix.join(" "),
      words: Array.isArray(segment.words) && segment.words.length === words.length
        ? segment.words.slice(skip) : [],
    });
  });
  state.pncPending = {
    requestId: state.pncRequestId,
    entries,
    endSegmentCount: state.rawSegments.length,
  };
  pncWorker.postMessage({
    type: "punctuate",
    requestId: state.pncRequestId,
    configUrl: signedAsset("model-config.json"),
    contextWords,
    capitalizeFirst,
    segments,
  });
}

function applyPunctuationResult(message) {
  const pending = state.pncPending;
  if (!pending || message.requestId !== pending.requestId) return;
  state.pncPending = null;
  const displayed = new Map(state.segments.map((segment) => [segment.id, segment]));
  if (!Array.isArray(message.segments)
    || message.segments.length !== pending.entries.length
    || message.segments.some((segment, index) => {
      const entry = pending.entries[index];
      return segment.id !== entry.id
        || state.rawSegments[entry.index]?.id !== entry.id
        || wordsOf(displayed.get(entry.id)).length < entry.skip
        || wordsOf(segment).length !== entry.count;
    })) {
    state.pncFailed = true;
    elements.pncStatus.textContent = "Tegnsætning utilgængelig";
    console.error("PnC returnerede segmenter i uventet rækkefølge");
    return;
  }
  const formatted = new Map(message.segments.map((segment, index) => {
    const entry = pending.entries[index];
    const raw = state.rawSegments[entry.index];
    const previous = displayed.get(entry.id);
    const prefix = wordsOf(previous).slice(0, entry.skip);
    const words = previous?.words?.length === wordsOf(raw).length
      && segment.words?.length === entry.count
      ? [...previous.words.slice(0, entry.skip), ...segment.words]
      : [];
    return [entry.id, {
      ...raw,
      text: [...prefix, ...wordsOf(segment)].join(" "),
      words,
    }];
  }));
  state.segments = state.rawSegments.map((segment) =>
    formatted.get(segment.id) || displayed.get(segment.id) || segment);
  if (state.previewSegment) state.segments.push(state.previewSegment);
  state.pncFormattedCount = pending.endSegmentCount;
  state.pncCommittedWords = Math.max(
    state.pncCommittedWords,
    stableSentenceBoundary(state.segments, state.pncFormattedCount),
  );
  state.punctuationApplied = state.pncFormattedCount === state.rawSegments.length;
  renderTranscript();
  persistCurrentConversation();
  requestPunctuation();
}

class StreamingLinearResampler {
  constructor(sourceRate, targetRate) {
    this.sourceRate = sourceRate;
    this.targetRate = targetRate;
    this.ratio = sourceRate / targetRate;
    this.tail = new Float32Array(0);
    this.position = 0;
  }

  process(samples) {
    if (this.sourceRate === this.targetRate) return new Float32Array(samples);
    const input = new Float32Array(this.tail.length + samples.length);
    input.set(this.tail);
    input.set(samples, this.tail.length);
    const output = [];
    while (this.position < input.length - 1) {
      const left = Math.floor(this.position);
      const fraction = this.position - left;
      output.push(input[left] * (1 - fraction) + input[left + 1] * fraction);
      this.position += this.ratio;
    }
    const consumed = Math.min(Math.floor(this.position), Math.max(0, input.length - 1));
    this.tail = input.slice(consumed);
    this.position -= consumed;
    return Float32Array.from(output);
  }

  reset() {
    this.tail = new Float32Array(0);
    this.position = 0;
  }
}

function appendResult(result, {animate = true} = {}) {
  state.punctuationApplied = false;
  state.previewSegment = null;
  state.segments = state.segments.filter((segment) => segment.id !== "live-preview");
  const segment = {
    ...result,
    id: `segment-${state.nextSegmentId}`,
    rawText: result.text,
  };
  state.nextSegmentId += 1;
  state.rawSegments.push(segment);
  state.segments.push({...segment});
  setConversationVisible(true);
  renderTranscript({animateNew: animate});
}

async function startRecording() {
  if (!state.ready || state.recording || state.stopping || state.startingRecording) return;
  state.startingRecording = true;
  elements.recordButton.disabled = true;
  setStatus("Venter på mikrofonadgang", "Vælg Tillad i browserens vindue");
  try {
    state.microphone = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: true,
      },
    });
    state.microphonePermissionGranted = true;
    state.audioContext = new AudioContext({sampleRate: 16000});
    if (state.audioContext.state === "suspended") await state.audioContext.resume();
    await state.audioContext.audioWorklet.addModule(signedAsset("capture-worklet.js?v=20260926-4"));
    state.sourceNode = state.audioContext.createMediaStreamSource(state.microphone);
    state.captureNode = new AudioWorkletNode(state.audioContext, "ekko-capture");
    const captureSampleRate = state.audioContext.sampleRate;
    state.resampler = new StreamingLinearResampler(captureSampleRate, 16000);
    state.silentGain = state.audioContext.createGain();
    state.silentGain.gain.value = 0;
    state.captureNode.port.onmessage = (event) => {
      if (event.data?.type === "stopped") {
        state.captureStopResolve?.();
        return;
      }
      if (!state.recording && !state.stopping) return;
      const original = event.data;
      const samples = state.resampler.process(original);
      if (!samples.length) return;
      let energy = 0;
      for (const sample of samples) energy += sample * sample;
      state.level = Math.min(1, Math.sqrt(energy / Math.max(1, samples.length)) * 8);
      worker.postMessage({type: "live-chunk", samples: samples.buffer}, [samples.buffer]);
    };
    clearTranscript();
    state.captureDrainTimedOut = false;
    state.currentConversationId = typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    state.currentConversationCreatedAt = new Date().toISOString();
    elements.app.classList.add("session-active");
    document.body.classList.add("session-active");
    ensurePunctuation();
    worker.postMessage({type: "live-start"});
    state.recording = true;
    state.sourceNode.connect(state.captureNode);
    state.captureNode.connect(state.silentGain);
    state.silentGain.connect(state.audioContext.destination);
    elements.recordButton.classList.add("recording");
    elements.recordButton.setAttribute("aria-label", "Stop optagelse");
    elements.recordLabel.textContent = "Lytter";
    setStatus("Mikrofon aktiv", "Alt behandles lokalt", "ready");
  } catch (error) {
    state.recording = false;
    if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
      state.microphonePermissionGranted = false;
    }
    state.captureNode?.disconnect();
    state.sourceNode?.disconnect();
    state.silentGain?.disconnect();
    state.microphone?.getTracks().forEach((track) => track.stop());
    await state.audioContext?.close();
    state.captureNode = null;
    state.sourceNode = null;
    state.silentGain = null;
    state.microphone = null;
    state.audioContext = null;
    state.resampler = null;
    state.currentConversationId = null;
    state.currentConversationCreatedAt = null;
    elements.app.classList.remove("session-active");
    document.body.classList.remove("session-active");
    setStatus(
      "Mikrofonen kunne ikke startes",
      error.name === "NotAllowedError" || error.name === "PermissionDeniedError"
        ? "Tillad mikrofonen i browserens indstillinger for denne side, og prøv igen"
        : error.message,
      "error",
    );
  } finally {
    state.startingRecording = false;
    elements.recordButton.disabled = !state.ready || state.stopping;
  }
}

async function requestStartRecording() {
  if (!state.ready || state.recording || state.stopping || state.startingRecording
    || state.checkingMicrophonePermission || elements.microphoneDialog.open) return;
  state.checkingMicrophonePermission = true;
  try {
    let permissionState = null;
    try {
      permissionState = (await navigator.permissions?.query({name: "microphone"}))?.state || null;
    } catch (_) {
      // Some browsers do not expose microphone through the Permissions API.
    }
    if (permissionState === "granted" || (state.microphonePermissionGranted && permissionState !== "denied")) {
      startRecording();
      return;
    }
    const blocked = permissionState === "denied";
    elements.microphoneTitle.textContent = blocked
      ? "Mikrofonen er blokeret"
      : "Giv Ekko adgang til din mikrofon";
    elements.microphoneDescription.textContent = blocked
      ? "Åbn browserens indstillinger for denne side, og tillad adgang til mikrofonen. Du kan derefter prøve igen."
      : "Mikrofonen bruges kun, mens du optager. Lyd og tekst behandles på din enhed, og lyd gemmes ikke.";
    elements.microphoneHelp.textContent = blocked
      ? "Se efter mikrofonikonet eller sideindstillingerne ved adresselinjen."
      : "Når du fortsætter, beder browseren dig om at vælge Tillad.";
    elements.microphoneCancelButton.textContent = blocked ? "Luk" : "Ikke nu";
    elements.microphoneContinueButton.hidden = blocked;
    elements.microphoneDialog.showModal();
  } finally {
    state.checkingMicrophonePermission = false;
  }
}

async function stopRecording() {
  if (!state.recording || state.stopping) return;
  state.recording = false;
  state.stopping = true;
  elements.recordButton.disabled = true;
  elements.recordButton.classList.remove("recording");
  elements.recordButton.setAttribute("aria-label", "Begynd optagelse");
  elements.recordLabel.textContent = "Færdiggør sidste segment";
  if (state.captureNode) {
    const drained = await new Promise((resolve) => {
      const timeout = window.setTimeout(() => resolve(false), 1000);
      state.captureStopResolve = () => {
        window.clearTimeout(timeout);
        resolve(true);
      };
      state.captureNode.port.postMessage({type: "stop"});
    });
    state.captureDrainTimedOut = !drained;
    state.captureStopResolve = null;
  }
  state.captureNode?.disconnect();
  state.sourceNode?.disconnect();
  state.silentGain?.disconnect();
  state.microphone?.getTracks().forEach((track) => track.stop());
  await state.audioContext?.close();
  state.captureNode = null;
  state.sourceNode = null;
  state.silentGain = null;
  state.microphone = null;
  state.audioContext = null;
  state.resampler?.reset();
  state.resampler = null;
  state.level = 0;
  worker.postMessage({type: "live-stop"});
}

worker.onmessage = (event) => {
  const message = event.data || {};
  if (message.type === "model-progress") {
    const percent = Math.round(message.overall * 100);
    setStatus(message.cached ? "Indlæser gemt model" : "Indlæser lokal model", `${percent}% · ${message.file}`);
    setProgress(message.overall);
  } else if (message.type === "ready") {
    state.ready = true;
    elements.recordButton.disabled = false;
    elements.recordLabel.textContent = "Klik for at tale";
    setStatus("Klar", `${formatBytes(message.bytes)} verificeret`, "ready");
    setProgress(1, true);
    ensurePunctuation();
  } else if (message.type === "live-result") {
    appendResult(message.result);
    persistCurrentConversation();
    requestPunctuation();
  } else if (message.type === "live-preview") {
    state.previewSegment = {...message.result, id: "live-preview", rawText: message.result.text};
    state.segments = [
      ...state.segments.filter((segment) => segment.id !== "live-preview"),
      state.previewSegment,
    ];
    setConversationVisible(true);
    renderTranscript();
  } else if (message.type === "live-preview-clear") {
    state.previewSegment = null;
    state.segments = state.segments.filter((segment) => segment.id !== "live-preview");
    renderTranscript();
  } else if (message.type === "speech-state") {
    state.speechDetected = Boolean(message.detected);
    if (state.recording) elements.recordLabel.textContent = state.speechDetected ? "Tale registreret" : "Lytter";
  } else if (message.type === "live-stopped") {
    state.stopping = false;
    elements.recordButton.disabled = false;
    persistCurrentConversation();
    elements.recordLabel.textContent = state.segments.length ? "Tryk for at optage igen" : "Klik for at tale";
    setStatus(
      state.captureDrainTimedOut ? "Optagelsen kan mangle afslutningen" : "Klar",
      state.captureDrainTimedOut
        ? "Lydkøen blev ikke bekræftet ved stop"
        : state.segments.length ? "Optagelse færdig" : "Ingen tale fundet",
      state.captureDrainTimedOut ? "error" : "ready",
    );
    if (!state.segments.length) {
      elements.app.classList.remove("session-active");
      document.body.classList.remove("session-active");
    }
  } else if (message.type === "error") {
    state.busy = false;
    elements.recordButton.disabled = !state.ready;
    setStatus("Noget gik galt", message.message, "error");
  }
};

pncWorker.onmessage = (event) => {
  const message = event.data || {};
  if (message.type === "pnc-progress") {
    elements.pncStatus.textContent = `${message.cached ? "Indlæser gemt model" : "Indlæser lokalt"} · ${Math.round(message.overall * 100)}%`;
  } else if (message.type === "pnc-ready") {
    state.pncReady = true;
    elements.pncStatus.textContent = `Klar · ${formatBytes(message.bytes)}`;
    requestPunctuation();
  } else if (message.type === "pnc-result") {
    applyPunctuationResult(message);
  } else if (message.type === "pnc-error") {
    state.pncFailed = true;
    elements.pncStatus.textContent = "Tegnsætning utilgængelig";
    console.error(message.message);
  }
};

worker.onerror = (event) => {
  console.error(event.message);
  setStatus("WASM-runtime kunne ikke startes", "Genindlæs siden", "error");
};

elements.recordButton.addEventListener("click", () => state.recording ? stopRecording() : requestStartRecording());
elements.microphoneCloseButton.addEventListener("click", () => elements.microphoneDialog.close());
elements.microphoneCancelButton.addEventListener("click", () => elements.microphoneDialog.close());
elements.microphoneContinueButton.addEventListener("click", () => {
  elements.microphoneDialog.close();
  startRecording();
});
elements.clearButton.addEventListener("click", clearTranscript);
elements.historyList.addEventListener("click", (event) => {
  const openButton = event.target.closest("[data-history-open]");
  if (openButton) {
    loadConversation(openButton.dataset.historyOpen);
    return;
  }
  const deleteButton = event.target.closest("[data-history-delete]");
  if (deleteButton) deleteConversation(deleteButton.dataset.historyDelete);
});
elements.clearHistoryButton.addEventListener("click", () => {
  if (state.recording || state.stopping) return;
  if (!window.confirm("Slet alle gemte samtaler fra denne browser?")) return;
  state.history = [];
  state.currentConversationId = null;
  writeConversationHistory();
});
elements.copyButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(transcriptText());
  elements.copyButton.textContent = "✓";
  window.setTimeout(() => { elements.copyButton.textContent = "⧉"; }, 1200);
});
elements.downloadButton.addEventListener("click", () => {
  const payload = {
    schema: "ekko-browser-transcript-v1",
    model: "Ekko v1 Tiny int8",
    generatedAt: new Date().toISOString(),
    punctuation: state.punctuationApplied ? "Ekko PnC v2 int8" : "pending_or_unavailable",
    text: transcriptText(),
    segments: state.segments,
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type: "application/json"}));
  const link = document.createElement("a");
  link.href = url;
  link.download = "ekko-transcript.json";
  link.click();
  URL.revokeObjectURL(url);
});

elements.themeButton.addEventListener("click", () => {
  const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  setTheme(theme);
  try { localStorage.setItem(THEME_STORAGE_KEY, theme); } catch { /* Storage may be disabled. */ }
});

elements.infoButton.addEventListener("click", () => elements.infoDialog.showModal());
elements.settingsButton.addEventListener("click", () => {
  renderHistory();
  elements.settingsDialog.showModal();
});

const ORB_VERTEX_SHADER = `#version 300 es
in vec2 position;
void main(){gl_Position=vec4(position,0.0,1.0);}`;

const ORB_FRAGMENT_SHADER = `#version 300 es
precision highp float;
uniform vec2 iResolution;
uniform float iTime;
uniform float uTimeSpeed;
uniform float uColorBalance;
uniform float uWarpStrength;
uniform float uWarpFrequency;
uniform float uWarpSpeed;
uniform float uWarpAmplitude;
uniform float uBlendAngle;
uniform float uBlendSoftness;
uniform float uRotationAmount;
uniform float uNoiseScale;
uniform float uGrainAmount;
uniform float uGrainScale;
uniform float uGrainAnimated;
uniform float uContrast;
uniform float uGamma;
uniform float uSaturation;
uniform vec2 uCenterOffset;
uniform float uZoom;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;
out vec4 fragColor;
#define S(a,b,t) smoothstep(a,b,t)
mat2 Rot(float a){float s=sin(a),c=cos(a);return mat2(c,-s,s,c);}
vec2 hash(vec2 p){p=vec2(dot(p,vec2(2127.1,81.17)),dot(p,vec2(1269.5,283.37)));return fract(sin(p)*43758.5453);}
float noise(vec2 p){vec2 i=floor(p),f=fract(p),u=f*f*(3.0-2.0*f);float n=mix(mix(dot(-1.0+2.0*hash(i+vec2(0.0,0.0)),f-vec2(0.0,0.0)),dot(-1.0+2.0*hash(i+vec2(1.0,0.0)),f-vec2(1.0,0.0)),u.x),mix(dot(-1.0+2.0*hash(i+vec2(0.0,1.0)),f-vec2(0.0,1.0)),dot(-1.0+2.0*hash(i+vec2(1.0,1.0)),f-vec2(1.0,1.0)),u.x),u.y);return 0.5+0.5*n;}
void mainImage(out vec4 o,vec2 C){
  float t=iTime*uTimeSpeed;
  vec2 uv=C/iResolution.xy;
  float ratio=iResolution.x/iResolution.y;
  vec2 tuv=uv-0.5+uCenterOffset;
  tuv/=max(uZoom,0.001);
  float degree=noise(vec2(t*0.1,tuv.x*tuv.y)*uNoiseScale);
  tuv.y*=1.0/ratio;
  tuv*=Rot(radians((degree-0.5)*uRotationAmount+180.0));
  tuv.y*=ratio;
  float frequency=uWarpFrequency;
  float ws=max(uWarpStrength,0.001);
  float amplitude=uWarpAmplitude/ws;
  float warpTime=t*uWarpSpeed;
  tuv.x+=sin(tuv.y*frequency+warpTime)/amplitude;
  tuv.y+=sin(tuv.x*(frequency*1.5)+warpTime)/(amplitude*0.5);
  vec3 colLav=uColor1;vec3 colOrg=uColor2;vec3 colDark=uColor3;
  float b=uColorBalance;float ss=max(uBlendSoftness,0.0);
  mat2 blendRot=Rot(radians(uBlendAngle));
  float blendX=(tuv*blendRot).x;
  float edge0=-0.3-b-ss;float edge1=0.2-b+ss;float v0=0.5-b+ss;float v1=-0.3-b-ss;
  vec3 layer1=mix(colDark,colOrg,S(edge0,edge1,blendX));
  vec3 layer2=mix(colOrg,colLav,S(edge0,edge1,blendX));
  vec3 col=mix(layer1,layer2,S(v0,v1,tuv.y));
  vec2 grainUv=uv*max(uGrainScale,0.001);
  if(uGrainAnimated>0.5){grainUv+=vec2(iTime*0.05);}
  float grain=fract(sin(dot(grainUv,vec2(12.9898,78.233)))*43758.5453);
  col+=(grain-0.5)*uGrainAmount;
  col=(col-0.5)*uContrast+0.5;
  float luma=dot(col,vec3(0.2126,0.7152,0.0722));
  col=mix(vec3(luma),col,uSaturation);
  col=pow(max(col,0.0),vec3(1.0/max(uGamma,0.001)));
  col=clamp(col,0.0,1.0);
  o=vec4(col,1.0);
}
void main(){vec4 o=vec4(0.0);mainImage(o,gl_FragCoord.xy);fragColor=o;}`;

function hexRgb(hex) {
  const match = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return match
    ? new Float32Array([
      parseInt(match[1], 16) / 255,
      parseInt(match[2], 16) / 255,
      parseInt(match[3], 16) / 255,
    ])
    : new Float32Array([1, 1, 1]);
}

function initGrainient() {
  const canvas = document.createElement("canvas");
  elements.orb.appendChild(canvas);
  const gl = canvas.getContext("webgl2", {alpha: false, antialias: false});
  if (!gl) return;
  const makeShader = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    return shader;
  };
  const program = gl.createProgram();
  gl.attachShader(program, makeShader(gl.VERTEX_SHADER, ORB_VERTEX_SHADER));
  gl.attachShader(program, makeShader(gl.FRAGMENT_SHADER, ORB_FRAGMENT_SHADER));
  gl.linkProgram(program);
  gl.useProgram(program);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  const uniform = (name) => gl.getUniformLocation(program, name);
  const uniforms = {
    time: uniform("iTime"), resolution: uniform("iResolution"), timeSpeed: uniform("uTimeSpeed"),
    colorBalance: uniform("uColorBalance"), warpStrength: uniform("uWarpStrength"),
    warpFrequency: uniform("uWarpFrequency"), warpSpeed: uniform("uWarpSpeed"),
    warpAmplitude: uniform("uWarpAmplitude"), blendAngle: uniform("uBlendAngle"),
    blendSoftness: uniform("uBlendSoftness"), rotation: uniform("uRotationAmount"),
    noiseScale: uniform("uNoiseScale"), grainAmount: uniform("uGrainAmount"),
    grainScale: uniform("uGrainScale"), grainAnimated: uniform("uGrainAnimated"),
    contrast: uniform("uContrast"), gamma: uniform("uGamma"), saturation: uniform("uSaturation"),
    center: uniform("uCenterOffset"), zoom: uniform("uZoom"), color1: uniform("uColor1"),
    color2: uniform("uColor2"), color3: uniform("uColor3"),
  };
  gl.uniform1f(uniforms.colorBalance, 0);
  gl.uniform1f(uniforms.blendAngle, 0);
  gl.uniform1f(uniforms.blendSoftness, 0.05);
  gl.uniform1f(uniforms.noiseScale, 2);
  gl.uniform1f(uniforms.grainAmount, 0.045);
  gl.uniform1f(uniforms.grainScale, 2);
  gl.uniform1f(uniforms.grainAnimated, 0);
  gl.uniform1f(uniforms.gamma, 1);
  gl.uniform2f(uniforms.center, 0, 0);
  gl.uniform1f(uniforms.timeSpeed, 0.25);
  gl.uniform1f(uniforms.warpAmplitude, 16);
  gl.uniform1f(uniforms.warpSpeed, 5);
  gl.uniform1f(uniforms.warpStrength, 1);
  gl.uniform1f(uniforms.warpFrequency, 1);
  gl.uniform1f(uniforms.rotation, 100);
  gl.uniform1f(uniforms.zoom, 0.95);
  gl.uniform1f(uniforms.contrast, 1.5);
  gl.uniform1f(uniforms.saturation, 1);
  gl.uniform3fv(uniforms.color1, hexRgb("#27AE60"));
  gl.uniform3fv(uniforms.color2, hexRgb("#8CBFFF"));
  gl.uniform3fv(uniforms.color3, hexRgb("#FFAAAD"));

  const resize = () => {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const rectangle = elements.orb.getBoundingClientRect();
    canvas.width = Math.ceil(rectangle.width * ratio);
    canvas.height = Math.ceil(rectangle.height * ratio);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.uniform2f(uniforms.resolution, canvas.width, canvas.height);
  };
  new ResizeObserver(resize).observe(elements.orb);
  resize();

  const startedAt = performance.now();
  let animationFrame = 0;
  function render(time) {
    animationFrame = requestAnimationFrame(render);
    gl.uniform1f(uniforms.time, (time - startedAt) * 0.001);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  render(performance.now());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) cancelAnimationFrame(animationFrame);
    else render(performance.now());
  });
}

function resizeCanvas(canvas) {
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const rectangle = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rectangle.width * ratio));
  canvas.height = Math.max(1, Math.round(rectangle.height * ratio));
  return ratio;
}

let ambientRatio = resizeCanvas(elements.ambient);
window.addEventListener("resize", () => { ambientRatio = resizeCanvas(elements.ambient); });

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let smoothedLevel = 0;
function drawVisuals(timestamp) {
  const time = reduceMotion ? 0 : timestamp / 1000;
  const target = state.recording ? state.level : 0;
  smoothedLevel += (target - smoothedLevel) * (target > smoothedLevel ? 0.3 : 0.075);
  if (smoothedLevel < 0.006) {
    elements.orb.style.scale = "";
    elements.ring.style.scale = "";
  } else {
    elements.orb.style.scale = String(1 + smoothedLevel * 0.085);
    elements.ring.style.scale = String(1 + smoothedLevel * 0.18);
  }
  const ambient = elements.ambient.getContext("2d");
  const aw = elements.ambient.width;
  const ah = elements.ambient.height;
  ambient.clearRect(0, 0, aw, ah);
  if (state.recording) {
    ambient.beginPath();
    for (let x = 0; x <= aw; x += 12 * ambientRatio) {
      const y = ah * 0.72 + Math.sin(x / (74 * ambientRatio) + time * 3) * (14 + state.level * 28) * ambientRatio;
      x === 0 ? ambient.moveTo(x, y) : ambient.lineTo(x, y);
    }
    ambient.strokeStyle = "rgba(198,12,46,0.18)";
    ambient.lineWidth = ambientRatio;
    ambient.stroke();
  }
  state.level *= 0.9;
  requestAnimationFrame(drawVisuals);
}

state.history = readConversationHistory();
renderHistory();
renderTranscript();
initGrainient();
requestAnimationFrame(drawVisuals);
worker.postMessage({type: "init", configUrl: signedAsset("model-config.json")});
