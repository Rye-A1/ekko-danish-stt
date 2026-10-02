class EkkoCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.stopped = false;
    this.port.onmessage = (event) => {
      if (event.data?.type !== "stop") return;
      this.stopped = true;
      this.port.postMessage({type: "stopped"});
    };
  }

  process(inputs) {
    if (this.stopped) return true;
    const input = inputs[0];
    if (input && input[0]) {
      const samples = new Float32Array(input[0]);
      this.port.postMessage(samples, [samples.buffer]);
    }
    return true;
  }
}

registerProcessor("ekko-capture", EkkoCaptureProcessor);
