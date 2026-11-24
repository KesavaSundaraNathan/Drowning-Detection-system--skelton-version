/**
 * Alarm and fault tones (Web Audio - no audio files). Browsers only allow sound after a user
 * gesture, which is why monitoring is shown only after the operator clicks "Arm audio".
 *   alarm: fast siren - every unacknowledged Drowning Alert ([Stage 4] one tone, no per-type tiers)
 *   fault: slow low pulse (NOT MONITORING / FAULT)
 * Both sound until acknowledged.
 */
export type Tone = "alarm" | "fault";

export interface ToneEngine {
  arm(): Promise<boolean>;
  play(tone: Tone): void;
  stop(): void;
  readonly playing: Tone | null;
}

/** Per tone: waveform, pitch, step length and the on-steps out of every `cycle` steps (alarm alternates pitch). */
export const TONE_SPECS: Record<Tone, { wave: OscillatorType; hz: number; stepMs: number; on: number; cycle: number }> = {
  alarm: { wave: "square", hz: 960, stepMs: 250, on: 1, cycle: 1 },
  fault: { wave: "sine", hz: 440, stepMs: 600, on: 1, cycle: 2 },
};

class WebAudioTones implements ToneEngine {
  private ctx: AudioContext | null = null;
  private osc: OscillatorNode | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  playing: Tone | null = null;

  async arm(): Promise<boolean> {
    const Ctor = globalThis.AudioContext as typeof AudioContext | undefined;
    if (!Ctor) return false;
    this.ctx ??= new Ctor();
    await this.ctx.resume();
    return this.ctx.state === "running";
  }

  /** Last audio failure (shown nowhere critical: a broken sound device must never break the screen). */
  lastError: string | null = null;

  play(tone: Tone): void {
    if (this.playing === tone || this.ctx === null) return;
    this.stop();
    try {
      this.start(this.ctx, tone);
    } catch (e) {
      this.lastError = e instanceof Error ? e.message : String(e);
      this.playing = null;
    }
  }

  private start(ctx: AudioContext, tone: Tone): void {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    gain.gain.value = 0.25;
    osc.connect(gain).connect(ctx.destination);
    const spec = TONE_SPECS[tone];
    osc.type = spec.wave;
    osc.frequency.value = spec.hz;
    osc.start();
    let tick = 0;
    this.timer = setInterval(() => {
      tick += 1;
      if (tone === "alarm") osc.frequency.value = tick % 2 === 0 ? 960 : 720;
      else gain.gain.value = tick % spec.cycle < spec.on ? 0.25 : 0;
    }, spec.stepMs);
    this.osc = osc;
    this.playing = tone;
  }

  stop(): void {
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
    try {
      this.osc?.stop();
    } catch {
      // already stopped / device gone
    }
    this.osc = null;
    this.playing = null;
  }
}

export const tones: ToneEngine = new WebAudioTones();
