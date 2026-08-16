import type { ByteSource } from "./byte-source.ts";
import { Mp4Video, type VideoTrackFacts } from "./mp4-video.ts";

/// Frames kept decoded ahead of the playhead. The decoder runs well faster than
/// realtime (4K60 measured at ~3.25x), so this is about absorbing jitter — a
/// chunk read landing late must not show as a dropped frame — not throughput.
const LOOKAHEAD_FRAMES = 12;

/// Where presented frames go. Ownership of the frame passes with the call: the
/// sink closes it, because a `VideoFrame` holds a slot in a small decoder-side
/// pool and leaking one stalls decoding.
export interface FrameSink {
  present(frame: VideoFrame, mediaSeconds: number): void;
  ended(): void;
  failed(message: string): void;
}

/// Plays an MP4 by decoding it, replacing what a `<video>` element used to do.
///
/// The element supplied pacing for free — it decided when each frame was shown
/// and `requestVideoFrameCallback` reported it after the fact. A `VideoDecoder`
/// has no clock, so the clock is here: wall time since play started maps to a
/// media time, and the frame whose presentation interval covers it is the one
/// shown. That is also what makes frames addressable, which is the point of
/// #282 — every frame carries its container timestamp rather than being
/// identified by rounding a `currentTime`.
export class VideoPlayer {
  readonly #video: Mp4Video;
  readonly #sink: FrameSink;
  /// Decoded and waiting, in presentation order.
  readonly #queue: VideoFrame[] = [];
  #playing = false;
  /// `performance.now()` when the current play run started.
  #wallStartMs = 0;
  /// Media time the current play run started from.
  #mediaStartSeconds = 0;
  /// Media time of the last frame handed to the sink, so a paused seek still
  /// knows where it is.
  #positionSeconds = 0;
  #filling = false;
  #exhausted = false;
  /// Bumped by every seek and by close, so an in-flight fill for an older
  /// position discards its frames instead of showing them.
  #generation = 0;
  #frameHandle: number | undefined;

  private constructor(video: Mp4Video, sink: FrameSink) {
    this.#video = video;
    this.#sink = sink;
  }

  static async open(source: ByteSource, sink: FrameSink): Promise<VideoPlayer> {
    const video = await Mp4Video.open(source);
    // Deliberately does not seek: the caller has to publish the timeline first,
    // and a frame presented before that has no valid index to carry.
    return new VideoPlayer(video, sink);
  }

  get facts(): VideoTrackFacts {
    return this.#video.facts;
  }

  get moovBytes(): Uint8Array {
    return this.#video.moovBytes;
  }

  get source(): ByteSource {
    return this.#video.source;
  }

  get playing(): boolean {
    return this.#playing;
  }

  get positionSeconds(): number {
    return this.#positionSeconds;
  }

  play(): void {
    if (this.#playing || this.#exhausted) {
      return;
    }
    this.#playing = true;
    this.#wallStartMs = performance.now();
    this.#mediaStartSeconds = this.#positionSeconds;
    void this.#fill();
    this.#schedule();
  }

  pause(): void {
    this.#playing = false;
    if (this.#frameHandle !== undefined) {
      cancelAnimationFrame(this.#frameHandle);
      this.#frameHandle = undefined;
    }
  }

  /// Jumps to `seconds` and shows the frame covering it, playing or not.
  async seekToSeconds(seconds: number): Promise<void> {
    const generation = ++this.#generation;
    this.#drain();
    this.#exhausted = false;
    const target = await this.#video.seekTo(seconds);
    if (generation !== this.#generation) {
      return;
    }
    this.#positionSeconds = target;
    this.#mediaStartSeconds = target;
    this.#wallStartMs = performance.now();
    // Show the destination immediately rather than waiting for the clock, so a
    // paused scrub updates on every step.
    const frame = await this.#video.nextFrame();
    if (generation !== this.#generation) {
      frame?.close();
      return;
    }
    if (frame) {
      this.#show(frame);
    } else {
      this.#exhausted = true;
      this.#sink.ended();
    }
    void this.#fill();
  }

  close(): void {
    this.#generation += 1;
    this.pause();
    this.#drain();
    this.#video.close();
  }

  #show(frame: VideoFrame): void {
    const seconds = this.#video.presentationSeconds(frame);
    this.#positionSeconds = seconds;
    // Ownership passes to the sink, which closes it after the GPU copy.
    this.#sink.present(frame, seconds);
  }

  #drain(): void {
    for (const frame of this.#queue) {
      frame.close();
    }
    this.#queue.length = 0;
  }

  /// Keeps the lookahead topped up. Guarded rather than queued: one fill at a
  /// time, and a fill whose generation is stale throws its frames away.
  async #fill(): Promise<void> {
    if (this.#filling || this.#exhausted) {
      return;
    }
    this.#filling = true;
    const generation = this.#generation;
    try {
      while (this.#queue.length < LOOKAHEAD_FRAMES && !this.#exhausted) {
        const frame = await this.#video.nextFrame();
        if (generation !== this.#generation) {
          frame?.close();
          return;
        }
        if (!frame) {
          this.#exhausted = true;
          break;
        }
        this.#queue.push(frame);
      }
    } catch (error) {
      this.#sink.failed(String(error));
    } finally {
      this.#filling = false;
    }
  }

  #schedule(): void {
    this.#frameHandle = requestAnimationFrame(() => {
      this.#frameHandle = undefined;
      this.#advance();
      if (this.#playing) {
        this.#schedule();
      }
    });
  }

  #advance(): void {
    const target = this.#mediaStartSeconds + (performance.now() - this.#wallStartMs) / 1000;
    let due: VideoFrame | undefined;
    // Take every frame whose time has come and show only the newest: if the
    // display cannot keep up, dropping the intermediate frames is what keeps
    // playback on the clock rather than letting it drift ever further behind.
    while (this.#queue.length > 0) {
      const next = this.#queue[0];
      if (!next || this.#video.presentationSeconds(next) > target) {
        break;
      }
      due?.close();
      due = this.#queue.shift();
    }
    if (due) {
      this.#show(due);
    }
    void this.#fill();
    if (this.#exhausted && this.#queue.length === 0) {
      this.#playing = false;
      this.#sink.ended();
    }
  }
}
