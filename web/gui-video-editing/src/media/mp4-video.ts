import {
  createFile,
  Endianness,
  type ISOFile,
  MP4BoxBuffer,
  MultiBufferStream,
  type Sample,
  VisualSampleEntry,
} from "mp4box";
import type { ByteSource } from "./byte-source.ts";

/// 32 KiB: enough to cover `ftyp` and locate `moov`.
const HEAD_BYTES = 32 * 1024;

/// Largest read for sample data during a seek.
const CHUNK_BYTES = 4 * 1024 * 1024;

/// Back-pressure ceiling: max decode queue depth before feeding pauses.
const MAX_QUEUED_CHUNKS = 32;

/// Max idle ticks before a stalled seek is abandoned.
const MAX_IDLE_TICKS = 10_000;

/// Max bytes read from a key frame before giving up on a seek.
/// Resets each time a frame is delivered, so it is a per-frame budget.
const SEEK_BUDGET_BYTES = 256 * 1024 * 1024;

export interface VideoTrackFacts {
  readonly id: number;
  readonly codec: string;
  readonly width: number;
  readonly height: number;
  readonly timescale: number;
  readonly sampleCount: number;
  readonly durationSeconds: number;
  /// Presentation time of the first frame; mp4box reports raw `cts`, so
  /// frame 0 of the timeline is counted from here, not from zero.
  readonly startSeconds: number;
  /// Max seek time relative to `startSeconds`. Clamped to mp4box's own
  /// duration ceiling (last sample in decode order) to avoid empty seeks.
  readonly lastFrameSeconds: number;
  /// `avcC`/`hvcC` payload; a decoder configured without it silently emits nothing.
  readonly description: Uint8Array | undefined;
}

/// What one seek actually cost, so "the seek stayed cheap" is a measurement.
export interface SeekReport {
  /// The time actually sought to, after clamping into the video's range.
  readonly target: number;
  /// Frames decoded to get from the preceding key frame to the target, then
  /// discarded — the price of the container's key-frame spacing.
  readonly skipped: number;
  readonly delivered: number;
  /// Bytes transferred by this seek alone.
  readonly bytesRead: number;
  /// Presentation time of the first delivered frame, in seconds.
  readonly firstTime: number;
}

/// Pulls the codec-specific configuration box out of the sample description.
function descriptionFor(file: ISOFile, trackId: number): Uint8Array | undefined {
  const track = file.getTrackById(trackId);
  for (const entry of track?.mdia?.minf?.stbl?.stsd?.entries ?? []) {
    if (!(entry instanceof VisualSampleEntry)) {
      continue;
    }
    const box = entry.avcC ?? entry.hvcC ?? entry.vpcC ?? entry.av1C;
    if (box) {
      const stream = new MultiBufferStream();
      stream.endianness = Endianness.BIG_ENDIAN;
      box.write(stream);
      // Strip the 8-byte box header the writer emits.
      return new Uint8Array(stream.buffer, 8);
    }
  }
  return undefined;
}

/// Where a top-level box sits in the file.
export interface BoxExtent {
  readonly offset: number;
  readonly size: number;
}

/// Steps through top-level ISOBMFF boxes by declared size — never reads `mdat`.
/// Exported for tests.
export async function locateMoov(
  source: ByteSource,
  head: ArrayBuffer,
): Promise<BoxExtent | undefined> {
  let offset = 0;
  while (offset + 8 <= source.size) {
    let view: DataView;
    if (offset + 16 <= head.byteLength) {
      view = new DataView(head, offset, 16);
    } else {
      const buffer = await source.read(offset, 16);
      if (buffer.byteLength < 8) {
        return undefined;
      }
      view = new DataView(buffer);
    }
    const declared = view.getUint32(0);
    const kind = String.fromCharCode(
      view.getUint8(4),
      view.getUint8(5),
      view.getUint8(6),
      view.getUint8(7),
    );
    // size===1: 64-bit length follows; size===0: runs to EOF.
    let size = declared;
    if (declared === 1) {
      if (view.byteLength < 16) {
        return undefined;
      }
      size = Number(view.getBigUint64(8));
    } else if (declared === 0) {
      size = source.size - offset;
    }
    if (size < 8 || offset + size > source.size) {
      return undefined;
    }
    if (kind === "moov") {
      return { offset, size };
    }
    offset += size;
  }
  return undefined;
}

/// Frame-accurate MP4 reader over a `ByteSource`, driven by mp4box.
export class Mp4Video {
  readonly facts: VideoTrackFacts;
  /// Raw `moov` bytes; Rust reads the rational frame rate from it via `probe_moov`.
  readonly moovBytes: Uint8Array;
  readonly #source: ByteSource;
  readonly #file: ISOFile;
  /// Decoded frames at or after the seek target, in presentation order.
  readonly #ready: VideoFrame[] = [];
  #decoder: VideoDecoder | undefined;
  /// Frames ending at or before this (µs) are catch-up frames and are dropped.
  #skipTarget = 0;
  #skipped = 0;
  #feedOffset = 0;
  #budgetFrom = 0;
  #exhausted = true;
  /// Serialises decoder access; concurrent seeks reset the decoder mid-feed,
  /// causing a fatal `VideoDecoder` error.
  #gate: Promise<unknown> = Promise.resolve();
  /// Pending drain; awaiting flush() while holding its output would deadlock.
  #draining: Promise<void> | undefined;
  #drained = false;
  /// Woken whenever the decoder emits, so a waiter is released by output rather
  /// than by polling for it.
  #outputWaiters: (() => void)[] = [];
  /// Set by `close()`; prevents queued pulls from reviving the reader after teardown.
  #closed = false;
  /// Bumped each seek; stale queued work is dropped when the generation has changed.
  #generation = 0;
  /// Set on fatal decoder error; reported to callers, cleared on decoder rebuild.
  #decoderError: Error | undefined;

  /// Runs `operation` with no other reader operation in flight.
  #exclusive<T>(operation: () => Promise<T>): Promise<T> {
    const result = this.#gate.then(operation, operation);
    this.#gate = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  /// Resolves the next time the decoder emits — or the drain finishes.
  #nextOutput(): Promise<void> {
    return new Promise((resolve) => {
      this.#outputWaiters.push(resolve);
    });
  }

  #wakeOutputWaiters(): void {
    for (const wake of this.#outputWaiters.splice(0)) {
      wake();
    }
  }

  private constructor(
    source: ByteSource,
    file: ISOFile,
    facts: VideoTrackFacts,
    moovBytes: Uint8Array,
  ) {
    this.#source = source;
    this.#file = file;
    this.facts = facts;
    this.moovBytes = moovBytes;
  }

  get source(): ByteSource {
    return this.#source;
  }

  static async open(source: ByteSource): Promise<Mp4Video> {
    const file = createFile();
    let facts: VideoTrackFacts | undefined;
    let failure: string | undefined;
    file.onError = (module, message) => {
      failure ??= `${module}: ${message}`;
    };
    file.onReady = (info) => {
      const track = info.videoTracks[0];
      if (!track?.video) {
        failure ??= "the file has no video track";
        return;
      }
      // Sample lists are available at `onReady`. Track both the max presentation
      // time and mp4box's own duration ceiling; clamp to the smaller so seeks
      // to the end land on a real frame.
      const samples = file.getTrackById(track.id)?.samples ?? [];
      let lastCts = 0;
      let firstCts = Number.POSITIVE_INFINITY;
      for (const sample of samples) {
        if (sample.cts > lastCts) {
          lastCts = sample.cts;
        }
        if (sample.cts < firstCts) {
          firstCts = sample.cts;
        }
      }
      const decodeLast = samples[samples.length - 1];
      const sampleTimescale = decodeLast?.timescale ?? track.timescale;
      const mp4boxEnd = decodeLast
        ? (decodeLast.cts + decodeLast.duration) / decodeLast.timescale
        : 0;
      const startSeconds = Number.isFinite(firstCts) ? firstCts / sampleTimescale : 0;
      facts = {
        id: track.id,
        codec: track.codec,
        width: track.video.width,
        height: track.video.height,
        timescale: track.timescale,
        sampleCount: track.nb_samples,
        durationSeconds: track.duration / track.timescale,
        startSeconds,
        lastFrameSeconds: Math.min(lastCts / sampleTimescale, mp4boxEnd) - startSeconds,
        description: descriptionFor(file, track.id),
      };
      file.setExtractionOptions(track.id, null, { nbSamples: 1 });
    };

    // Fetch `moov` directly by size; mp4box's incremental discovery needs multiple reads.
    const headLength = Math.min(HEAD_BYTES, source.size);
    const head = await source.read(0, headLength);
    const moov = await locateMoov(source, head);
    if (!moov) {
      throw new Error(`no moov box found in "${source.label}"`);
    }
    // Feed any boxes before `moov` first (the parser needs position 0 to start).
    const prefix = Math.min(head.byteLength, moov.offset);
    if (prefix > 0) {
      file.appendBuffer(MP4BoxBuffer.fromArrayBuffer(head.slice(0, prefix), 0));
    }
    const moovEnd = moov.offset + moov.size;
    const moovBytes =
      moovEnd <= head.byteLength
        ? head.slice(moov.offset, moovEnd)
        : await source.read(moov.offset, moov.size);
    file.appendBuffer(MP4BoxBuffer.fromArrayBuffer(moovBytes, moov.offset));
    if (failure !== undefined) {
      throw new Error(failure);
    }
    if (!facts) {
      throw new Error(`"${source.label}" has a moov box but no readable video track`);
    }
    return new Mp4Video(source, file, facts, new Uint8Array(moovBytes));
  }

  #ensureDecoder(): VideoDecoder {
    // Replace a closed/errored decoder — reusing it throws `InvalidStateError`.
    if (!this.#decoder || this.#decoder.state === "closed") {
      this.#decoderError = undefined;
      this.#decoder = new VideoDecoder({
        output: (frame) => {
          // `?? 1`: a frame starting exactly at the target must not be skipped.
          if (frame.timestamp + (frame.duration ?? 1) <= this.#skipTarget) {
            this.#skipped += 1;
            frame.close();
            this.#wakeOutputWaiters();
            return;
          }
          this.#ready.push(frame);
          this.#wakeOutputWaiters();
        },
        error: (error) => {
          this.#decoderError = error;
          this.#exhausted = true;
          this.#wakeOutputWaiters();
        },
      });
    }
    return this.#decoder;
  }

  /// Seeks to the key frame at or before `seconds` (presentation time from video start).
  /// Returns the clamped time actually sought to.
  async seekTo(seconds: number): Promise<number> {
    return this.#exclusive(() => this.#seek(seconds));
  }

  async #seek(seconds: number): Promise<number> {
    if (this.#closed) {
      return 0;
    }
    // Clamp to the valid range; mp4box returns a bad offset for out-of-range seeks.
    const wantedSeconds = Math.min(Math.max(0, seconds), this.facts.lastFrameSeconds);
    const containerSeconds = wantedSeconds + this.facts.startSeconds;
    this.#generation += 1;
    this.#skipTarget = containerSeconds * 1_000_000;
    this.#skipped = 0;
    this.#exhausted = false;
    this.#draining = undefined;
    this.#drained = false;
    this.#wakeOutputWaiters();
    this.#budgetFrom = this.#source.bytesRead;
    for (const frame of this.#ready) {
      frame.close();
    }
    this.#ready.length = 0;

    const decoder = this.#ensureDecoder();
    if (decoder.state === "configured") {
      decoder.reset();
    }
    decoder.configure({
      codec: this.facts.codec,
      codedWidth: this.facts.width,
      codedHeight: this.facts.height,
      ...(this.facts.description ? { description: this.facts.description } : {}),
    });
    this.#file.onSamples = (_id: number, _user: unknown, samples: Sample[]) => {
      for (const sample of samples) {
        if (!sample.data) {
          continue;
        }
        decoder.decode(
          new EncodedVideoChunk({
            type: sample.is_sync ? "key" : "delta",
            // mp4box reconciles `ctts` reordering for B-frames.
            timestamp: (sample.cts * 1_000_000) / sample.timescale,
            duration: (sample.duration * 1_000_000) / sample.timescale,
            data: sample.data,
          }),
        );
      }
      // The samples are in the decoder's queue now; mp4box can drop its copies.
      this.#file.releaseUsedSamples(this.facts.id, samples[samples.length - 1]?.number ?? 0);
    };
    // `seek()` returns the next read position, not the key frame offset.
    // If mp4box already holds all bytes, it returns EOF — not an error.
    this.#feedOffset = this.#file.seek(containerSeconds, true).offset;
    // `start()` must be called after each seek for mp4box to re-emit from held data.
    this.#file.start();
    return wantedSeconds;
  }

  /// Presentation time of `frame` measured from the start of the video, which
  /// is what a timeline index is derived from — not the raw container `cts`.
  presentationSeconds(frame: VideoFrame): number {
    return frame.timestamp / 1_000_000 - this.facts.startSeconds;
  }

  /// Next frame in presentation order from the most recent seek, or `undefined` at end.
  /// **Caller owns the frame and must `close()` it** — leaking stalls the decoder pool.
  async nextFrame(): Promise<VideoFrame | undefined> {
    return this.#exclusive(() => this.#pullFrame());
  }

  async #pullFrame(): Promise<VideoFrame | undefined> {
    if (this.#closed) {
      return undefined;
    }
    const decoder = this.#ensureDecoder();
    const generation = this.#generation;
    let idleTicks = 0;
    while (this.#ready.length === 0 && !this.#exhausted) {
      if (this.#decoderError) {
        throw this.#decoderError;
      }
      if (this.#closed) {
        return undefined;
      }
      if (generation !== this.#generation) {
        return undefined;
      }
      if (
        this.#feedOffset >= this.#source.size ||
        this.#source.bytesRead - this.#budgetFrom > SEEK_BUDGET_BYTES
      ) {
        // Awaiting flush() directly deadlocks: flush needs frames closed,
        // frames are closed by the caller, caller is blocked here.
        // Race the drain against the next output so both can make progress.
        if (!this.#draining) {
          this.#draining = decoder
            .flush()
            .catch(() => undefined)
            .then(() => {
              this.#drained = true;
              this.#wakeOutputWaiters();
            });
        }
        await Promise.race([this.#draining, this.#nextOutput()]);
        if (this.#ready.length > 0) {
          break;
        }
        if (this.#drained) {
          // A completed drain on an empty decoder isn't end-of-stream:
          // mp4box delivers samples async, so chunks may still arrive.
          if (decoder.decodeQueueSize > 0) {
            this.#draining = undefined;
            this.#drained = false;
            continue;
          }
          this.#exhausted = true;
          break;
        }
        continue;
      }
      // Back-pressure: pause feeding while the decode queue is full.
      while (decoder.decodeQueueSize > MAX_QUEUED_CHUNKS && this.#ready.length === 0) {
        await new Promise((resolve) => {
          setTimeout(resolve, 0);
        });
        if (this.#decoderError || ++idleTicks > MAX_IDLE_TICKS) {
          break;
        }
      }
      if (this.#ready.length > 0) {
        break;
      }
      const buffer = await this.#source.read(this.#feedOffset, CHUNK_BYTES);
      if (buffer.byteLength === 0) {
        this.#feedOffset = this.#source.size;
        continue;
      }
      const next = this.#file.appendBuffer(MP4BoxBuffer.fromArrayBuffer(buffer, this.#feedOffset));
      this.#feedOffset = next > this.#feedOffset ? next : this.#feedOffset + buffer.byteLength;
    }
    if (this.#decoderError) {
      throw this.#decoderError;
    }
    const frame = this.#ready.shift();
    if (frame) {
      // Reset the per-frame read budget.
      this.#budgetFrom = this.#source.bytesRead;
    }
    return frame;
  }

  /// Seeks and takes the next `wanted` frames — the measurement shape the probe
  /// reports, on top of the same pull the player uses.
  async seekAndDecode(
    seconds: number,
    wanted: number,
    onFrame: (frame: VideoFrame) => void,
  ): Promise<SeekReport> {
    const before = this.#source.bytesRead;
    const target = await this.seekTo(seconds);
    let delivered = 0;
    let firstTime = Number.NaN;
    while (delivered < wanted) {
      const frame = await this.nextFrame();
      if (!frame) {
        break;
      }
      if (delivered === 0) {
        firstTime = this.presentationSeconds(frame);
      }
      delivered += 1;
      onFrame(frame);
    }
    return {
      target,
      skipped: this.#skipped,
      delivered,
      bytesRead: this.#source.bytesRead - before,
      firstTime,
    };
  }

  close(): void {
    this.#closed = true;
    this.#wakeOutputWaiters();
    for (const frame of this.#ready) {
      frame.close();
    }
    this.#ready.length = 0;
    // Closing an errored decoder throws; guard against it.
    if (this.#decoder && this.#decoder.state !== "closed") {
      this.#decoder.close();
    }
    this.#decoder = undefined;
  }
}
