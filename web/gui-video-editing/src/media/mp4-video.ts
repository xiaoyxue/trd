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

/// First read when opening: enough to cover `ftyp` plus the header of whatever
/// follows, which is all the box walk below needs to decide where `moov` is.
const HEAD_BYTES = 32 * 1024;

/// Largest read for sample data during a seek.
const CHUNK_BYTES = 4 * 1024 * 1024;

/// Chunks allowed in the decoder's queue before feeding pauses. Enough to keep
/// it busy while catching up from a distant key frame, few enough that the span
/// is not held in memory all at once.
const MAX_QUEUED_CHUNKS = 32;

/// Ceiling on back-pressure waits, so a decoder that stops draining ends the
/// seek rather than hanging it.
const MAX_IDLE_TICKS = 10_000;

/// Most one seek will transfer before giving up. A key-frame interval is a few
/// seconds at worst, so needing more than this means something is wrong —
/// report an empty seek rather than walk a multi-gigabyte file.
const SEEK_BUDGET_BYTES = 256 * 1024 * 1024;

/// What a decoded video's timeline is made of. Every field comes from the
/// container, so a frame's identity is read rather than inferred — the reason
/// for moving off `<video>`, which reports no frame rate at all (#282).
export interface VideoTrackFacts {
  readonly id: number;
  readonly codec: string;
  readonly width: number;
  readonly height: number;
  readonly timescale: number;
  readonly sampleCount: number;
  readonly durationSeconds: number;
  /// Presentation time of the first frame in the container. Not always zero:
  /// with B-frames the composition times are commonly shifted forward and an
  /// edit list shifts the timeline back to compensate. A `<video>` element
  /// applies that list; mp4box reports raw `cts`, so frame *N* of the timeline
  /// is the *N*-th frame from here, not from zero. Ignoring it put the FIBA
  /// clip two frames late on every seek.
  readonly startSeconds: number;
  /// The largest time a seek may ask for, relative to `startSeconds`. Neither
  /// `durationSeconds` nor simply the last frame's presentation time: mp4box
  /// refuses to seek past its own duration, taken from the last sample in
  /// *decode* order, and answers such a request with an offset that decodes
  /// nothing — which then reads to the end of the file looking for a frame that
  /// cannot exist.
  readonly lastFrameSeconds: number;
  /// The `avcC`/`hvcC` payload. An AVC decoder configured without it accepts
  /// the configuration and then silently emits nothing, so its absence is worth
  /// reporting rather than tolerating.
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
    // Only a visual entry carries one, and narrowing to it is what makes the
    // per-codec boxes visible at all.
    if (!(entry instanceof VisualSampleEntry)) {
      continue;
    }
    const box = entry.avcC ?? entry.hvcC ?? entry.vpcC ?? entry.av1C;
    if (box) {
      // `write` serialises through the same stream type the parser reads, and
      // these boxes are big-endian like everything else in ISOBMFF.
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
interface BoxExtent {
  readonly offset: number;
  readonly size: number;
}

/// Finds `moov` by stepping through the top-level box list, reading headers
/// only — never the `mdat` payload between them.
///
/// Stepping by each box's recorded size is the only sound way to do this.
/// Scanning for the bytes `6D 6F 6F 76` would also match them occurring inside
/// compressed sample data or a metadata payload, and assuming `moov` is last
/// (`totalSize - moovSize`) is wrong for any file that ends with `free`, `skip`
/// or `mfra`.
async function locateMoov(source: ByteSource, head: ArrayBuffer): Promise<BoxExtent | undefined> {
  let offset = 0;
  while (offset + 8 <= source.size) {
    // Reuse the head read wherever it reaches; past it a bare 16-byte header is
    // all that has to be fetched to keep walking.
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
    // `size === 1` puts a 64-bit length after the type — how a `mdat` larger
    // than 4 GiB is expressed, which is exactly the case here. `size === 0`
    // means the box runs to the end of the file.
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

/// An MP4 opened for frame-accurate reading over a `ByteSource`.
///
/// mp4box drives the byte fetching: `appendBuffer` returns the file position it
/// wants next, which is how a `moov` at the end of the file is reached without
/// reading the `mdat` in between, and how a seek reaches its key frame. So the
/// whole loader is one loop that keeps feeding whatever position mp4box asks
/// for.
export class Mp4Video {
  readonly facts: VideoTrackFacts;
  /// The `moov` box exactly as read. Rust derives the timeline from it — the
  /// frame rate as a rational, which mp4box reports only as a sample count over
  /// a duration — through the same `probe_moov` every other front-end uses.
  readonly moovBytes: Uint8Array;
  readonly #source: ByteSource;
  readonly #file: ISOFile;
  /// Decoded frames at or after the seek target, in presentation order. The
  /// decoder emits in presentation order, so this needs no reordering.
  readonly #ready: VideoFrame[] = [];
  #decoder: VideoDecoder | undefined;
  #started = false;
  /// Frames ending at or before this (microseconds) are the catch-up from the
  /// key frame and get dropped.
  #skipTarget = 0;
  #skipped = 0;
  #feedOffset = 0;
  #budgetFrom = 0;
  #exhausted = true;
  /// Serialises the operations that touch the decoder and the feed cursor. A
  /// seek that lands while an earlier feed is mid-`await` resets the decoder
  /// under it, and the samples that arrive after the reset begin with a delta
  /// frame — which a `VideoDecoder` reports as a **fatal** error and never
  /// recovers from. Dragging a scrubber produces exactly that overlap.
  #gate: Promise<unknown> = Promise.resolve();
  /// Bumped by every seek, so work queued for a superseded one is dropped
  /// instead of decoding a span nobody is waiting for any more.
  #generation = 0;
  /// Set by the decoder's error callback. A fatal error closes the codec for
  /// good, so it is recorded and reported rather than only logged — the next
  /// seek then builds a fresh decoder.
  #decoderError: Error | undefined;

  /// Runs `operation` with no other reader operation in flight.
  #exclusive<T>(operation: () => Promise<T>): Promise<T> {
    const result = this.#gate.then(operation, operation);
    // The gate must survive a rejected operation, or one failure would wedge
    // every later call behind it.
    this.#gate = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
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
      // Sample lists are built before `onReady` fires, so the true last frame
      // is available here. Taking the maximum rather than the last entry
      // because with B-frames decode order is not presentation order.
      // Sample lists are built before `onReady` fires, so the true extent is
      // available here. Two different "ends" matter: the largest presentation
      // time (decode order is not presentation order with B-frames), and
      // mp4box's own idea of the duration, which it takes from the last sample
      // in *decode* order and refuses to seek past. Clamping to the smaller is
      // what makes a seek to the end land on a frame instead of falling back to
      // a meaningless offset.
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
      // Registering the track does not start delivery; `start()` does, and that
      // waits until a seek has said where to read from.
      file.setExtractionOptions(track.id, null, { nbSamples: 1 });
    };

    // Walk the top-level boxes to find `moov`, then fetch exactly it. Letting
    // mp4box discover it by asking for more data works too, but it cannot know
    // how much a `moov` needs until it has it, so a large sample table costs
    // several growing reads; its own header states the size, so one read does.
    const headLength = Math.min(HEAD_BYTES, source.size);
    const head = await source.read(0, headLength);
    const moov = await locateMoov(source, head);
    if (!moov) {
      throw new Error(`no moov box found in "${source.label}"`);
    }
    // The parser starts at position 0, so it needs the boxes in front of `moov`
    // before `moov` itself means anything — for a file that ends with `moov`
    // that is just the header of the `mdat` it will skip over.
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
    // A codec that has errored is `closed` for good: reusing it makes every
    // later `configure`/`decode` throw `InvalidStateError`, which is how one
    // transient fault used to wedge playback permanently. Replace it instead.
    if (!this.#decoder || this.#decoder.state === "closed") {
      this.#decoderError = undefined;
      this.#decoder = new VideoDecoder({
        output: (frame) => {
          // `?? 1` so a frame starting exactly at the target is not mistaken for
          // one that ends there.
          if (frame.timestamp + (frame.duration ?? 1) <= this.#skipTarget) {
            this.#skipped += 1;
            frame.close();
            return;
          }
          this.#ready.push(frame);
        },
        error: (error) => {
          // Recorded, not just logged: whoever is awaiting a frame has to learn
          // that none is coming, and the decoder is unusable from here on.
          this.#decoderError = error;
          this.#exhausted = true;
        },
      });
    }
    return this.#decoder;
  }

  /// Positions the reader at the key frame at or before `seconds`, discarding
  /// what precedes the target. Returns the time actually sought to.
  ///
  /// `seconds` is **presentation time from the start of the video**, so 0 is the
  /// first frame; `startSeconds` converts to the container's own clock.
  ///
  /// This is what a `<video>` element does internally for `currentTime = t`.
  /// The difference is that every frame that comes out carries its container
  /// timestamp, so it is identified rather than approximated.
  async seekTo(seconds: number): Promise<number> {
    return this.#exclusive(() => this.#seek(seconds));
  }

  async #seek(seconds: number): Promise<number> {
    // Clamp into the video's real range. A scrubber can ask for a time past the
    // end, and mp4box answers that with an offset that decodes nothing — which
    // would then read to the end of the file looking for a frame that does not
    // exist.
    const wantedSeconds = Math.min(Math.max(0, seconds), this.facts.lastFrameSeconds);
    const containerSeconds = wantedSeconds + this.facts.startSeconds;
    this.#generation += 1;
    this.#skipTarget = containerSeconds * 1_000_000;
    this.#skipped = 0;
    this.#exhausted = false;
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
            // Presentation time, in microseconds. mp4box reconciles `ctts`
            // reordering here, which is precisely what a hand-written demuxer
            // gets wrong on a stream with B-frames.
            timestamp: (sample.cts * 1_000_000) / sample.timescale,
            duration: (sample.duration * 1_000_000) / sample.timescale,
            data: sample.data,
          }),
        );
      }
      // The samples are in the decoder's queue now; mp4box can drop its copies.
      this.#file.releaseUsedSamples(this.facts.id, samples[samples.length - 1]?.number ?? 0);
    };
    this.#feedOffset = this.#file.seek(containerSeconds, true).offset;
    if (!this.#started) {
      this.#file.start();
      this.#started = true;
    }
    return wantedSeconds;
  }

  /// Presentation time of `frame` measured from the start of the video, which
  /// is what a timeline index is derived from — not the raw container `cts`.
  presentationSeconds(frame: VideoFrame): number {
    return frame.timestamp / 1_000_000 - this.facts.startSeconds;
  }

  /// The next frame in presentation order, or `undefined` past the end.
  ///
  /// Frames come from the **most recent** seek. A pull that is still in flight
  /// when a newer seek arrives yields the newer seek's frames rather than the
  /// span it originally asked for — a scrubber only cares where it ended up,
  /// and a caller that needs to tell the two apart tracks its own generation
  /// the way `VideoPlayer` does.
  ///
  /// **The caller owns the frame and must `close()` it.** A `VideoFrame` holds a
  /// slot in a small decoder-side pool, so leaking one stalls decoding.
  async nextFrame(): Promise<VideoFrame | undefined> {
    return this.#exclusive(() => this.#pullFrame());
  }

  async #pullFrame(): Promise<VideoFrame | undefined> {
    const decoder = this.#ensureDecoder();
    const generation = this.#generation;
    let idleTicks = 0;
    // Feed until a frame is ready. Waiting on frames *out* rather than samples
    // *in* is what makes this correct across a distant key frame: the decoder
    // reports what it skipped only after decoding it.
    while (this.#ready.length === 0 && !this.#exhausted) {
      if (this.#decoderError) {
        throw this.#decoderError;
      }
      if (generation !== this.#generation) {
        return undefined;
      }
      if (
        this.#feedOffset >= this.#source.size ||
        this.#source.bytesRead - this.#budgetFrom > SEEK_BUDGET_BYTES
      ) {
        await decoder.flush();
        this.#exhausted = true;
        break;
      }
      // Back-pressure. Catching up from a distant key frame can mean hundreds
      // of frames, and queueing all of them at once would hold the whole span
      // in memory for no gain.
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
        await decoder.flush();
        this.#exhausted = true;
        break;
      }
      const next = this.#file.appendBuffer(MP4BoxBuffer.fromArrayBuffer(buffer, this.#feedOffset));
      this.#feedOffset = next > this.#feedOffset ? next : this.#feedOffset + buffer.byteLength;
    }
    if (this.#decoderError) {
      throw this.#decoderError;
    }
    return this.#ready.shift();
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
    for (const frame of this.#ready) {
      frame.close();
    }
    this.#ready.length = 0;
    // Closing a codec that has already errored throws; a teardown must not
    // depend on whether decoding happened to fail first.
    if (this.#decoder && this.#decoder.state !== "closed") {
      this.#decoder.close();
    }
    this.#decoder = undefined;
  }
}
