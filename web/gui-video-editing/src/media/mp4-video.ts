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
  /// The `avcC`/`hvcC` payload. An AVC decoder configured without it accepts
  /// the configuration and then silently emits nothing, so its absence is worth
  /// reporting rather than tolerating.
  readonly description: Uint8Array | undefined;
}

/// What one seek actually cost, so "the seek stayed cheap" is a measurement.
export interface SeekReport {
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
  readonly #source: ByteSource;
  readonly #file: ISOFile;
  #decoder: VideoDecoder | undefined;
  #onOutput: ((frame: VideoFrame) => void) | undefined;
  #started = false;

  private constructor(source: ByteSource, file: ISOFile, facts: VideoTrackFacts) {
    this.#source = source;
    this.#file = file;
    this.facts = facts;
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
      facts = {
        id: track.id,
        codec: track.codec,
        width: track.video.width,
        height: track.video.height,
        timescale: track.timescale,
        sampleCount: track.nb_samples,
        durationSeconds: track.duration / track.timescale,
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
    return new Mp4Video(source, file, facts);
  }

  #ensureDecoder(): VideoDecoder {
    this.#decoder ??= new VideoDecoder({
      output: (frame) => {
        const handler = this.#onOutput;
        if (handler) {
          handler(frame);
        } else {
          frame.close();
        }
      },
      error: (error) => {
        console.error("video decode failed:", error);
      },
    });
    return this.#decoder;
  }

  /// Decodes the frame covering `seconds` and the `wanted - 1` frames after it.
  ///
  /// This is what a `<video>` element does internally for `currentTime = t`:
  /// start at the key frame at or before the target, decode forward, and drop
  /// what precedes it. The difference is that every delivered frame carries its
  /// container timestamp, so it is identified rather than approximated.
  async seekAndDecode(
    seconds: number,
    wanted: number,
    onFrame: (frame: VideoFrame) => void,
  ): Promise<SeekReport> {
    const before = this.#source.bytesRead;
    const target = Math.max(0, seconds) * 1_000_000;
    let skipped = 0;
    let delivered = 0;
    let firstTime = Number.NaN;

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
    this.#onOutput = (frame) => {
      // `?? 1` so a frame starting exactly at the target is not mistaken for one
      // that ends there.
      const end = frame.timestamp + (frame.duration ?? 1);
      if (end <= target) {
        skipped += 1;
        frame.close();
        return;
      }
      if (delivered >= wanted) {
        frame.close();
        return;
      }
      if (delivered === 0) {
        firstTime = frame.timestamp / 1_000_000;
      }
      delivered += 1;
      onFrame(frame);
    };

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

    const seek = this.#file.seek(seconds, true);
    if (!this.#started) {
      this.#file.start();
      this.#started = true;
    }
    // Feed until the wanted frames have actually come out. Counting queued
    // samples instead would race: the decoder reports what it skipped only
    // after it has decoded it, so a key frame seconds ahead of the target would
    // end the loop before a single frame was delivered.
    let offset = seek.offset;
    let idleTicks = 0;
    while (delivered < wanted && offset < this.#source.size) {
      // Back-pressure. Catching up from a distant key frame can mean hundreds
      // of frames, and queueing all of them at once would hold the whole span
      // in memory for no gain.
      while (decoder.decodeQueueSize > MAX_QUEUED_CHUNKS && delivered < wanted) {
        await new Promise((resolve) => {
          setTimeout(resolve, 0);
        });
        if (++idleTicks > MAX_IDLE_TICKS) {
          break;
        }
      }
      if (delivered >= wanted) {
        break;
      }
      const buffer = await this.#source.read(offset, CHUNK_BYTES);
      if (buffer.byteLength === 0) {
        break;
      }
      const next = this.#file.appendBuffer(MP4BoxBuffer.fromArrayBuffer(buffer, offset));
      offset = next > offset ? next : offset + buffer.byteLength;
    }
    await decoder.flush();
    this.#onOutput = undefined;
    return {
      skipped,
      delivered,
      bytesRead: this.#source.bytesRead - before,
      firstTime,
    };
  }

  close(): void {
    this.#decoder?.close();
    this.#decoder = undefined;
  }
}
