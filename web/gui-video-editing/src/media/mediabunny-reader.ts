/// The browser media layer delegated to [mediabunny], as an alternative to the
/// hand-written mp4box + `VideoDecoder` reader next door: range reads, locating
/// `moov`, feeding the demuxer, decoder configuration and reset, key-frame
/// catch-up and the end-of-stream drain are all its job.
///
/// It does not supply the raw `moov`, which Rust parses to get the frame rate as
/// a rational; that is one extra range read, done here with the same
/// `ByteSource` and box walk the other reader uses.
///
/// [mediabunny]: https://mediabunny.dev/
import { ALL_FORMATS, BlobSource, Input, UrlSource, VideoSampleSink } from "mediabunny";

import type { ByteSource } from "./byte-source.ts";
import { fileByteSource, urlByteSource } from "./byte-source.ts";
import type { FrameReader } from "./frame-reader.ts";
import { locateMoov, type VideoTrackFacts } from "./mp4-video.ts";

/// What a player is opened from, before either reader has been chosen.
export type MediaInput =
  | { readonly kind: "file"; readonly file: File }
  | {
      readonly kind: "url";
      readonly url: string;
    };

/// Opens the byte source the *other* reader uses. Kept out of the class so the
/// two readers cannot drift on how a local file and a URL are opened.
export async function byteSourceFor(input: MediaInput): Promise<ByteSource> {
  return input.kind === "file" ? fileByteSource(input.file) : urlByteSource(input.url);
}

/// Reads the `moov` box whole. Two range reads at worst — the box walk needs
/// only headers — which is what keeps opening a 200 GiB file cheap.
async function readMoovBytes(source: ByteSource): Promise<Uint8Array> {
  const head = await source.read(0, Math.min(32 * 1024, source.size));
  const moov = await locateMoov(source, head);
  if (!moov) {
    throw new Error(`no moov box found in "${source.label}"`);
  }
  const end = moov.offset + moov.size;
  const bytes =
    end <= head.byteLength
      ? head.slice(moov.offset, end)
      : await source.read(moov.offset, moov.size);
  return new Uint8Array(bytes);
}

export class MediabunnyReader implements FrameReader {
  readonly facts: VideoTrackFacts;
  readonly moovBytes: Uint8Array;
  readonly #sink: VideoSampleSink;
  /// The current run of frames, in presentation order. A seek replaces it;
  /// mediabunny frees the iterator's decoder when it is returned.
  #samples: AsyncGenerator<import("mediabunny").VideoSample, void, unknown> | undefined;
  #closed = false;
  /// Bumped by every seek, so that overlapping ones cannot install the loser's
  /// run — the same guard the mp4box reader keeps next door. Both clear
  /// `#samples` before awaiting, so without it the seek that *resumes* last
  /// installs its iterator over the one the seek asked for last, and that
  /// overwritten iterator is never returned — which mediabunny reports as a
  /// `VideoSample` garbage collected without being closed.
  #generation = 0;

  private constructor(facts: VideoTrackFacts, moovBytes: Uint8Array, sink: VideoSampleSink) {
    this.facts = facts;
    this.moovBytes = moovBytes;
    this.#sink = sink;
  }

  static async open(input: MediaInput): Promise<MediabunnyReader> {
    const source = input.kind === "file" ? new BlobSource(input.file) : new UrlSource(input.url);
    const label = input.kind === "file" ? input.file.name : input.url;
    // `ALL_FORMATS` costs bundle size for parsers we do not use; the editor
    // validates MP4 and nothing else, so say so.
    const media = new Input({ formats: ALL_FORMATS, source });
    const track = await media.getPrimaryVideoTrack();
    if (!track) {
      throw new Error(`"${label}" has no video track`);
    }
    const [codec, startSeconds, metadataDuration, resolution] = await Promise.all([
      track.getCodecParameterString(),
      track.getFirstTimestamp(),
      track.getDurationFromMetadata(),
      track.getTimeResolution(),
    ]);
    // `compute*` walks the sample table; `get*FromMetadata` reads what the
    // container already states. On the 218 GiB recording — 694,840 packets —
    // that distinction was 99 seconds against under two, and it buys nothing the
    // editor uses: the authoritative timeline is the `moov` Rust parses.
    const durationSeconds = metadataDuration ?? (await track.computeDuration());
    const frameInterval = resolution > 0 ? 1 / resolution : 0;
    const facts: VideoTrackFacts = {
      id: track.id,
      codec: codec ?? "",
      width: track.displayWidth,
      height: track.displayHeight,
      timescale: resolution,
      // Not available without walking the packets, and not authoritative here
      // in any case — the frame count the editor uses is read from the `moov`.
      sampleCount: 0,
      durationSeconds,
      startSeconds,
      // The reader's own clamp for a scrub past the end: the last frame starts
      // one interval before the track ends.
      lastFrameSeconds: Math.max(0, durationSeconds - startSeconds - frameInterval),
      // The decoder is mediabunny's; nothing here has to hand a description to
      // one.
      description: undefined,
    };

    const byteSource = await byteSourceFor(input);
    const moovBytes = await readMoovBytes(byteSource);
    return new MediabunnyReader(facts, moovBytes, new VideoSampleSink(track));
  }

  async seekTo(seconds: number): Promise<number> {
    const wanted = Math.min(Math.max(0, seconds), this.facts.lastFrameSeconds);
    const generation = ++this.#generation;
    await this.#endRun();
    // A later seek having started while this one waited makes this one stale:
    // its frames are not the ones a scrubber is asking for, and installing them
    // would orphan the run the winner already opened.
    if (this.#closed || generation !== this.#generation) {
      return wanted;
    }
    // Presentation time in the container's own clock, the same conversion the
    // other reader makes: a timeline index counts from the first frame, not
    // from zero.
    this.#samples = this.#sink.samples(wanted + this.facts.startSeconds);
    return wanted;
  }

  async nextFrame(): Promise<VideoFrame | undefined> {
    if (this.#closed || !this.#samples) {
      return undefined;
    }
    const next = await this.#samples.next();
    if (next.done || !next.value) {
      return undefined;
    }
    const sample = next.value;
    // The frame and the sample are separate handles on the same picture and
    // both have to be released; the frame's owner is the caller, so the sample
    // is closed here and now.
    const frame = sample.toVideoFrame();
    sample.close();
    return frame;
  }

  presentationSeconds(frame: VideoFrame): number {
    return frame.timestamp / 1_000_000 - this.facts.startSeconds;
  }

  close(): void {
    this.#closed = true;
    void this.#endRun();
  }

  /// Returns the current iterator, which is how mediabunny is told to free the
  /// decoder it opened for that run.
  async #endRun(): Promise<void> {
    const samples = this.#samples;
    this.#samples = undefined;
    await samples?.return();
  }
}
