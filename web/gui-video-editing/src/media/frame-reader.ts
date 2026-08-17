import type { VideoTrackFacts } from "./mp4-video.ts";

/// Where decoded frames come from, in presentation order, for a player that
/// does not care how they were demuxed.
///
/// The seam exists because two things are being separated. *Pacing* — which
/// frame is due against the clock, which are dropped, who closes them — is the
/// editor's policy and lives in `VideoPlayer`. *Demuxing and decoding* — byte
/// ranges, box parsing, decoder lifecycle, key-frame catch-up — is standard
/// media plumbing, and a library that already does it is preferable to a
/// hand-written one.
///
/// Implementations: `Mp4Video` (mp4box + `VideoDecoder`, written here) and
/// `MediabunnyReader` (delegated). Running both under the same player is what
/// makes them comparable on the same file.
export interface FrameReader {
  readonly facts: VideoTrackFacts;
  /// The `moov` box exactly as it appears in the file. Rust derives the
  /// timeline from it — the frame rate as the rational the sample table
  /// states — so every reader has to supply it, whatever it uses internally.
  readonly moovBytes: Uint8Array;

  /// Positions at the frame covering `seconds`, measured from the start of the
  /// video. Returns the time actually sought to, after clamping.
  seekTo(seconds: number): Promise<number>;

  /// The next frame in presentation order, or `undefined` at the end.
  ///
  /// **The caller owns the frame and must `close()` it.** A `VideoFrame` holds
  /// a slot in a small decoder-side pool, so leaking one stalls decoding.
  nextFrame(): Promise<VideoFrame | undefined>;

  /// Presentation time of `frame` measured from the start of the video, which
  /// is what a timeline index is derived from — not the raw container `cts`.
  presentationSeconds(frame: VideoFrame): number;

  close(): void;
}
