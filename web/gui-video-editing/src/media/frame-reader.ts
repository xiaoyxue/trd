import type { VideoTrackFacts } from "./mp4-video.ts";

/// Where decoded frames come from, in presentation order, for a player that
/// does not care how they were demuxed.
///
/// The seam separates *pacing* — which frame is due, which are dropped, who
/// closes them, all `VideoPlayer`'s policy — from *demuxing and decoding*.
/// Implementations: `Mp4Video` (hand-written) and `MediabunnyReader`.
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
