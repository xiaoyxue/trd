import { beforeEach, describe, expect, test } from "bun:test";

import type { FrameReader } from "./frame-reader.ts";
import { type FrameSink, VideoPlayer } from "./player.ts";

/// A frame that only records whether it was closed. The player treats a
/// `VideoFrame` as a handle it must release, and that is the whole contract
/// these tests are about — a leaked frame holds a decoder-pool slot and stalls
/// decoding, which is invisible until playback stops.
interface FakeFrame {
  timestamp: number;
  closed: boolean;
  close(): void;
}

function fakeFrame(seconds: number): FakeFrame {
  return {
    timestamp: seconds * 1_000_000,
    closed: false,
    close() {
      this.closed = true;
    },
  };
}

/// A reader that hands out frames on a fixed grid, with hooks for the failure
/// modes: running out, and throwing.
class FakeReader implements FrameReader {
  readonly facts = {
    id: 1,
    codec: "avc1.640028",
    width: 1920,
    height: 1080,
    timescale: 25,
    sampleCount: 100,
    durationSeconds: 4,
    startSeconds: 0,
    lastFrameSeconds: 3.96,
    description: undefined,
  };
  readonly moovBytes = new Uint8Array(0);
  readonly handedOut: FakeFrame[] = [];
  /// Frames produced before `nextFrame` reports the end.
  available: number;
  /// When set, the next pull throws instead of producing a frame.
  failWith: string | undefined;
  #index = 0;

  constructor(available = 1000) {
    this.available = available;
  }

  async seekTo(seconds: number): Promise<number> {
    this.#index = Math.round(seconds / 0.04);
    return seconds;
  }

  async nextFrame(): Promise<VideoFrame | undefined> {
    if (this.failWith) {
      throw new Error(this.failWith);
    }
    if (this.#index >= this.available) {
      return undefined;
    }
    const frame = fakeFrame(this.#index * 0.04);
    this.#index += 1;
    this.handedOut.push(frame);
    return frame as unknown as VideoFrame;
  }

  presentationSeconds(frame: VideoFrame): number {
    return frame.timestamp / 1_000_000;
  }

  close(): void {}
}

function recordingSink() {
  const shown: number[] = [];
  const failures: string[] = [];
  let ended = 0;
  const sink: FrameSink = {
    present(frame, mediaSeconds) {
      shown.push(mediaSeconds);
      // The editor's sink closes the frame after the GPU copy; a test sink has
      // to do the same or it is not modelling the contract.
      frame.close();
    },
    ended() {
      ended += 1;
    },
    failed(message) {
      failures.push(message);
    },
  };
  return {
    sink,
    shown,
    failures,
    get ended() {
      return ended;
    },
  };
}

/// A controllable clock and animation-frame pump, so pacing is decided by the
/// test rather than by wall time.
function clock() {
  let now = 0;
  let pending: FrameRequestCallback[] = [];
  globalThis.performance = { now: () => now } as Performance;
  globalThis.requestAnimationFrame = ((callback: FrameRequestCallback) => {
    pending.push(callback);
    return pending.length;
  }) as typeof requestAnimationFrame;
  globalThis.cancelAnimationFrame = (() => {
    pending = [];
  }) as typeof cancelAnimationFrame;
  return {
    set(ms: number) {
      now = ms;
    },
    async tick(): Promise<void> {
      const due = pending;
      pending = [];
      for (const callback of due) {
        callback(now);
      }
      await settle();
    },
  };
}

/// Lets the player's queued `#fill` promises run to completion.
async function settle(): Promise<void> {
  for (let turn = 0; turn < 50; turn += 1) {
    await Promise.resolve();
  }
}

let time: ReturnType<typeof clock>;
beforeEach(() => {
  time = clock();
});

describe("VideoPlayer pacing", () => {
  test("shows the frame whose interval covers the clock", async () => {
    const reader = new FakeReader();
    const sink = recordingSink();
    const player = VideoPlayer.attach(reader, sink.sink);

    await player.seekToSeconds(0);
    expect(sink.shown).toEqual([0]);

    player.play();
    await settle();
    time.set(80);
    await time.tick();

    // 0.08s in: frame 2 is due, and it is the newest one that is.
    expect(sink.shown.at(-1)).toBeCloseTo(0.08, 5);
  });

  test("drops the frames it skipped past instead of leaking them", async () => {
    const reader = new FakeReader();
    const sink = recordingSink();
    const player = VideoPlayer.attach(reader, sink.sink);
    await player.seekToSeconds(0);
    player.play();
    await settle();

    // A quarter-second jump: eight frames become due at once. Showing them all
    // would fall further behind, so only the newest is shown — but every other
    // one must still be released.
    time.set(240);
    await time.tick();

    expect(sink.shown.at(-1)).toBeCloseTo(0.24, 5);
    // Precisely: every frame *older* than the one shown has been released.
    // Frames still ahead of the clock are legitimately open — that is the
    // lookahead — so counting those as leaks would pin the wrong thing.
    const shownAt = sink.shown.at(-1) ?? 0;
    const skipped = reader.handedOut.filter((frame) => frame.timestamp / 1_000_000 < shownAt);
    expect(skipped.length).toBeGreaterThan(1);
    expect(skipped.filter((frame) => !frame.closed)).toEqual([]);
  });

  test("ends once when the reader runs out, not on every frame after", async () => {
    const reader = new FakeReader(3);
    const sink = recordingSink();
    const player = VideoPlayer.attach(reader, sink.sink);
    await player.seekToSeconds(0);
    player.play();
    await settle();

    for (const at of [40, 80, 120, 160, 200]) {
      time.set(at);
      await time.tick();
    }

    expect(sink.ended).toBe(1);
    expect(player.playing).toBe(false);
  });

  test("reports a reader failure once rather than once per animation frame", async () => {
    // The regression this pins: `#fill` runs on every frame, so a reader that
    // throws — a decoder that has errored throws on every call — produced one
    // report per frame for as long as the tab stayed open.
    const reader = new FakeReader();
    const sink = recordingSink();
    const player = VideoPlayer.attach(reader, sink.sink);
    await player.seekToSeconds(0);
    player.play();
    await settle();

    reader.failWith = "decoder is closed";
    for (const at of [40, 80, 120, 160, 200, 240]) {
      time.set(at);
      await time.tick();
    }

    expect(sink.failures.length).toBe(1);
    expect(sink.failures[0]).toContain("decoder is closed");
  });

  test("a seek recovers a player that had run out", async () => {
    const reader = new FakeReader(2);
    const sink = recordingSink();
    const player = VideoPlayer.attach(reader, sink.sink);
    await player.seekToSeconds(0);
    player.play();
    await settle();
    for (const at of [40, 80, 120]) {
      time.set(at);
      await time.tick();
    }
    expect(sink.ended).toBe(1);

    // The end of the stream is a position, not a broken player: seeking back
    // has to make it usable again.
    reader.available = 100;
    await player.seekToSeconds(0.4);

    expect(sink.shown.at(-1)).toBeCloseTo(0.4, 5);
  });

  test("close releases the frames still queued", async () => {
    const reader = new FakeReader();
    const sink = recordingSink();
    const player = VideoPlayer.attach(reader, sink.sink);
    await player.seekToSeconds(0);
    player.play();
    await settle();

    player.close();

    // Everything the player was holding is released; what it already handed to
    // the sink is the sink's to close, and it did.
    const leaked = reader.handedOut.filter((frame) => !frame.closed);
    expect(leaked).toEqual([]);
  });

  test("pause stops advancing and play resumes from where it stopped", async () => {
    const reader = new FakeReader();
    const sink = recordingSink();
    const player = VideoPlayer.attach(reader, sink.sink);
    await player.seekToSeconds(0);
    player.play();
    await settle();
    time.set(120);
    await time.tick();
    const atPause = player.positionSeconds;
    player.pause();

    // Wall time passes while paused; it must not become media time.
    time.set(5_000);
    await time.tick();
    expect(player.positionSeconds).toBeCloseTo(atPause, 5);

    player.play();
    await settle();
    time.set(5_040);
    await time.tick();
    expect(player.positionSeconds).toBeGreaterThan(atPause);
  });
});
