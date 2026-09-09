import { describe, expect, test } from "bun:test";
import { createFrameClock } from "../src/frame-clock";

describe("stream playback clock", () => {
  test("starts at row zero even when the first RAF timestamp precedes setup", () => {
    const frame = createFrameClock(24, 72);
    expect(frame(0)).toBe(0);
    expect(frame(125)).toBe(3);
  });

  test("does not count time spent preparing the renderer", () => {
    const frame = createFrameClock(24, 72);
    expect(frame(1_000_000)).toBe(0);
    expect(frame(1_000_125)).toBe(3);
  });

  test("keeps one row throughout its interval", () => {
    const frame = createFrameClock(10, 30);
    expect(frame(50)).toBe(0);
    expect(frame(149)).toBe(0);
    expect(frame(150)).toBe(1);
  });

  test("uses elapsed RAF time and loops without accumulating callback drift", () => {
    const frame = createFrameClock(24, 72);
    expect(frame(1_000)).toBe(0);
    expect(frame(2_500)).toBe(36);
    expect(frame(4_000)).toBe(0);
    expect(frame(11_000)).toBe(24);
  });

  test("gives independently initialized renderers the same row sequence", () => {
    const canvas = createFrameClock(24, 72);
    const offscreen = createFrameClock(24, 72);
    for (const elapsed of [0, 125, 1_500, 3_000]) {
      expect(canvas(500 + elapsed)).toBe(offscreen(8_000 + elapsed));
    }
  });

  test("keeps a single-row stream at zero", () => {
    const frame = createFrameClock(24, 1);
    expect(frame(0)).toBe(0);
    expect(frame(10_000)).toBe(0);
  });
});
