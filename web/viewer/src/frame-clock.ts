export function createFrameClock(fps: number, frameCount: number): (timestamp: number) => number {
  let start: number | undefined;
  return (timestamp) => {
    // The first RAF timestamp can precede performance.now() during asynchronous startup.
    start ??= timestamp;
    return Math.floor(((timestamp - start) / 1000) * fps) % frameCount;
  };
}
