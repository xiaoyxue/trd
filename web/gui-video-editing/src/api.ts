/** One complete 0.0.7 document, optionally supplied as ordered byte parts. */
export type ArrowInput = Uint8Array | readonly Uint8Array[];

export interface VideoEditorApi {
  loadArrow(input: ArrowInput): Promise<void>;
  resetState(): Promise<void>;
  exportArrow(): Promise<Uint8Array>;
  seekToSeconds(seconds: number): Promise<void>;
}

/** Host I/O only. The backend retains all Arrow validation and rendering in Rust. */
export interface VideoEditorBackend {
  loadArrow(bytes: Uint8Array): Promise<void>;
  resetState(): Promise<void>;
  exportArrow(): Promise<Uint8Array>;
  seekToSeconds(seconds: number): Promise<void>;
}

declare global {
  interface Window {
    /** Installed by the demo bootstrap; rejects if mounting the editor fails. */
    trdVideoEditorReady: Promise<VideoEditorApi>;
  }
}

function snapshotInput(input: ArrowInput): Uint8Array {
  const parts = input instanceof Uint8Array ? [input] : input;
  if (!Array.isArray(parts) || parts.length === 0) {
    throw new TypeError("loadArrow requires a complete Arrow document");
  }
  let length = 0;
  for (const part of parts) {
    if (!(part instanceof Uint8Array)) {
      throw new TypeError("Arrow input parts must be Uint8Array values");
    }
    length += part.byteLength;
  }
  if (length === 0 || !Number.isSafeInteger(length)) {
    throw new RangeError("Arrow input must have a nonzero, representable byte length");
  }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    bytes.set(part, offset);
    offset += part.byteLength;
  }
  return bytes;
}

/** Serializes host operations and snapshots input before queued callers can mutate it. */
export function createVideoEditorApi(backend: VideoEditorBackend): VideoEditorApi {
  let tail: Promise<unknown> = Promise.resolve();
  function enqueue<T>(operation: () => Promise<T>): Promise<T> {
    // Each caller receives its rejection; a failed load must not poison later resets/loads.
    const result = tail.then(operation, operation);
    tail = result;
    return result;
  }
  return {
    async loadArrow(input) {
      const bytes = snapshotInput(input);
      await enqueue(() => backend.loadArrow(bytes));
    },
    resetState: () => enqueue(() => backend.resetState()),
    exportArrow: () => enqueue(() => backend.exportArrow()),
    async seekToSeconds(seconds) {
      if (!Number.isFinite(seconds) || seconds < 0) {
        throw new RangeError("seek time must be finite and nonnegative");
      }
      await enqueue(() => backend.seekToSeconds(seconds));
    },
  };
}
