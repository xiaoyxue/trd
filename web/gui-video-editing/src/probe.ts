// WebCodecs decode probe (#282): can this project's videos be opened over
// *range reads* — a local file or a URL — and seeked to an exact frame without
// transferring the whole thing?
//
// Standalone: it touches none of the editor's code, sharing only `src/media/`.
// The number to watch is "read": opening a multi-gigabyte video must cost
// megabytes, and so must each seek.
import { type ByteSource, fileByteSource, urlByteSource } from "./media/byte-source.ts";
import type { FrameReader } from "./media/frame-reader.ts";
import { MediabunnyReader } from "./media/mediabunny-reader.ts";
import { Mp4Video } from "./media/mp4-video.ts";

const logElement = document.getElementById("log") as HTMLDivElement;
const canvas = document.getElementById("preview") as HTMLCanvasElement;
const context = canvas.getContext("2d");
const seekInput = document.getElementById("seek") as HTMLInputElement;

function log(message: string): void {
  logElement.textContent += `${message}\n`;
}

function mib(bytes: number): string {
  return `${(bytes / 1_048_576).toFixed(2)} MiB`;
}

async function probe(source: ByteSource): Promise<void> {
  logElement.textContent = "";
  const openedAt = performance.now();
  log(`source: ${source.label} (${mib(source.size)})`);

  const video = await Mp4Video.open(source);
  const facts = video.facts;
  // Opening reads only the boxes it needs. For a `moov` at the end of the file
  // that means the header, then a jump straight past the `mdat`.
  log(
    `opened in ${(performance.now() - openedAt).toFixed(0)}ms, ` +
      `read ${mib(source.bytesRead)} of ${mib(source.size)} ` +
      `(${((100 * source.bytesRead) / source.size).toFixed(3)}%)`,
  );
  log(
    `track ${facts.id}: ${facts.codec} ${facts.width}x${facts.height}, ` +
      `${facts.sampleCount} samples, timescale ${facts.timescale}, ` +
      `${facts.durationSeconds.toFixed(3)}s (first frame at ${facts.startSeconds.toFixed(4)}s, ` +
      `last at ${facts.lastFrameSeconds.toFixed(3)}s)`,
  );
  log(
    facts.description
      ? `description: ${facts.description.byteLength} bytes [${[...facts.description.slice(0, 8)]
          .map((b) => b.toString(16).padStart(2, "0"))
          .join(" ")}]`
      : "description: NONE — an AVC/HEVC decoder cannot decode without it",
  );

  const seconds = Number(seekInput.value);
  const target = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  const requested = Number(new URLSearchParams(location.search).get("frames") ?? 4);
  const frames = Number.isFinite(requested) && requested > 0 ? Math.floor(requested) : 4;
  log(`seeking to ${target.toFixed(3)}s, pulling ${frames} frames…`);
  const seekAt = performance.now();
  let firstDrawn = false;
  let previous = Number.NaN;
  let maxGap = 0;
  const report = await video.seekAndDecode(target, frames, (frame) => {
    if (context && !firstDrawn) {
      // Only to prove a real image came out; the editor uploads frames to the
      // GPU instead of drawing them here.
      canvas.width = frame.displayWidth;
      canvas.height = frame.displayHeight;
      context.drawImage(frame, 0, 0);
      firstDrawn = true;
    }
    // Presentation order and no gaps is what playback depends on; a dropped or
    // reordered frame shows up here as an irregular step.
    const seconds = video.presentationSeconds(frame);
    if (Number.isFinite(previous)) {
      maxGap = Math.max(maxGap, seconds - previous);
    }
    previous = seconds;
    if (frames <= 8) {
      log(
        `  frame pts=${seconds.toFixed(4)}s coded=${frame.codedWidth}x${frame.codedHeight} ` +
          `display=${frame.displayWidth}x${frame.displayHeight} ` +
          `visible=${frame.visibleRect?.width}x${frame.visibleRect?.height}`,
      );
    }
    frame.close();
  });
  const elapsed = performance.now() - seekAt;
  log(
    `seek to ${target.toFixed(3)}s${
      report.target === target ? "" : ` (past the end — clamped to ${report.target.toFixed(3)}s)`
    }: landed at ${report.firstTime.toFixed(4)}s in ` +
      `${elapsed.toFixed(0)}ms, delivered ${report.delivered}, ` +
      `decoded-and-dropped ${report.skipped} to reach the key frame, ` +
      `read ${mib(report.bytesRead)}`,
  );
  if (report.delivered > 1) {
    log(
      `pull: ${report.delivered} frames, largest pts step ${maxGap.toFixed(4)}s, ` +
        `${((report.delivered * 1000) / elapsed).toFixed(1)} frames/s end to end`,
    );
  }
  log(`total read: ${mib(source.bytesRead)} of ${mib(source.size)}`);
  video.close();
}

/// Repeated seeks against **one** reader, which is what the editor does and
/// what a single-seek probe run structurally cannot reach: a decoder is
/// configured once and reused, so a fault that only appears on the second and
/// later seek — a reset racing a feed, or a decoder left `closed` by an earlier
/// error — is invisible until the same instance is seeked twice.
///
/// `overlap` fires the seeks without awaiting the previous one, the way dragging
/// a scrubber does. Taking a [`FrameReader`] rather than a `ByteSource` is what
/// lets the hand-written and the delegated reader be measured by the same run.
async function scrub(
  video: FrameReader,
  label: string,
  targets: number[],
  overlap: boolean,
): Promise<void> {
  logElement.textContent = "";
  log(`source: ${label}`);
  log(
    `opened: ${video.facts.codec} ${video.facts.width}x${video.facts.height}, ` +
      `${video.facts.durationSeconds.toFixed(1)}s`,
  );
  log(`scrubbing ${targets.length} targets on one reader, overlap=${overlap}`);

  let failures = 0;
  // Under `overlap` every pull is expected to deliver the *winning* seek's
  // frames, because a scrubber only cares where the drag ended. What must not
  // happen is a throw, or a reader that stops working afterwards.
  const expected = (target: number) => (overlap ? (targets[targets.length - 1] ?? target) : target);
  const step = async (target: number, index: number): Promise<void> => {
    const startedAt = performance.now();
    try {
      await video.seekTo(target);
      const frame = await video.nextFrame();
      const elapsed = performance.now() - startedAt;
      if (!frame) {
        failures += 1;
        log(
          `  ${index}: seek ${target.toFixed(1)}s → no frame after ${elapsed.toFixed(0)}ms  ← MISSING`,
        );
        return;
      }
      const landed = video.presentationSeconds(frame);
      frame.close();
      // A tolerance of a second covers the consecutive frames handed to
      // coalesced pulls; it is far tighter than the 10s key-frame interval, so
      // landing in the wrong GOP still fails.
      const drift = Math.abs(landed - expected(target));
      if (drift > 1) {
        failures += 1;
      }
      log(
        `  ${index}: seek ${target.toFixed(1)}s → landed ${landed.toFixed(3)}s ` +
          `in ${elapsed.toFixed(0)}ms${drift > 1 ? "  ← WRONG FRAME" : ""}`,
      );
    } catch (error) {
      failures += 1;
      log(`  ${index}: seek ${target.toFixed(1)}s → THREW ${String(error)}`);
    }
  };

  if (overlap) {
    await Promise.all(targets.map(step));
    // Whatever the overlap did to the intermediate seeks, the reader has to
    // still work afterwards — that is the property a wedged decoder breaks.
    const last = targets[targets.length - 1] ?? 0;
    await step(last, targets.length);
  } else {
    for (const [index, target] of targets.entries()) {
      await step(target, index);
    }
  }
  log(`${failures === 0 ? "OK" : `${failures} FAILED`} — reader usable after the run`);
  video.close();
}

function run(open: () => Promise<ByteSource>): void {
  void open()
    .then(probe)
    .catch((error: unknown) => log(`ERROR: ${String(error)}`));
}

document.getElementById("file")?.addEventListener("change", (event) => {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (file) {
    run(() => Promise.resolve(fileByteSource(file)));
  }
});

document.getElementById("load")?.addEventListener("click", () => {
  const url = (document.getElementById("url") as HTMLInputElement).value.trim();
  if (url) {
    run(() => urlByteSource(url));
  }
});

// `?url=…&seek=…` runs the whole probe without a click, so a result is a
// command rather than a click-through — repeatable, and checkable from a
// driven browser. `?seek=` alone pre-fills the target for a local pick.
// `?scrub=t1,t2,…` instead seeks repeatedly on one reader, and `?overlap=1`
// fires those seeks without awaiting the previous one — the shape a dragged
// scrubber produces. `?reader=mediabunny` runs the scrub against the delegated
// reader instead of the hand-written one, on the same file and the same targets.
const auto = new URLSearchParams(location.search);
seekInput.value = auto.get("seek") ?? "0";
const autoUrl = auto.get("url");
const scrubTargets = (auto.get("scrub") ?? "")
  .split(",")
  .map((value) => value.trim())
  // Without this an absent `?scrub=` splits to `[""]`, and `Number("")` is 0 —
  // a silent scrub to the start that replaces the `?seek=`/`?frames=` run.
  .filter((value) => value.length > 0)
  .map(Number)
  .filter((value) => Number.isFinite(value));

/// Opens whichever reader `?reader=` names, timing it: how long a multi-gigabyte
/// file takes to become seekable is the first thing that separates the two.
async function openReader(url: string): Promise<{ reader: FrameReader; label: string }> {
  const startedAt = performance.now();
  if (auto.get("reader") === "mediabunny") {
    const reader = await MediabunnyReader.open({ kind: "url", url });
    return {
      reader,
      label: `${url} · mediabunny · opened in ${(performance.now() - startedAt).toFixed(0)}ms`,
    };
  }
  const source = await urlByteSource(url);
  const reader = await Mp4Video.open(source);
  return {
    reader,
    label:
      `${url} (${mib(source.size)}) · mp4box · opened in ${(performance.now() - startedAt).toFixed(0)}ms, ` +
      `read ${mib(source.bytesRead)}`,
  };
}

if (autoUrl) {
  (document.getElementById("url") as HTMLInputElement).value = autoUrl;
  if (scrubTargets.length > 0) {
    void openReader(autoUrl)
      .then(({ reader, label }) => scrub(reader, label, scrubTargets, auto.get("overlap") === "1"))
      .catch((error: unknown) => log(`ERROR: ${String(error)}`));
  } else {
    run(() => urlByteSource(autoUrl));
  }
} else {
  log("pick a local MP4, or enter a URL and press Decode from URL");
}
