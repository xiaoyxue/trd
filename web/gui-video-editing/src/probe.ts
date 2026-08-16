// WebCodecs decode probe (#282): can mp4box.js + `VideoDecoder` open this
// project's videos over *range reads* — a local file or a URL — and land on an
// exact frame without transferring the whole thing?
//
// Deliberately standalone: it touches none of the editor's code, so a failure
// here is cheap. What it shares with the editor is the part that must be right,
// `src/media/` — the byte source and the ranged MP4 reader the playback path
// will be rebuilt on.
//
// The number to watch is "read": opening a multi-gigabyte video must cost
// megabytes, and so must each seek.
import { type ByteSource, fileByteSource, urlByteSource } from "./media/byte-source.ts";
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
const auto = new URLSearchParams(location.search);
seekInput.value = auto.get("seek") ?? "0";
const autoUrl = auto.get("url");
if (autoUrl) {
  (document.getElementById("url") as HTMLInputElement).value = autoUrl;
  run(() => urlByteSource(autoUrl));
} else {
  log("pick a local MP4, or enter a URL and press Decode from URL");
}
