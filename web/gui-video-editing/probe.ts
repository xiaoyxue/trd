// Minimal WebCodecs decode probe (#282): does mp4box.js + `VideoDecoder` decode
// the videos this project actually uses, from a local file *and* from a URL?
//
// Deliberately standalone — it touches none of the editor's code. The point is
// to settle the mechanism (demux → configure → decode → a frame on screen)
// before the playback path is rebuilt on it, so a failure here is cheap.
import { createFile, type MP4ArrayBuffer, type MP4File, type MP4Sample } from "mp4box";

const logElement = document.getElementById("log") as HTMLDivElement;
const canvas = document.getElementById("preview") as HTMLCanvasElement;
const context = canvas.getContext("2d");

function log(message: string): void {
  logElement.textContent += `${message}\n`;
}

/// mp4box wants `ArrayBuffer`s tagged with their offset in the file, so it can
/// be fed a stream in pieces rather than the whole source at once — which is the
/// property that matters for a multi-gigabyte video.
function tagged(buffer: ArrayBuffer, offset: number): MP4ArrayBuffer {
  const tagged = buffer as MP4ArrayBuffer;
  tagged.fileStart = offset;
  return tagged;
}

async function decodeFirstFrames(source: Blob | string, wanted = 8): Promise<void> {
  logElement.textContent = "";
  const started = performance.now();
  const file: MP4File = createFile();
  let decoder: VideoDecoder | undefined;
  let decoded = 0;
  let queued = 0;
  let drawn = false;

  const done = new Promise<void>((resolve, reject) => {
    file.onError = (error: string) => reject(new Error(error));

    file.onReady = (info) => {
      const track = info.videoTracks[0];
      if (!track) {
        reject(new Error("no video track"));
        return;
      }
      log(
        `track: ${track.codec} ${track.video.width}x${track.video.height}, ` +
          `${track.nb_samples} samples, timescale ${track.timescale}`,
      );
      // The timeline mp4_probe currently computes in Rust comes free here.
      log(`duration: ${(track.duration / track.timescale).toFixed(3)}s`);

      decoder = new VideoDecoder({
        output: (frame) => {
          decoded += 1;
          if (!drawn && context) {
            // Only to prove a real image came out; the editor uploads frames to
            // the GPU instead of drawing them here.
            canvas.width = frame.displayWidth;
            canvas.height = frame.displayHeight;
            context.drawImage(frame, 0, 0);
            drawn = true;
          }
          if (decoded <= 3) {
            log(
              `frame ${decoded}: pts=${frame.timestamp}us ${frame.codedWidth}x${frame.codedHeight}`,
            );
          }
          frame.close();
          if (decoded >= wanted) {
            resolve();
          }
        },
        error: (error) => {
          log(`decoder error: ${String(error)}`);
          reject(error);
        },
      });

      // `description` is the avcC/hvcC payload; mp4box extracts it, which is the
      // fiddly part of configuring a decoder by hand. Without it an AVC decoder
      // accepts the configuration and then silently produces nothing, so its
      // absence is worth saying out loud.
      const description = descriptionFor(file, track.id);
      log(
        description
          ? `description: ${description.byteLength} bytes`
          : "description: NONE — an AVC/HEVC decoder cannot decode without it",
      );
      decoder.configure({
        codec: track.codec,
        codedWidth: track.video.width,
        codedHeight: track.video.height,
        ...(description ? { description } : {}),
      });
      log(`decoder: ${decoder.state}`);

      file.setExtractionOptions(track.id, null, { nbSamples: wanted });
      file.start();
    };

    file.onSamples = (_id: number, _user: unknown, samples: MP4Sample[]) => {
      log(`samples: +${samples.length}`);
      for (const sample of samples) {
        queued += 1;
        decoder?.decode(
          new EncodedVideoChunk({
            type: sample.is_sync ? "key" : "delta",
            // Presentation time in microseconds. mp4box hands over cts/timescale
            // already reconciled, which is exactly the B-frame reordering a
            // hand-written demuxer gets wrong.
            timestamp: (sample.cts * 1_000_000) / sample.timescale,
            duration: (sample.duration * 1_000_000) / sample.timescale,
            data: sample.data,
          }),
        );
      }
    };
  });

  if (typeof source === "string") {
    // Fetched whole, exactly like the local path, because that is what this
    // probe is for: whether mp4box + WebCodecs decode our videos, not whether a
    // streaming loop is right.
    //
    // Streaming needs more than a loop here: this file has `moov` at the *end*
    // (its first sample sits at offset 48, so `mdat` comes first), so nothing
    // can be extracted until the whole file has arrived anyway. A multi-gigabyte
    // source needs range reads that find `moov` first — W2's problem, and one
    // `serve-documents.ts` already supports.
    const response = await fetch(source);
    if (!response.ok) {
      throw new Error(`fetch failed: ${response.status} ${response.statusText}`);
    }
    const buffer = await response.arrayBuffer();
    log(`fetched ${(buffer.byteLength / 1_048_576).toFixed(1)} MiB`);
    file.appendBuffer(tagged(buffer, 0));
  } else {
    const buffer = await source.arrayBuffer();
    file.appendBuffer(tagged(buffer, 0));
  }
  file.flush();
  if (decoded === 0) {
    log(`fed ${queued} chunks, decoder queue ${decoder?.decodeQueueSize ?? 0}`);
  }
  await done;
  log(`decoded ${decoded} frames in ${(performance.now() - started).toFixed(0)}ms`);
}
/// Pulls the codec-specific configuration box out of the sample description —
/// `VideoDecoder.configure` needs it for AVC/HEVC.
function descriptionFor(file: MP4File, trackId: number): Uint8Array | undefined {
  const track = file.getTrackById(trackId);
  for (const entry of track?.mdia?.minf?.stbl?.stsd?.entries ?? []) {
    const box = entry.avcC ?? entry.hvcC ?? entry.vpcC ?? entry.av1C;
    if (box) {
      const stream = new DataStream(undefined, 0, DataStream.BIG_ENDIAN);
      box.write(stream);
      // Strip the 8-byte box header the writer emits.
      return new Uint8Array(stream.buffer, 8);
    }
  }
  return undefined;
}

// mp4box exposes `DataStream` on the module namespace rather than as a named
// export in every build, so reach it through the import that definitely exists.
const { DataStream } = await import("mp4box");

document.getElementById("file")?.addEventListener("change", (event) => {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (file) {
    void decodeFirstFrames(file).catch((error: unknown) => log(`ERROR: ${String(error)}`));
  }
});

document.getElementById("load")?.addEventListener("click", () => {
  const url = (document.getElementById("url") as HTMLInputElement).value.trim();
  if (url) {
    void decodeFirstFrames(url).catch((error: unknown) => log(`ERROR: ${String(error)}`));
  }
});

log("pick a local MP4, or enter a URL and press Decode from URL");
