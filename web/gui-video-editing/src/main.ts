import uffiziEnvUrl from "../../../assets/envmap/uffizi-large.hdr" with { type: "file" };
import cokeTextureUrl from "../../../assets/meshes/can/can_around.jpg" with { type: "file" };
import cokeObjUrl from "../../../assets/meshes/can/coke.obj" with { type: "file" };
import dragonUrl from "../../../assets/meshes/glb/Meshy_AI_Dragon_0804104424_texture.glb" with {
  type: "file",
};
import beerObjUrl from "../../../assets/meshes/qd_beer/source/3d66.com_JDH5455878326.obj" with {
  type: "file",
};
import beerTextureUrl from "../../../assets/meshes/qd_beer/textures/3d66-export-JDH5455878326-001.jpg" with {
  type: "file",
};
import init, { startVideoEditing, startVideoEditingSynthetic } from "../pkg/trd_wasm.js";
import wasmUrl from "../pkg/trd_wasm_bg.wasm" with { type: "file" };

// Frame-numbering grid used only until the container's own metadata is read
// (see `readMoov`). `<video>` never reports a frame rate, so a clip whose `moov`
// cannot be parsed keeps this grid: playback and seeking stay self-consistent
// because both directions of the media-time conversion share it, but the
// displayed frame *numbers* are synthetic.
const VIRTUAL_FPS = 30;

/// Reads the `moov` box out of an MP4 without loading the file.
///
/// `moov` may sit at either end (before `mdat` for "faststart" files, after it
/// otherwise), so this walks the top-level boxes reading only their 8/16-byte
/// headers and slices out just the one box. A multi-gigabyte clip therefore
/// costs a handful of small reads rather than a full read into memory.
async function readMoov(file: Blob): Promise<Uint8Array | undefined> {
  const header = async (at: number): Promise<DataView | undefined> => {
    if (at + 8 > file.size) {
      return undefined;
    }
    return new DataView(await file.slice(at, at + 16).arrayBuffer());
  };
  for (let offset = 0; offset < file.size; ) {
    const view = await header(offset);
    if (!view || view.byteLength < 8) {
      return undefined;
    }
    const small = view.getUint32(0);
    const kind = String.fromCharCode(
      view.getUint8(4),
      view.getUint8(5),
      view.getUint8(6),
      view.getUint8(7),
    );
    // `1` means a 64-bit size follows the type; `0` means "to end of file".
    let size = small;
    if (small === 1) {
      if (view.byteLength < 16) {
        return undefined;
      }
      size = Number(view.getBigUint64(8));
    } else if (small === 0) {
      size = file.size - offset;
    }
    if (size < 8) {
      return undefined;
    }
    if (kind === "moov") {
      return new Uint8Array(await file.slice(offset, offset + size).arrayBuffer());
    }
    offset += size;
  }
  return undefined;
}


async function main(): Promise<void> {
  await init({ module_or_path: wasmUrl });
  const canvas = document.getElementById("video-editing-canvas");
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new Error("missing #video-editing-canvas");
  }
  const query = new URLSearchParams(location.search);
  // TOY BRANCH: a document is optional. Without `?document=` the editor starts
  // on a synthesized video-only timeline, and the real timeline is re-derived
  // from each video that gets opened.
  const documentUrl = query.get("document");
  let editor: Awaited<ReturnType<typeof startVideoEditing>>;
  if (documentUrl) {
    const response = await fetch(documentUrl);
    if (!response.ok) {
      throw new Error(
        `failed to fetch editing document "${documentUrl}": ${response.status} ${response.statusText}`,
      );
    }
    editor = await startVideoEditing(canvas, new Uint8Array(await response.arrayBuffer()));
  } else {
    // A one-second placeholder; the real span is set from the clip's duration on
    // `loadedmetadata` (see `setVideoTimeline`).
    editor = await startVideoEditingSynthetic(canvas, 1280, 720, VIRTUAL_FPS, 1, VIRTUAL_FPS);
  }
  const catalog = new Map<number, { modelUrl: string; textureUrl?: string }>([
    [1, { modelUrl: cokeObjUrl, textureUrl: cokeTextureUrl }],
    [2, { modelUrl: beerObjUrl, textureUrl: beerTextureUrl }],
    [3, { modelUrl: dragonUrl }],
  ]);
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "video/mp4";
  input.hidden = true;
  document.body.append(input);

  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.crossOrigin = "anonymous";
  video.hidden = true;
  document.body.append(video);
  // Background frames are copied straight from this element on the GPU; see
  // `presentCurrentFrame`.
  editor.setVideoElement(video);

  let objectUrl: string | undefined;
  let callbackActive = false;
  let callbackId: number | undefined;
  let loadingAsset = false;
  let envBytesPromise: Promise<Uint8Array> | undefined;
  let sourceReady = false;
  let sourceGeneration = 0;

  // Only media-element state transitions. `mediaTime` is not published here: it
  // travels with its own frame through `presentVideoFrame`, so the Details
  // timeline always describes the frame that reached the screen.
  function syncMediaState(): void {
    editor.setVideoMediaState(video.readyState, video.ended);
  }

  // The decoded frame already lives in GPU memory, and so does the render
  // target, so it is copied GPU→GPU by `copy_external_image_to_texture` on the
  // Rust side. `VideoFrame.copyTo` is deliberately *not* used: it would pull the
  // frame down to the CPU (with a YUV→RGBA conversion), cross the wasm boundary,
  // and get pushed back up — three full-resolution traversals of a frame that
  // never needed to leave the GPU. Only the frame's identity crosses now.
  //
  // `presentVideoFrame` is synchronous, so it also removes the await that used
  // to sit between the callback firing and the frame reaching the renderer.
  function presentCurrentFrame(mediaTime: number, generation = sourceGeneration): void {
    if (generation !== sourceGeneration) {
      return;
    }
    editor.presentVideoFrame(
      video.videoWidth,
      video.videoHeight,
      editor.frameIndexAtMediaTime(mediaTime),
      mediaTime,
    );
  }

  function scheduleVideoFrame(): void {
    if (callbackActive) {
      return;
    }
    callbackActive = true;
    callbackId = video.requestVideoFrameCallback((_now, metadata) => {
      const generation = sourceGeneration;
      callbackId = undefined;
      callbackActive = false;
      void Promise.resolve()
        .then(() => presentCurrentFrame(metadata.mediaTime, generation))
        .catch((error: unknown) => editor.setVideoError(String(error)))
        .finally(() => {
          if (!video.paused && !video.ended) {
            scheduleVideoFrame();
          }
        });
    });
  }

  video.addEventListener("play", () => {
    editor.setVideoStatus(sourceReady, sourceReady);
    syncMediaState();
    scheduleVideoFrame();
  });
  video.addEventListener("pause", () => {
    editor.setVideoStatus(sourceReady, false);
    syncMediaState();
  });
  video.addEventListener("ended", () => {
    editor.setVideoStatus(sourceReady, false);
    syncMediaState();
  });
  video.addEventListener("error", () => {
    sourceReady = false;
    editor.setVideoStatus(false, false);
    syncMediaState();
    editor.setVideoError(video.error?.message ?? "failed to load video");
  });
  video.addEventListener("seeked", () => {
    if (video.paused) {
      presentCurrentFrame(video.currentTime);
    }
  });

  function loadVideoSource(
    source: string,
    localFile?: { filename: string; byteLength: number; blob?: Blob },
  ): void {
    if (localFile) {
      editor.validateVideoFile(localFile.filename, localFile.byteLength);
    }
    const generation = ++sourceGeneration;
    if (callbackId !== undefined) {
      video.cancelVideoFrameCallback(callbackId);
      callbackId = undefined;
      callbackActive = false;
    }
    video.pause();
    sourceReady = false;
    editor.setVideoStatus(false, false);
    editor.setVideoSourceInfo(
      localFile ? 1 : 2,
      localFile?.filename ?? source,
      localFile?.byteLength ?? -1,
    );
    video.src = source;
    video.addEventListener(
      "loadeddata",
      () => {
        if (generation !== sourceGeneration) {
          return;
        }
        try {
          editor.validateVideoMetadata(video.videoWidth, video.videoHeight, video.duration);
          // Prefer the container's own numbers: `<video>` reports no frame rate,
          // so a virtual grid would number a 25 fps clip as if it were 30 and
          // every displayed frame number would be fiction. Reading `moov` gives
          // the same facts `ffprobe` gives the native shell.
          void (async () => {
            if (localFile?.blob) {
              try {
                const moov = await readMoov(localFile.blob);
                if (moov && generation === sourceGeneration) {
                  editor.setVideoTimelineFromMoov(moov);
                  return;
                }
              } catch (error) {
                console.warn("moov probe failed; using the virtual grid", error);
              }
            }
            // No container to read (a remote URL) or an unparsable one: fall
            // back to the virtual grid, which still spans the right duration.
            if (generation === sourceGeneration) {
              editor.setVideoTimeline(
                video.videoWidth,
                video.videoHeight,
                VIRTUAL_FPS,
                1,
                video.duration,
              );
            }
          })();
          video.pause();
          video.currentTime = 0;
          sourceReady = true;
          editor.setVideoStatus(true, false);
          syncMediaState();
          presentCurrentFrame(0);
        } catch (error) {
          sourceReady = false;
          editor.setVideoStatus(false, false);
          editor.setVideoError(String(error));
        }
      },
      { once: true },
    );
    video.load();
  }

  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (!file) {
      return;
    }
    try {
      editor.validateVideoFile(file.name, file.size);
    } catch (error) {
      editor.setVideoError(String(error));
      input.value = "";
      return;
    }
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
    }
    objectUrl = URL.createObjectURL(file);
    loadVideoSource(objectUrl, { filename: file.name, byteLength: file.size, blob: file });
  });

  function serviceRustCommands(): void {
    const command = editor.takeCommand();
    if (command === 1) {
      input.value = "";
      input.click();
    } else if (command === 2) {
      void video.play().catch((error: unknown) => editor.setVideoError(String(error)));
    } else if (command === 3) {
      video.pause();
    }
    const requestedVideoUrl = editor.takeVideoUrlRequest();
    if (requestedVideoUrl) {
      try {
        const url = new URL(requestedVideoUrl);
        if (url.protocol !== "http:" && url.protocol !== "https:") {
          throw new Error("video URL must use http:// or https://");
        }
        if (objectUrl) {
          URL.revokeObjectURL(objectUrl);
          objectUrl = undefined;
        }
        loadVideoSource(url.href);
      } catch (error) {
        editor.setVideoError(String(error));
      }
    }
    const seekFrame = editor.takeSeekFrame();
    if (seekFrame >= 0 && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      video.currentTime = editor.mediaTimeAtFrame(seekFrame);
    }
    const assetCode = loadingAsset ? 0 : editor.takeAssetRequest();
    const entry = catalog.get(assetCode);
    if (entry) {
      loadingAsset = true;
      envBytesPromise ??= fetch(uffiziEnvUrl).then(async (response) => {
        if (!response.ok) {
          throw new Error(`failed to fetch Uffizi environment: ${response.status}`);
        }
        return new Uint8Array(await response.arrayBuffer());
      });
      void Promise.all([
        fetch(entry.modelUrl).then(async (response) => {
          if (!response.ok) {
            throw new Error(`failed to fetch catalog model: ${response.status}`);
          }
          return new Uint8Array(await response.arrayBuffer());
        }),
        entry.textureUrl
          ? fetch(entry.textureUrl).then(async (response) => {
              if (!response.ok) {
                throw new Error(`failed to fetch catalog texture: ${response.status}`);
              }
              return new Uint8Array(await response.arrayBuffer());
            })
          : Promise.resolve(new Uint8Array()),
        envBytesPromise,
      ])
        .then(([modelBytes, textureBytes, envBytes]) =>
          editor.loadCatalogAsset(assetCode, modelBytes, textureBytes, envBytes),
        )
        .catch((error: unknown) => editor.setVideoError(String(error)))
        .finally(() => {
          loadingAsset = false;
        });
    }
    requestAnimationFrame(serviceRustCommands);
  }
  requestAnimationFrame(serviceRustCommands);
}

main().catch((error: unknown) => {
  console.error("video editing failed:", error);
  document.body.innerHTML = `<pre style="color:#f88;padding:1rem">${String(error)}</pre>`;
});
