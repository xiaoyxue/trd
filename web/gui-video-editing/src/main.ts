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
import editingDocumentUrl from "../data/fiba-shot1.arrow" with { type: "file" };
import init, { startVideoEditing } from "../pkg/trd_wasm.js";
import wasmUrl from "../pkg/trd_wasm_bg.wasm" with { type: "file" };

/// Locates the `moov` box with small range reads and hands it to Rust.
///
/// `moov` may sit at either end of a file, and this one may be gigabytes, so the
/// box list is walked from the front reading only 16-byte headers — never the
/// `mdat` payload in between (#264).
async function readMoov(file: File): Promise<Uint8Array | undefined> {
  let offset = 0;
  while (offset + 8 <= file.size) {
    const header = new DataView(await file.slice(offset, offset + 16).arrayBuffer());
    if (header.byteLength < 8) {
      return undefined;
    }
    const size32 = header.getUint32(0);
    const kind = String.fromCharCode(
      header.getUint8(4),
      header.getUint8(5),
      header.getUint8(6),
      header.getUint8(7),
    );
    // `size === 1` puts a 64-bit length after the type; `size === 0` runs to EOF.
    let size = size32;
    if (size32 === 1) {
      if (header.byteLength < 16) {
        return undefined;
      }
      size = Number(header.getBigUint64(8));
    } else if (size32 === 0) {
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
  // The annotation document is optional (#264). `?document=none` opens the
  // editor as a plain player; anything else names a document to load, and the
  // FIBA one is the default so the demo keeps working unchanged.
  const requestedDocument = query.get("document") ?? editingDocumentUrl;
  let documentBytes: Uint8Array | undefined;
  if (requestedDocument !== "none") {
    const response = await fetch(requestedDocument);
    if (!response.ok) {
      throw new Error(
        `failed to fetch editing document "${requestedDocument}": ${response.status} ${response.statusText}`,
      );
    }
    documentBytes = new Uint8Array(await response.arrayBuffer());
  }
  const editor = await startVideoEditing(canvas, documentBytes);
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

  // The optional annotation document. A second picker rather than one filtered
  // for both, because the two sources are independent: either may be chosen
  // first, and the document may be cleared without touching the video (#264).
  const documentInput = document.createElement("input");
  documentInput.type = "file";
  documentInput.accept = ".arrow,.parquet";
  documentInput.hidden = true;
  document.body.append(documentInput);
  documentInput.addEventListener("change", () => {
    const file = documentInput.files?.[0];
    if (file) {
      // Mock: the choice is echoed into the dialog; decoding is its own slice,
      // so this one stays reviewable as pure UI.
      editor.setPendingDocumentSelection(file.name);
    }
  });

  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.crossOrigin = "anonymous";
  video.hidden = true;
  document.body.append(video);
  // Handed over once. Every later frame is presented by index: the browser
  // already decoded it into GPU memory, and Rust copies it GPU→GPU, so no pixels
  // cross the wasm boundary at all.
  editor.setVideoElement(video);

  let objectUrl: string | undefined;
  let pendingVideoFile: File | undefined;
  let callbackActive = false;
  let callbackId: number | undefined;
  let loadingAsset = false;
  let envBytesPromise: Promise<Uint8Array> | undefined;
  let sourceReady = false;
  let sourceGeneration = 0;

  // Only media-element state transitions. `mediaTime` is not published here: it
  // travels with its own frame through `updateVideoFrameRgba`, so the Details
  // timeline always describes the frame that reached the screen.
  function syncMediaState(): void {
    editor.setVideoMediaState(video.readyState, video.ended);
  }

  async function copyCurrentFrame(mediaTime: number, generation = sourceGeneration): Promise<void> {
    if (generation !== sourceGeneration) {
      return;
    }
    // No `VideoFrame`, no `copyTo`, no `Uint8Array`: the element itself is the
    // source and Rust copies it on the GPU. Async only so the call sites (which
    // await it) stay unchanged.
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
      void copyCurrentFrame(metadata.mediaTime, generation)
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
      void copyCurrentFrame(video.currentTime).catch((error: unknown) =>
        editor.setVideoError(String(error)),
      );
    }
  });

  function loadVideoSource(
    source: string,
    localFile?: { filename: string; byteLength: number },
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
          video.pause();
          video.currentTime = 0;
          sourceReady = true;
          editor.setVideoStatus(true, false);
          syncMediaState();
          void copyCurrentFrame(0).catch((error: unknown) => editor.setVideoError(String(error)));
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
    // Selected, not loaded: the dialog stays open so the optional document can
    // be chosen too, and its Load button commits both (#264).
    pendingVideoFile = file;
    editor.setPendingVideoSelection(file.name);
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
    } else if (command === 4) {
      documentInput.value = "";
      documentInput.click();
    } else if (command === 5) {
      // The dialog's single commit point: load the picked file, or the URL it
      // accepted, whichever is pending.
      const pendingUrl = editor.pendingVideoUrl();
      if (pendingUrl) {
        try {
          const url = new URL(pendingUrl);
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
      } else if (pendingVideoFile) {
        if (objectUrl) {
          URL.revokeObjectURL(objectUrl);
        }
        const file = pendingVideoFile;
        objectUrl = URL.createObjectURL(file);
        // Adopt the container's own timeline before the first frame arrives:
        // `<video>` never reports a frame rate, so without this the scrubber is
        // numbered on an invented grid (#264).
        void readMoov(file)
          .then((moov) => {
            if (moov) {
              editor.setVideoTimelineFromMoov(moov, file.name);
            }
          })
          .catch((error: unknown) => console.warn("moov probe failed:", error));
        loadVideoSource(objectUrl, {
          filename: file.name,
          byteLength: file.size,
        });
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
