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
    // A virtual 30 fps grid spanning an hour: web playback is driven by the
    // `<video>` element itself, so the document's fps only maps media time to a
    // frame index. Time mapping stays self-consistent
    // (`frameIndexAtMediaTime` / `mediaTimeAtFrame` share this grid); only the
    // displayed frame numbers are virtual rather than the source's own.
    editor = await startVideoEditingSynthetic(canvas, 1280, 720, 30, 1, 30 * 3600);
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

  let objectUrl: string | undefined;
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
    const frame = new VideoFrame(video, { timestamp: Math.round(mediaTime * 1_000_000) });
    try {
      const rgba = new Uint8Array(frame.allocationSize({ format: "RGBA" }));
      await frame.copyTo(rgba, { format: "RGBA" });
      if (generation !== sourceGeneration) {
        return;
      }
      editor.updateVideoFrameRgba(
        rgba,
        frame.displayWidth,
        frame.displayHeight,
        editor.frameIndexAtMediaTime(mediaTime),
        mediaTime,
      );
    } finally {
      frame.close();
    }
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
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
    }
    objectUrl = URL.createObjectURL(file);
    loadVideoSource(objectUrl, { filename: file.name, byteLength: file.size });
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
