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
import { byteSourceFor, MediabunnyReader, type MediaInput } from "./media/mediabunny-reader.ts";
import { VideoPlayer } from "./media/player.ts";

/// Which path an error came from. Mirrors Rust's `ErrorScope` codes, which the
/// editor uses to keep one path's success from clearing another's failure.
type ErrorScope = "media" | "catalog" | "document" | "export";
const errorScopes: Record<ErrorScope, number> = {
  media: 1,
  catalog: 2,
  document: 3,
  export: 6,
};

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

  /// Surfaces a failure. The editor's UI is a canvas, so an error drawn there
  /// can be read but not selected, copied or scrolled back to — logging it as
  /// well is what makes a failure reportable and reproducible.
  ///
  /// `scope` names the path that failed, so a recovered decode cannot clear a
  /// catalog or document failure (#329).
  function reportError(scope: ErrorScope, message: string): void {
    console.error(`video editing (${scope}): ${message}`);
    editor.setError(errorScopes[scope], message);
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
      // Selected, not loaded: Load commits the video and the document together.
      pendingDocumentFile = file;
      editor.setPendingDocumentSelection(file.name);
    }
  });

  let pendingVideoFile: File | undefined;
  let pendingDocumentFile: File | undefined;
  let player: VideoPlayer | undefined;

  /// Applies the dialog's document selection: a picked file, a fetched URL, or
  /// nothing — which means "play unannotated", since Load commits the whole
  /// selection (#264).
  async function loadSelectedDocument(): Promise<void> {
    const url = editor.pendingDocumentUrl();
    if (url) {
      let response: Response;
      try {
        response = await fetch(url);
      } catch (error) {
        // A `fetch` rejection is almost always the browser refusing the request
        // rather than the server answering: no CORS header, or nothing
        // listening. Say so, because the status-code path below cannot.
        throw new Error(
          `${String(error)} — the server must send Access-Control-Allow-Origin for a cross-origin document`,
        );
      }
      if (!response.ok) {
        throw new Error(`failed to fetch document: ${response.status} ${response.statusText}`);
      }
      await editor.loadDocument(new Uint8Array(await response.arrayBuffer()));
      return;
    }
    if (editor.hasPendingDocument() && pendingDocumentFile) {
      await editor.loadDocument(new Uint8Array(await pendingDocumentFile.arrayBuffer()));
      return;
    }
    editor.clearDocument();
  }
  let loadingAsset = false;
  let envBytesPromise: Promise<Uint8Array> | undefined;
  let sourceReady = false;
  let sourceGeneration = 0;

  function downloadArrow(filename: string, bytes: Uint8Array): void {
    const url = URL.createObjectURL(
      new Blob([Uint8Array.from(bytes).buffer], {
        type: "application/vnd.apache.arrow.stream",
      }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.hidden = true;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30_000);
  }

  /// Receives decoded frames. Ownership arrives with the frame and passes
  /// straight to Rust, which closes it after the GPU copy — no pixels cross the
  /// wasm boundary in either direction.
  function frameSink(generation: number) {
    return {
      present(frame: VideoFrame, mediaSeconds: number): void {
        if (generation !== sourceGeneration) {
          frame.close();
          return;
        }
        try {
          editor.presentVideoFrame(
            frame,
            editor.frameIndexAtMediaTime(mediaSeconds),
            mediaSeconds,
            // The container's own duration for this frame. It is what decides
            // whether a seek landed, because a nominal interval derived from the
            // mean rate is not an upper bound on a variable-rate recording
            // (#317). `null` when the decoder dropped it; Rust falls back.
            (frame.duration ?? 0) / 1_000_000,
          );
        } catch (error) {
          reportError("media", String(error));
        }
      },
      ended(): void {
        if (generation === sourceGeneration) {
          editor.setVideoStatus(sourceReady, false);
        }
      },
      failed(message: string): void {
        if (generation === sourceGeneration) {
          reportError("media", message);
        }
      },
    };
  }

  /// Opens a video for decoding. The source is a local file or a URL behind the
  /// same interface, so this path no longer forks on which one it has (#282).
  ///
  /// Which reader demuxes it *is* a fork, deliberately: `?reader=mediabunny`
  /// delegates that layer to a library, and the default keeps the hand-written
  /// mp4box path. Both run under the same player, so a difference between them
  /// is a difference in demuxing and nothing else.
  async function loadVideoSource(
    media: MediaInput,
    localFile?: { filename: string; byteLength: number },
  ): Promise<void> {
    if (localFile) {
      editor.validateVideoFile(localFile.filename, localFile.byteLength);
    }
    const generation = ++sourceGeneration;
    player?.close();
    player = undefined;
    sourceReady = false;
    editor.setVideoStatus(false, false);
    try {
      const label = media.kind === "file" ? media.file.name : media.url;
      let opened: VideoPlayer;
      let byteLength: number;
      if (query.get("reader") === "mediabunny") {
        const reader = await MediabunnyReader.open(media);
        byteLength = localFile?.byteLength ?? 0;
        opened = VideoPlayer.attach(reader, frameSink(generation));
      } else {
        const source = await byteSourceFor(media);
        byteLength = source.size;
        opened = await VideoPlayer.open(source, frameSink(generation));
      }
      if (generation !== sourceGeneration) {
        opened.close();
        return;
      }
      editor.setVideoSourceInfo(
        localFile ? 1 : 2,
        localFile?.filename ?? label,
        localFile?.byteLength ?? byteLength,
      );
      // Adopt the container's own timeline before anything is drawn. The frame
      // rate is a rational the sample table states; deriving it from a sample
      // count over a duration would reintroduce the invented grid #264 removed.
      editor.setVideoTimelineFromMoov(opened.moovBytes, localFile?.filename ?? label);
      editor.validateVideoMetadata(
        opened.facts.width,
        opened.facts.height,
        opened.facts.durationSeconds,
      );
      player = opened;
      sourceReady = true;
      editor.setVideoStatus(true, false);
      // `readyState`/`ended` are `<video>` vocabulary; with a decoder the only
      // meaningful report is "there is data and it has not run out".
      editor.setVideoMediaState(4, false);
      await opened.seekToSeconds(0);
    } catch (error) {
      if (generation === sourceGeneration) {
        sourceReady = false;
        editor.setVideoStatus(false, false);
        reportError("media", String(error));
      }
    }
  }

  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (!file) {
      return;
    }
    try {
      editor.validateVideoFile(file.name, file.size);
    } catch (error) {
      reportError("media", String(error));
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
      player?.play();
      editor.setVideoStatus(sourceReady, sourceReady && player !== undefined);
    } else if (command === 3) {
      player?.pause();
      editor.setVideoStatus(sourceReady, false);
    } else if (command === 4) {
      documentInput.value = "";
      documentInput.click();
    } else if (command === 5) {
      void (async () => {
        // The video timeline must land before a protocol scene is validated
        // against it; native follows the same video-then-Arrow ordering.
        const pendingUrl = editor.pendingVideoUrl();
        let videoRequested = false;
        if (pendingUrl) {
          videoRequested = true;
          try {
            const url = new URL(pendingUrl);
            if (url.protocol !== "http:" && url.protocol !== "https:") {
              throw new Error("video URL must use http:// or https://");
            }
            await loadVideoSource({ kind: "url", url: url.href });
          } catch (error) {
            reportError("media", String(error));
            return;
          }
        } else if (pendingVideoFile) {
          videoRequested = true;
          const file = pendingVideoFile;
          await loadVideoSource(
            { kind: "file", file },
            { filename: file.name, byteLength: file.size },
          );
        }
        if (!videoRequested || sourceReady) {
          await loadSelectedDocument();
        }
      })().catch((error: unknown) => reportError("document", String(error)));
    } else if (command === 6) {
      const filename = editor.pendingArrowExportFilename();
      try {
        if (!filename) {
          throw new Error("the editor requested an export without a filename");
        }
        const bytes = editor.takeExportArrow();
        downloadArrow(filename, bytes);
        editor.finishArrowExport(true, `Downloaded ${bytes.byteLength} bytes as ${filename}`);
      } catch (error) {
        const message = String(error);
        console.error(`video editing (export): ${message}`);
        editor.finishArrowExport(false, message);
      }
    }

    const seekFrame = editor.takeSeekFrame();
    if (seekFrame >= 0 && player) {
      void player
        .seekToSeconds(editor.mediaTimeAtFrame(seekFrame))
        .catch((error: unknown) => reportError("media", String(error)));
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
        .catch((error: unknown) => reportError("catalog", String(error)))
        .finally(() => {
          loadingAsset = false;
        });
    }
    requestAnimationFrame(serviceRustCommands);
  }
  requestAnimationFrame(serviceRustCommands);

  // `?video=<url>` opens a video without going through the dialog, the same way
  // `?document=` already works. The dialog is an egui canvas, so this is also
  // the only way a scripted browser run can reach the playback path.
  const requestedVideo = query.get("video");
  if (requestedVideo) {
    void loadVideoSource({ kind: "url", url: requestedVideo }).then(() => {
      // `&play=1` starts playback too, so a scripted run can exercise the
      // decode/pace loop without driving the egui transport bar.
      if (query.get("play") === "1") {
        player?.play();
        editor.setVideoStatus(true, true);
      }
    });
  }
}

main().catch((error: unknown) => {
  console.error("video editing failed:", error);
  document.body.innerHTML = `<pre style="color:#f88;padding:1rem">${String(error)}</pre>`;
});
