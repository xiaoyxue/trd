# trd-wasm: public APIs for TypeScript and JavaScript

`crates/trd-wasm` is the repository's only browser/WASM delivery surface.
Rendering, scene validation, placement and editing live in Rust. JavaScript
supplies input bytes, browser media and the host event loop; it does not call
WebGPU to implement another renderer.

This guide covers the **current protocol 0.0.7** APIs. The generated
`trd_wasm.d.ts` alongside each build is the exact signature reference. Do not
call the generated `InitOutput` function pointers or `__wbindgen_*` helpers
directly; use the exported classes and functions.

## Choose an entry point

| Use case | Entry point |
|---|---|
| Control the running video editor from external TS | `await window.trdVideoEditorReady` |
| Embed the video-editing GUI with your own media shell | `startVideoEditing(...)` -> `VideoEditingHandle` |
| Read, edit or export scene data without a GPU | `ArrowSceneDocument` |
| Render a document to an on-screen canvas | `CanvasRenderer` |
| Render to RGBA bytes or an Arrow image stream | `OffscreenRenderer` |
| Embed the interactive OBJ/GLB model viewer | `start(...)` -> `GuiHandle` |

The video's time and the document's row index are **not interchangeable**.
`VideoEditingHandle.seekToSeconds` seeks the external video; the two standalone
renderers' `renderIndex` selects a params row, not a video frame number.

## Build, serve and initialize

Build from the repository's configured toolchain. On Windows, run these from
the repository root:

```powershell
cd web
bun run --cwd viewer build:wasm
bun run --cwd gui-viewer build:wasm
bun run --cwd gui-video-editing build:wasm
bun install --frozen-lockfile
bun run --cwd gui-video-editing build:web
```

The three packages stage the same crate independently:

| Consumer | Generated WASM/JS/declarations |
|---|---|
| `web/viewer` | `crates/trd-wasm/pkg` (its local `trd-wasm` dependency) |
| `web/gui-viewer` | `web/gui-viewer/pkg` |
| `web/gui-video-editing` | `web/gui-video-editing/pkg` |

Do not import another delivery surface's generated output. Nix users can build
the generated library with `nix build .#trd-wasm`; see [build/setup details](rendering.md).

Serve the generated `.js` and `.wasm` together over HTTP(S), with the WASM served
as `application/wasm`. GPU entry points require a WebGPU-capable browser in a
secure context: HTTPS or localhost. Remote input needs appropriate CORS headers;
remote video also needs HTTP byte ranges. WASM initialization alone does not
create a GPU device.

For an unbundled host that serves its own generated package under `/pkg/`:

```js
import init, {
  ArrowSceneDocument,
  CanvasRenderer,
  OffscreenRenderer,
} from "/pkg/trd_wasm.js";

await init({ module_or_path: new URL("/pkg/trd_wasm_bg.wasm", location.href) });

async function readBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}
```

Bundled TypeScript uses the package/local generated-module import and the
bundler's emitted WASM URL; the repository's viewer bootstraps demonstrate both.
Initialize the module before calling its exports. `init` also accepts supplied
WASM bytes or a `WebAssembly.Module` through `module_or_path`.
`initSync({ module: wasmBytes })` is available for already-loaded bytes/modules;
it is not a replacement for asynchronous GPU creation.

## Input contract

One document is `[params]` followed by **one optional** `[mesh]` Arrow IPC stream.
The mesh stream may contain one or many UUID/GLB rows and record batches.
It is not a sequence of independently concatenated scene documents.

| Input | Meaning |
|---|---|
| `[params]` | Reference geometry without external mesh resources |
| `[params][mesh]`, one resource row | One original GLB resource |
| `[params][mesh]`, multiple resource rows | Multiple resources, resolved through their UUID bindings |

Declared protocol versions must match **0.0.7**. Retired mesh-first envelopes,
texture/inline-frame resource streams and GLB path/URL resource rows are not
accepted. Unversioned FHC ingestion is a separate explicit source adapter, not
an upgrade path for old versioned data. See the [protocol specification](protocol/0.0.7.md).

Video stays outside Arrow. Sparse rows are retained: absent placement rows mean
video-only playback, not invented model rows. GLB bytes contain geometry,
textures and materials; resource count is not necessarily drawn-instance count.

## External TS: control the running video editor

The bundled video-editor page installs
`window.trdVideoEditorReady: Promise<VideoEditorApi>`. Await it; mounting failures
reject the promise. It controls the existing GUI/player, not a second editor.

```ts
interface VideoEditorApi {
  loadArrow(input: Uint8Array | readonly Uint8Array[]): Promise<void>;
  resetState(): Promise<void>;
  exportArrow(): Promise<Uint8Array>;
  seekToSeconds(seconds: number): Promise<void>;
}

const editor = await window.trdVideoEditorReady;

await editor.loadArrow(paramsBytes);                    // params only
await editor.loadArrow([nextParamsBytes, meshBytes]);    // replace with a bundle
await editor.seekToSeconds(4.5);
const exported: Uint8Array = await editor.exportArrow(); // no automatic download

await editor.resetState();                              // return to video only
await editor.loadArrow(exported);
await editor.seekToSeconds(4.5);
```

Here `paramsBytes`, `nextParamsBytes` and `meshBytes` are caller-provided bytes.
Array input is copied and concatenated in order as **one document**. It can be
split at stream boundaries or arbitrary byte boundaries. It does not merge
independent documents, append scenes or allow repeated mesh streams.

### Lifecycle guarantees

| Operation | Behavior |
|---|---|
| `loadArrow` | Copies input at call time. Successful loading replaces the old scene/edit state and corresponding renderer assets. Invalid input preserves the prior scene. No preceding reset is required. |
| `resetState` | Discards Arrow, selection/edit/pick state, pending scene export and the mesh-bearing renderer. Keeps the existing video source, timeline, position and play/pause state. Reuses the GPU for video-only rendering. |
| `exportArrow` | Flushes current source edits and returns a retained 0.0.7 scene snapshot. Preserves sparse matrices, other columns, metadata, batch boundaries, UUIDs and original GLB bytes. Rejects if no current document is loaded. |
| `seekToSeconds` | Requires a loaded video and finite, nonnegative seconds. Uses the GUI's timeline conversion and clamps to its presentable range. Resolves after the corresponding frame is displayed, or a superseding seek is displayed. |

The TS adapter serializes operations, including recovery after a rejected call.
Load/reset completion acknowledges application on the UI thread after
outstanding render/pick work; it does not mean the browser has completed a
screen refresh. Reset does **not** rewind, pause, close or reopen the video.

The GUI's **Load Arrow** uses this same WASM-backed loading path.
Its **Unload Arrow** uses `resetState`. The GUI's **Reset all** is different:
it resets editing controls/selection while retaining the document.

### Importable adapter

`web/gui-video-editing` builds `dist/api.js` and `dist/api.d.ts`. Its local
workspace package, `trd-gui-video-editing-web`, exports `createVideoEditorApi`
and the `VideoEditorApi`, `VideoEditorBackend` and `ArrowInput` types. This is a
repository-local package, not a claim that it is published on npm.

A host that already owns a live WASM editor can wrap it:

```ts
import { createVideoEditorApi } from "trd-gui-video-editing-web";
import type { VideoEditingHandle } from "./pkg/trd_wasm.js";

function wrapEditor(handle: VideoEditingHandle) {
  return createVideoEditorApi({
    loadArrow: (bytes) => handle.loadArrow(bytes),
    resetState: () => handle.resetState(),
    exportArrow: () => handle.exportArrow(),
    seekToSeconds: (seconds) => handle.seekToSeconds(seconds),
  });
}
```

The adapter does not mount a GUI or attach a video. In a standalone deployment,
import its emitted `api.js` and keep the corresponding declarations available
to your TS build. Only the existing demo bootstrap installs the ready promise.

## VideoEditingHandle: embedding a media shell

All four lifecycle methods above are actual exports on `VideoEditingHandle`.
Unlike the TS adapter, its `loadArrow` accepts a **single `Uint8Array`**.
Await operations explicitly when using the raw handle.

```js
import { startVideoEditing } from "/pkg/trd_wasm.js";

// Module initialization and readBytes are as above.
const canvas = document.querySelector("#editor");
if (!(canvas instanceof HTMLCanvasElement)) throw new Error("missing editor canvas");

const handle = await startVideoEditing(
  canvas,
  undefined, // no initial Arrow; may instead supply a complete document
  [],        // no external glTF reference resources in the current protocol
  await readBytes("/assets/envmap/uffizi-large.hdr"),
);
```

**This mounts the Rust GUI, not the media reader.** A custom host must service
the media/GUI bridge below. Without that service loop, a valid seek cannot
complete. Reuse the [existing bootstrap](../web/gui-video-editing/src/main.ts),
[`VideoPlayer`](../web/gui-video-editing/src/media/player.ts) and mediabunny
reader rather than implementing another decoder or assuming an HTML
`<video>` element exposes rational fps.

| Bridge method | Host responsibility |
|---|---|
| `setVideoTimelineFromMoov(moov, sourceName)` | Supply the container's raw `moov` bytes; Rust adopts rational fps/frame count. |
| `validateVideoFile(name, byteLength)` / `validateVideoMetadata(width, height, durationSeconds)` | Perform the handle's available source validation; these methods do not open media. |
| `setVideoSourceInfo(kind, name, byteLength)` | Report source identity: kind `1` = local file, `2` = HTTP URL. |
| `setVideoStatus(loaded, playing)` / `setVideoMediaState(readyState, ended)` | Report actual media state. Setting status is not a play/pause command; `loaded = false` invalidates old source work. |
| `frameIndexAtMediaTime(seconds)` / `mediaTimeAtFrame(index)` | Use the adopted timeline mapping rather than hard-coded 30 fps. |
| `presentVideoFrame(frame, index, seconds, durationSeconds)` | Transfer ownership of a decoded browser `VideoFrame` to Rust. Do not close it after a successful transfer. |
| `updateVideoFrameRgba(rgba, width, height, index, seconds, durationSeconds)` | CPU-upload alternative; supply tightly packed `width * height * 4` RGBA bytes. |
| `takeSeekFrame()` | Poll the seek command; `-1` means none. Seek the existing reader to `mediaTimeAtFrame(index)` and present the decoded answer. |
| `takeCommand()` | Poll and service the GUI action code below. |
| `setError(scope, message)` | Surface host failures. Codes: media `1`, document `3`, render `4`, pick `5`, export `6`; `2` is reserved by the existing enum, not a current catalog feature. |

`takeCommand()` codes are `0` none, `1` choose video, `2` play, `3` pause,
`4` choose Arrow, `5` load the pending selection, `6` save queued Arrow export,
`7` load video, `8` load/unload Arrow.

The picker bridge uses `setPendingVideoSelection(name)` and
`setPendingDocumentSelection(name)` for local choices; retain the corresponding
`File` in JS. `pendingVideoUrl()`, `pendingDocumentUrl()` and
`hasPendingDocument()` expose the GUI's pending selection. Picking is not loading.

For command `6`, read `pendingArrowExportFilename()` and `takeExportArrow()`,
save/download the bytes, then call `finishArrowExport(success, message)`.
`cancelArrowExport()` cancels the pending UI export. This workflow is separate
from `await exportArrow()`, which simply returns a snapshot to an external caller.

### Existing aliases and lower-level entry points

`loadDocument(bytes)` aliases current document loading.
`loadDocumentWithGltf(bytes, [], envBytes)` accepts an explicit environment probe;
the current params/GLB protocol requires an empty reference array.
`videoEditingGltfReferences(bytes)` remains exported, but a valid current
document has no such reference resources and returns an empty array.
Neither method restores GLB URL/path or old-protocol support.

`clearDocument()` is an older, queued clear operation without the full
acknowledged resource-reset contract. Use **`await resetState()`** for external
reset-then-load code.

## ArrowSceneDocument: GPU-free source editing

Initialize WASM, then choose a constructor:

| Constructor | Input |
|---|---|
| `fromArrow(bytes)` | Complete params-only or bundled document |
| `fromArrowWithGlb(params, glb)` | One GLB; infer a unique source UUID or generate one |
| `fromArrowWithMesh(params, uuid, glb)` | One GLB with an explicit UUID |
| `fromArrowWithGlbs(params, bindings)` | A `Map<string, Uint8Array>` keyed by resource UUID |

Separate-GLB constructors bind resources to params-only input; they do not
replace resources in an already bundled document. For multiple bindings,
provide actual UUID strings, not renderer-local numeric slots.

`frameCount()`, `objectCount(row)` and `meshIds()` inspect the document.
`renderConfig()` returns the document-wide JSON configuration;
`setRenderConfig(json)` validates and replaces only its params schema metadata.
The default is `{"shadow":{"enable":true,"shadow_type":"blob"}}`.
Unknown fields, invalid types, and reserved `shadow_map` are explicit errors.
This setting is shared by every row and retained by `exportArrow()`.

`getModel(row, object)` returns a `Float32Array` copy with **16 column-major**
elements. Changing that copy alone does not edit the document:

```js
const sceneDocument = ArrowSceneDocument.fromArrow(await readBytes("/scene.arrow"));
try {
  const model = sceneDocument.getModel(0, 0);
  model[12] += 0.2;
  sceneDocument.setModel(0, 0, model);
  const exported = sceneDocument.exportArrow();
  // Store/export these bytes in the host application.
} finally {
  sceneDocument.free();
}
```

`setModel` edits exactly **one row/object** and rejects invalid matrices/indices.
It is not the video GUI's track-wide editing operation. Source export restores
the original wire layout: tracked nested matrices are row-major; CG/CV matrices
are column-major. The editor's transform controls apply their adjustment to all
matching sparse rows. See [scene editing semantics](protocol/scene-documents.md).

## CanvasRenderer and OffscreenRenderer

The bundled stream viewer starts its playback clock on the first animation-frame
callback, not during asynchronous renderer setup. Both targets begin at row zero
and then select rows from elapsed RAF time; initialization cannot produce a
negative row index.

Both accept the same source document and appearance controls. They do not own a
video reader or a playback clock. The host selects rows and supplies backgrounds.

```js
const canvas = document.querySelector("#scene");
if (!(canvas instanceof HTMLCanvasElement)) throw new Error("missing scene canvas");
canvas.width = 512;
canvas.height = 768;

const renderer = await CanvasRenderer.create(canvas);
try {
  renderer.setPbr(true);
  renderer.setEnvMapHdr(await readBytes("/assets/envmap/uffizi-large.hdr"));
  const sceneDocument = ArrowSceneDocument.fromArrow(await readBytes("/scene.arrow"));
  try {
    const rows = renderer.loadSceneDocument(sceneDocument);
    if (rows > 0) renderer.renderIndex(0); // synchronous submit/present
    renderer.finish();
  } finally {
    sceneDocument.free();
  }
} finally {
  renderer.free();
}
```

The examples are separate recipes, not one concatenated script.

| Method | Canvas | Offscreen |
|---|---|---|
| `create(...)` | `Promise<CanvasRenderer>`; nonzero canvas width/height | `Promise<OffscreenRenderer>`; explicit nonzero width/height |
| `loadSceneDocument(document)` | Returns params-row count | Same |
| `loadIpc(bytes)` | Loads one complete document, returns row count | Same |
| `frameCount()` / `meshResourceCount()` | Loaded row/resource counts | Same |
| `frameRef(row)` | External background reference or `undefined`; invalid row throws | Same |
| `renderIndex(row)` | `void`: synchronous presentation | `Promise<Uint8Array>`: tightly packed RGBA |
| `renderIpc(row)` | Not provided | `Promise<Uint8Array>`: one rendered image appended to output IPC |
| `finish()` | Closes rendering; returns `void` | Closes rendering; returns remaining output IPC bytes, including EOS |

`loadIpc` is **not** an incremental parser. There is no `pushIpc`, `resolveGltf`,
`gltfPath` or `gltfUrl` input path. Load a new complete document to replace one.
These standalone renderers do not provide the video editor's `resetState`.

`loadSceneDocument` retains a reference to the same source, so later `setRenderConfig` or `setModel`
calls are visible when rendering another row without re-uploading meshes.
`loadIpc` is the convenience path when the caller does not need an editable handle.

### RGBA readback and image IPC

```js
const renderer = await OffscreenRenderer.create(512, 768);
try {
  renderer.loadIpc(await readBytes("/scene.arrow"));
  const rgba = await renderer.renderIndex(0);
  // rgba.length === 512 * 768 * 4; paint/store it in the host.
  const first = await renderer.renderIpc(0);
  const last = renderer.finish();
  const imageStream = new Uint8Array(first.length + last.length);
  imageStream.set(first);
  imageStream.set(last, first.length);
} finally {
  renderer.free();
}
```

Concatenate **every** `renderIpc` result and the final `finish()` result, in
order, for a complete Arrow **image** stream. Its `r/g/b/a` columns are planar
`fixed_shape_tensor<u8>[H,W]`. Only explicit `renderIpc` calls append images;
RGBA playback via `renderIndex` does not accumulate output.
Each call writes one image in call order, not the source's params batch layout.
This stream is not a scene document and cannot replace `exportArrow()`.

Await offscreen rendering before issuing the next mutable renderer operation.
After `finish`, further loading, rendering, frame uploads and another `finish`
are rejected. `finish` is terminal, not a rewind operation.

### Appearance and external backgrounds

Both renderers expose:

| Controls | Effect |
|---|---|
| `setPbr`, `setTextured`, `setWireframe` | Choose shaded, textured or wireframe mode; setting one to `false` selects filled mode. Explicit per-draw modes still take precedence. |
| `setPbrMaterial(metallic, roughness, specular, clearcoat, envIntensity, exposure, ambient, tonemap)` | Set a material/lighting override for meshes; use `"aces"` or `"reinhard"`. This can replace imported GLB material values, so do not call it if those should be retained. |
| `setEnvMapHdr(bytes)` / `setEnvBackground(enabled, blur)` | Bind an HDR environment and optionally show its sky (`blur` from 0 to 1). Use `assets/envmap/uffizi-large.hdr` by default. |
| `setShowAabb`, `setShowAxes`, `setShowLocalAxes` | Per-instance bounds, world axes and object-local axes |
| `setCompositeFrame(enabled)` / `updateFrameTextureRgba(bytes, width, height)` | Composite a supplied RGBA background under the scene |

For each externally referenced background: call `frameRef(row)`, fetch/decode
that image in the host, upload its RGBA bytes, then render that row. Upload
before **each render** needing that background; it is not automatically reused
as a valid background for the next row. A row with no background must not
display a previous still. Enable/disable compositing accordingly.

The document's validated tone-map override, when present, takes precedence for
mesh output and the sky. For a working playback loop, see
[`web/viewer/src/viewer.ts`](../web/viewer/src/viewer.ts).

## Interactive model viewer

`start(canvas, meshBytes, textureBytes, envBytes?, onPickModel?)` mounts the
Rust GUI and returns `Promise<GuiHandle>`. Supply arrays of OBJ/GLB bytes;
texture entries align positionally with meshes. These are the internal model
viewer's inputs, not permission to encode OBJ geometry in protocol Arrow.

The host's `onPickModel` callback opens a browser file picker after the GUI's
Load model action. Hand the selected GLB back with
`handle.loadModel(name, glbBytes, envBytes?)`. This queues work and requests a GUI
repaint, so an idle viewer processes the upload without another pointer or keyboard
event. Its `void` return is **not** a load-completion promise.

Use the [GUI viewer bootstrap](../web/gui-viewer/src/main.ts) for complete
picker/texture/environment handling. It is a different application from the
video editor; `GuiHandle` does not expose the editor's Arrow/reset/export APIs.

## Errors, ownership and regression coverage

Synchronous fallible WASM calls throw; asynchronous ones reject their promises.
Some older bridge paths throw string values, so catch `unknown` and report
`String(error)` rather than assuming every failure has `.message`. Do not
silently retry old protocol input or treat a failed load as a reset.

Generated classes expose `free()` and `[Symbol.dispose]()`. Release handles you
own after their outstanding work completes; never call them after disposal.
`free()` is not an editor reset or a documented stop mechanism for the mounted
GUI/media loop. Stop the host's scheduled callbacks and media resources through
the host's own lifecycle. Do not create repeated GUI runners on the same canvas
to simulate reset.

The [video editor API E2E recipe](../web/gui-video-editing/README.md#real-browser-api-e2e)
exercises actual WASM, external TS, GUI loading, reset while playing, source
retention, exported edits and seeks. The
[standalone renderer recipe](rendering.md#web-wasm) covers Canvas/Offscreen
pixels, lifecycle and image IPC. The shared startup/loop clock also has a
GPU-free regression: `bun test web\viewer\tests\frame-clock.test.js`.
Full platform gates and handoff rules remain
in [AGENTS.md](../AGENTS.md#testing).
