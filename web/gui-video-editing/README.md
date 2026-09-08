# GUI video editing web

The native and browser editors use one current **protocol 0.0.7 params/GLB** workflow.
There is no annotation/catalog UI or implicit legacy demo on startup.
Open Video and Load Arrow are independent. Without Arrow, the page is a plain
video player; load params to show placement controls.

The input is `[params][mesh?]`: one complete params Arrow IPC stream followed
by an optional UUID/original-GLB resource stream. Params alone show the reference
cube. A bundled document shows its bound meshes. Parquet, old
`trd.video_edit.version = 0.2.0` annotation input and mesh-first scene streams
are rejected, never silently converted by the application.

See [editable scene documents](../../docs/protocol/scene-documents.md) for the
contract and [video editing](../../docs/video-editing.md) for media/diagnostics.
The [0.0.7 specification](../../docs/protocol/0.0.7.md) is authoritative.
The [cutover status](../../docs/protocol/README.md#implementation-migration-status)
documents the shared 0.0.7 runtime/producer stamps and fixture/schema checks.

## Prepare existing annotation data offline

Original calibration/annotation data is retained. Convert it explicitly before
loading either editor; do not pass the source annotation as `document=`.

```powershell
uv run --with pyarrow python scripts\timeline_to_params.py `
  assets\videos\fiba\fiba-shot1.arrow -o output\fiba.params.arrow
uv run --with pyarrow python scripts\timeline_to_params.py `
  assets\videos\fiba\fiba-shot1.arrow -o output\fiba.dragon.arrow `
  --glb assets\meshes\glb\Meshy_AI_Dragon_0804104424_texture.glb --identity-model
```

Use annotation generated from the matching external video/calibration. FIBA has
222 sparse placement rows (0-221) and a video-only tail (222-287).
Conversion preserves the original quad order and K semantics.

## Run

From the repository root:

```powershell
cd web\gui-video-editing
bun run build:wasm
$env:BUN_PORT='8085'
bun .\index.html .\probe.html
```

The empty page is `http://localhost:8085/`. Load Video and Arrow via the UI, or
use `?document=<params-or-bundle-url>&video=<video-url>`. HTTP resources must
allow CORS; video must support byte ranges:

```powershell
bun web\gui-video-editing\serve-documents.ts output --port 8090 --log
bun web\gui-video-editing\serve-documents.ts "E:\Asset\Video" --port 8092 --log
```

For example, open
`http://localhost:8085/?document=http://localhost:8090/fiba.dragon.arrow&video=http://localhost:8092/shot_0001.mp4`.
No document is loaded by default. The editor always uses mediabunny; selecting
an old reader through a URL does not restore a legacy application path.

Native uses the same converted input:

```powershell
cargo run -p trd-gui-video-editing -- --document output\fiba.dragon.arrow `
  --video "E:\Asset\Video\shot_0001.mp4" --preview-width 1920
```

## External TS/JS API

For the complete generated WASM surface, standalone renderers and custom media
integration, see the [trd-wasm public API guide](../../docs/trd-wasm.md).

The running page exposes `window.trdVideoEditorReady: Promise<VideoEditorApi>`.
Await it before calling the editor. The API operates on the existing player and
renderer; it does not create a second media pipeline.

All four operations delegate to `VideoEditingHandle`'s real WASM exports.
The GUI's **Load Arrow** uses the same `loadArrow` path as external TS. Its file
picker only supplies bytes; validation, scene replacement and export remain in
Rust. `seekToSeconds` dispatches through the GUI's media-command loop and resolves
when the corresponding frame (or a superseding seek's frame) is displayed.

```ts
interface VideoEditorApi {
  loadArrow(input: Uint8Array | readonly Uint8Array[]): Promise<void>;
  resetState(): Promise<void>;
  exportArrow(): Promise<Uint8Array>;
  seekToSeconds(seconds: number): Promise<void>;
}

async function readArrow(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Arrow request failed: ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

const editor = await window.trdVideoEditorReady;
await editor.loadArrow(await readArrow("/first-scene.arrow"));
await editor.resetState(); // Keep the video, time, timeline and play/pause state.
await editor.loadArrow([
  await readArrow("/next-params.arrow"),
  await readArrow("/next-meshes.arrow"),
]);
await editor.seekToSeconds(5);
const exported = await editor.exportArrow(); // Bytes returned; no automatic download.
```

`loadArrow` accepts **one complete 0.0.7 document**. A byte-array list supplies
ordered parts of that same `[params][mesh?]` envelope, not separate scene
documents. The one optional mesh stream may have one or many UUID/GLB rows and
batches. Params-only input is supported. Empty input, retired versions,
malformed input and extra independent mesh streams are rejected.

Calls are serialized, and input bytes are copied when the call is made.
Successful loading atomically replaces the old scene and its renderer assets;
failed loading preserves the old scene. A resolved load/reset promise means
the UI thread applied it after outstanding render/pick work completed.
Loading does not require a preceding reset.

`resetState` drops the old Arrow source, selection, editing/picking state,
pending scene export and mesh-bearing renderer. The video-only renderer reuses
the existing GPU and video source. It neither seeks nor closes/reopens the video.
This is different from the UI's **Reset all**, which retains the loaded document.
`exportArrow` rejects after reset until another Arrow document is loaded.

Export flushes current source edits and preserves the complete 0.0.7 source
schema, unknown fields, metadata, batch boundaries, all sparse-frame matrices,
UUIDs and original GLB bytes. It is separate from the UI's download workflow.

`bun run build:web` emits the importable adapter `dist/api.js` and declarations
`dist/api.d.ts` as well as the application. An embedding host can import
`createVideoEditorApi` and provide its existing `VideoEditorBackend`; the demo
already connects that adapter to its WASM handle and `VideoPlayer`.

Targeted regressions (Rust's second command requires a real GPU):

```powershell
cd web\gui-video-editing
bun test src\api.test.ts src\media\player.test.ts
cd ..\..
cargo test -p trd-gui --lib video_editing::scene_api -- --include-ignored
```

### Real-browser API E2E

`tests/api.html` mounts the actual GUI bootstrap and observes its real WASM
calls. It does not substitute a fake player or renderer. Supply matching 0.0.7
params-only, one-resource and multiple-resource inputs plus the external FIBA
video, then run the test server from `web/gui-video-editing`:

```powershell
$env:TRD_API_PARAMS='<absolute path to params-only.arrow>'
$env:TRD_API_SINGLE='<absolute path to single-glb.arrow>'
$env:TRD_API_MULTIPLE='<absolute path to multiple-glb.arrow>'
$env:TRD_API_VIDEO='<absolute path to shot_0001.mp4>'
bun tests\serve-api.ts
```

Open `http://127.0.0.1:18370/?document=none&video=/video.mp4` in a dedicated
Chrome instance with `--remote-debugging-port=19370` and a temporary user-data
directory. Keep native desktop DPI. In another terminal:

```powershell
$env:TRD_API_CDP='http://127.0.0.1:19370'
$env:TRD_API_E2E_URL='http://127.0.0.1:18370/'
bun test tests\api.e2e.test.js --test-name-pattern integration
# Click Play in the actual GUI, then immediately run:
$env:TRD_API_PLAYING='1'
bun test tests\api.e2e.test.js --test-name-pattern playing
```

The integration case compares direct WASM calls with the external TS adapter,
checks source/export hashes, rejects old/empty input, exercises reset/reload,
all-row edited export/replay, exact seeks and overlapping WASM seeks. The playing
case proves video advancement and unchanged source identity across reset/reload.
Also load a params file through the GUI: `window.trdApiE2e.state().calls.loadArrow`
must increment and `window.trdVideoEditorReady` must export the same document.
Close the owned browser and server after the case. Optional `TRD_API_RESULTS`
records small JSON reports; large GLB bytes are compared in the browser, not
transferred through the debugging socket.

## Two-round acceptance

1. Edit the selected object's translation, rotation and scale. The edit must
   affect every corresponding existing sparse frame. Actually play,
   pause/resume and seek through early/middle/last tracked rows and the video tail.
2. Export through the UI. Audit every affected model, unchanged camera/quad/frame
   columns and original resource bytes/schema/batch boundaries.
3. Close the case and verify all owned PIDs/ports are clear. Freshly load the
   exported Arrow with the same video; repeat playback and seeks. Saved models
   must reproduce the first round at matching frames, without double application.

Use 1920x1080 source/render resolution, normal Chrome and native Windows DPI.
Screenshots are only for named visual milestones; use fresh **Copy details**
and exported-data comparisons between them. Details must name one consistent
requested/presented/displayed/rendered frame with no pending work.

## Large-file probe

The explicitly started second HTML entrypoint is served at `/probe`, not
`/probe.html`. `seek` is in seconds:

```text
/probe?reader=mediabunny&url=<video-url>&seek=<seconds>&frames=8
/probe?reader=mediabunny&url=<video-url>&scrub=t1,t2,t3&overlap=1
/?document=none&video=<video-url>
```

Test both the probe and actual editor on a >4 GiB MP4 (preferably
multi-hundred-GiB), record file size, delivered bytes and elapsed time, verify
deep/end seeks and subsequent reuse. No benchmark result is implied by a short
clip or a successful metadata probe. Full platform gates live in [AGENTS.md](../../AGENTS.md).
