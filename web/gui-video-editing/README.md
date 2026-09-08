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
tracks remaining old runtime/producer stamps separately from that contract.

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
