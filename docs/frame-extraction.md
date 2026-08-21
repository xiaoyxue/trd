# Frame extraction, external references, and inline tables

[`scripts/extract_frames.py`](../scripts/extract_frames.py) deterministically
turns video into zero-padded stills plus metadata. Video/container decoding
stays at the boundary (`ffmpeg`); the rendering protocol consumes either
external image references or a self-contained inline frames table.

## Contents

- [Extract frames](#extract-frames)
- [External-reference mode (default)](#external-reference-mode-default)
- [Inline mode](#inline-mode)
  - [Standard full-clip tensor e2e](#standard-full-clip-tensor-e2e)
- [Pack existing images](#pack-existing-images)
- [Determinism](#determinism)

## Extract frames

```sh
uv run --with pyarrow scripts/extract_frames.py \
  assets/videos/cornellbox/CameraMovement.mp4 -o output/cornellbox
```

Default output:

```text
output/cornellbox/
├── frames/frame_000000.png
├── frames/frame_000001.png
├── ...
├── frames.arrow
└── frames.json
```

The extractor uses `-fps_mode passthrough -start_number 0`, so extraction order
is stable and row/frame ID `N` identifies decoded video frame `N`. `--fps`
changes recorded playback metadata only; it does not resample.

Run the script without arguments for all flags. Important options:

- `--format png|jpg`
- `--width` / `--height` to scale during extraction
- `--url-base` for browser references
- `--embed bytes|pixels` to make `frames.arrow` a protocol resource table
- `--no-arrow` for JSON/stills only (`--embed` is incompatible)

## External-reference mode (default)

**Rule: external-reference mode keeps image payloads outside the Arrow input.**
`frames.arrow` is a small mapping manifest:

| Column | Type | Meaning |
|---|---|---|
| `row` | `UInt32` | 0-based extracted frame. |
| `frame_path` | `Utf8` | Native path relative to the manifest directory. |
| `frame_url` | `Utf8` | Browser URL relative to the served base. |

Metadata includes `trd.protocol.version = 0.0.6`,
`trd.stream.frame_rate`, dimensions, and count. `frames.json` mirrors the data
and includes `frame_id = row`.

A scene producer copies `frame_path` / `frame_url` into params rows. Native
front-ends resolve `frame_path` under `--frames-base`; the browser resolves
`frame_url`. This mode scales to large/unbounded clips because image payloads
are not retained in the Arrow input.

## Inline mode

**Rule: inline mode makes `frames.arrow` a protocol resource table.** Use it for
self-contained streams, choosing compressed bytes for clips and raw pixels for
small fixtures or zero-codec producers.

```sh
# Recommended: preserve compressed PNG/JPEG bytes.
uv run --with pyarrow scripts/extract_frames.py input.mp4 \
  -o output/clip --embed bytes

# Raw fixed-shape RGBA tensor (also needs Pillow + NumPy).
uv run --with pyarrow --with pillow --with numpy \
  scripts/extract_frames.py input.mp4 -o output/clip --embed pixels
```

Here `frames.arrow` is a protocol `trd.table.kind = frames` stream:

| Mode | Column | Representation |
|---|---|---|
| `bytes` | `frame_bytes: Binary` | Original encoded PNG/JPEG per row. |
| `pixels` | `frame_pixels: fixed_shape_tensor<UInt8>[H,W,4]` | Raw RGBA per row; all dimensions equal. |

Params rows select resources with nullable `frame_id: UInt32`. IDs are 0-based
frames-table row ordinals and may be reused.

Compose a self-contained renderer input:

```sh
# frames.json provides frame_id values for authoring params JSONL.
{ scripts/obj_to_arrow.py bunny.obj
  cat output/clip/frames.arrow
  scripts/jsonl_to_arrow.py clip-params.jsonl
} | trd-cli --width 320 --height 180
```

No `--frames-base` is needed. `trd-core` decodes Binary PNG/JPEG when selected
and uploads tensor RGBA directly. Binary is recommended for clips; raw pixels
retain the full uncompressed resource table in memory.

### Standard full-clip tensor e2e

**Rule: this e2e intentionally exercises the raw tensor wire path.** The
repository's inline protocol e2e uses every annotated frame from the Cornell-box
source (250 frames at 25 fps) at its native 1920×1080 resolution.
The params are produced by the same perception and placement stages
as the external `frame_path` demo, but emit `frame_id`; the scene contains only
the textured bunny.

```sh
mkdir -p output/cornellbox-inline
uv run --with pyarrow --with numpy scripts/perception_to_arrow.py \
  --assets assets/videos/cornellbox --step 1 \
  -o output/cornellbox-inline/perception.arrow
uv run --with pyarrow --with numpy examples/placement_quad_by_local_coord.py \
  --from-perception output/cornellbox-inline/perception.arrow --inline-frames \
  --width 1920 --height 1080 \
  -o examples/frames.cornellbox.inline.jsonl
uv run --with pyarrow --with pillow --with numpy scripts/extract_frames.py \
  assets/videos/cornellbox/CameraMovement.mp4 --format jpg \
  --width 1920 --height 1080 --embed pixels -o output/cornellbox-inline
examples/render.sh --cli \
  --frames-table output/cornellbox-inline/frames.arrow \
  --mesh assets/meshes/bunny_with_texture/bunny.obj \
  --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg \
  examples/frames.cornellbox.inline.jsonl \
  output/cornellbox-inline-tensor-bunny.gif 1920 1080 25
```

The raw tensor table is approximately 1.93 GiB. This intentionally exercises the
tensor wire path end to end; use Binary or external references for scalable
production clips.

## Pack existing images

Use [`scripts/frames_to_arrow.py`](../scripts/frames_to_arrow.py) when stills
already exist:

```sh
uv run --with pyarrow scripts/frames_to_arrow.py --storage bytes \
  frames/frame_000000.jpg frames/frame_000001.jpg -o frames.arrow
```

For `--storage pixels`, add `--with pillow --with numpy`; every image must have
the same dimensions.

## Determinism

PNG extraction is byte-stable for the same source/toolchain. Manifests and
inline tables preserve sorted input order, so `frame_id` mapping is reproducible.
The committed golden fixtures use both inline forms and are regenerated by
`scripts/golden_fixtures.py`.
