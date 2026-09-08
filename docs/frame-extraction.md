# Frame extraction and external image references

[`scripts/extract_frames.py`](../scripts/extract_frames.py) turns video into
zero-padded stills and a mapping manifest. **Protocol
[0.0.7](protocol/0.0.7.md) uses external still references, not an inline frames
table.** Video-editor playback stays external and normally needs no extraction.

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

The output directory contains `frames/frame_000000.png`, subsequent numbered
images, `frames.arrow` and `frames.json`. The Arrow file here is an **offline
mapping manifest**, not another stream to append to renderer input.

The extractor uses `-fps_mode passthrough -start_number 0`. `--fps` changes
recorded playback metadata; it does not resample video or establish real
packet-to-presentation identity.

Useful options are `--format png|jpg`, `--width`, `--height`, `--url-base` and
`--no-arrow` for JSON/stills only. `--embed` exists in legacy tooling but does
not produce a valid 0.0.7 scene resource.

## External-reference mode (default)

| Manifest field | Type | Meaning |
|---|---|---|
| `row` | `UInt32` | extracted-frame ordinal |
| `frame_path` | `Utf8` | relative path to the external image |
| `frame_url` | `Utf8` | browser-resolvable image reference |

A scene producer copies the appropriate `frame_path`/`frame_url` into its CG/CV
params rows. Native resolves a path under `--frames-base`; the browser preloads
and uploads the referenced image. A row without a reference must not retain
the previous row's background.

Example using existing external-reference demo params:

```sh
examples/render.sh --cli \
  --mesh assets/meshes/bunny_with_texture/bunny.obj \
  --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg \
  --frames-base output/cornellbox \
  examples/frames.cornellbox.stage2.jsonl output/cornellbox.gif 960 540 25
```

The wrapper converts OBJ/albedo offline into GLB and emits `[params][mesh]`.
Do not append `frames.arrow` or a texture stream. The manifest's old tool
metadata is not a license to submit it as current input; remaining producer
stamps are recorded in the
[atomic migration status](protocol/README.md#implementation-migration-status).

## Inline mode

**Historical, retired scene-input path.** Protocol 0.0.6 allowed an indexed
`frames` stream with compressed `frame_bytes` or raw `frame_pixels`, selected
by params `frame_id`. Those columns/streams are not part of 0.0.7.

For old data, explicitly extract external stills and change source references
as a migration step; do not add a runtime compatibility fallback.
`--frames-table` / `-FramesTable` now report that the old input is retired.

### Standard full-clip tensor e2e

The former 250-frame, 1920x1080 Cornell-box raw tensor example was a regression
for the **old** input envelope. It is not current acceptance coverage.
Use the external-reference recipe above and the migrated params/GLB goldens,
without changing their camera/draw values or expected images.

## Pack existing images

For current input, reference existing PNG/JPEG files directly from params.
`scripts/frames_to_arrow.py` remains legacy/offline tooling; its inline output
must not be presented as a 0.0.7 renderer resource.

## Determinism

PNG extraction is byte-stable for the same source/toolchain. Preserve mapping
order and source camera/frame identity when generating references. The committed
`stage{1,2}.arrow` goldens now use params/GLB and external stills under their
`frames/` directory. Their old inline resources were externalized without
changing pre-existing golden PNGs.
