# Changelog

All notable changes to the trd stream protocol are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/) and the protocol version
follows `MAJOR.MINOR.PATCH`.

## Unreleased

### Added
- Optional schema metadata **`trd.stream.frame_rate`** (float fps, default 30):
  declares the stream's playback rate. Version-independent (applies to 0.0.1 and
  0.0.2). `trd-cli` copies it into the rendered image stream so `encode.py` uses
  it as the default GIF/WebP frame rate; `trd-app` plays at this rate (or `--fps`)
  and, by default, presents decoupled from the monitor refresh (`--vsync` to lock).

## 0.0.3 — Current

### Added
- Optional **leading mesh Arrow stream**, concatenated before the params stream
  (`[mesh][params]`) — the epic's "multiple tables + glue logic". One row = one
  mesh, geometry nested in list columns: **`position`** and optional **`color`**
  as `List<FixedSizeList<Float32>[3]>`, optional **`index`** as `List<UInt32>`.
  Decoded by `Mesh::from_arrow` and drawn with `draw_indexed`.
- **Multi-stream framing** with schema sniffing: a first stream carrying a
  `position` column is a mesh table (decode + upload, then render the following
  params stream); otherwise the input is a legacy params-only stream (renders the
  hello-triangle).
- **Scale-to-fit preview transform**: a loaded mesh renders centered at the world
  origin and uniformly scaled to a reasonable size (`s = target / max_extent`),
  composed beneath the per-frame `model`. Producer helper `scripts/obj_to_arrow.py`
  encodes an OBJ into the mesh stream.
- **Per-frame camera**, resolved in precedence order: a **CV** camera from the
  `k` (3×3 intrinsics) + `pose` (4×4 camera-to-world) columns (view = `inverse(pose)`),
  else a **CG** look-at from `eye` + `target` (or `eye` + `direction`) with `up`,
  `fovy`, `aspect`, `znear`, `zfar`. Absent any camera column, an identity view and
  default perspective are used. Camera columns are optional and per-row (one row =
  one frame), so a stream can animate the camera alongside the model.
- **Per-frame instanced draw list** for multi-mesh scenes: optional parallel
  columns **`draw_mesh`** (`List<UInt32>`, 0-based indices into the leading mesh
  table) and **`draw_model`** (`List<FixedSizeList<Float32>[16]>`, per-instance 4×4
  models composed beneath each mesh's preview transform). The two lists must be
  equal length per row. A frame with a draw list renders each referenced mesh at
  its own model in one instanced batch; a frame without one falls back to drawing
  the single mesh with the frame's `model`. (`scripts/jsonl_to_arrow.py` accepts a
  convenience `"draws": [{"mesh", "model"}, …]` JSONL form and emits the two
  Arrow columns when every row provides it.)

### Compatibility
- **Backward-compatible with 0.0.2 and 0.0.1.** A params-only stream (no leading
  mesh) renders identically. Decoders accept `{0.0.1, 0.0.2, 0.0.3}`. Only the
  native path decodes a mesh-first stream today; the wasm path (mesh-first
  handling) is tracked separately.

## 0.0.2 — Superseded

### Added
- Optional input column **`model`**: `FixedSizeList<Float32>[16]`, a column-major
  4×4 **model** matrix (object → world). Supersedes `center`/`size`/`theta` when
  present.
- Optional input column **`k`**: `FixedSizeList<Float32>[9]`, a column-major 3×3
  **camera intrinsics** (pinhole `K`). Derives the projection `P`.
- Optional input column **`pose`**: `FixedSizeList<Float32>[16]`, a column-major
  4×4 **camera pose** (world-from-camera). The view matrix `V` is its inverse.
- Full MVP transform in the vertex shader: `clip = P · V · M · (position, 0, 1)`.

### Changed
- The vertex transform now runs through a single 4×4 `transform` uniform (`P·V·M`)
  instead of a `{center, size, theta}` struct. The 2D affine path is preserved by
  synthesizing `M = translate(center) · rotate_z(theta) · scale(size)`.
- Output schemas are stamped `trd.protocol.version = 0.0.2` (image schema
  unchanged).

### Compatibility
- **Backward-compatible with 0.0.1.** Decoders accept both `0.0.1` and `0.0.2`
  input. A stream with no matrix columns, or with identity matrices, renders
  byte-for-byte identically to `0.0.1`.

## 0.0.1 — Superseded

### Added
- Initial protocol. Input columns `center` / `size` (`FixedSizeList<Float32>[2]`)
  and `theta` (`Float32`); one row per frame.
- Output: four `fixed_shape_tensor<UInt8>` channels `r, g, b, a` of shape
  `[height, width]`, one row per rendered image.
- `trd.protocol.version` schema metadata key.
