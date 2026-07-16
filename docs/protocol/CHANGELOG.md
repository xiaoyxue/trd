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

## 0.0.2 — Current

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
