# Frame extraction & the frame-to-row mapping manifest

Boundary tooling for the **frame-compositing pipeline** (issue
[#62](https://github.com/Hong-Xiang/trd/issues/62), prep slice
[#76](https://github.com/Hong-Xiang/trd/issues/76)). It turns a source
`video.mp4` into still images plus a small **manifest** so a producer can stamp a
per-frame `frame_path` (native) / `frame_url` (browser) reference into the Arrow
scene stream, and the boundary can resolve that reference back to an image.

Like [`scripts/encode.py`](../scripts/encode.py) on egress, this lives entirely
at the **boundary**: `ffmpeg`/`ffprobe` run at the edge, and **no codec or video
logic enters `trd-core`**. The renderer only ever sees decoded pixels.

## Tool

[`scripts/extract_frames.py`](../scripts/extract_frames.py) — deterministic
`video → frames/ + manifest`:

```sh
# inside `nix develop` (needs ffmpeg/ffprobe; pyarrow for the .arrow manifest)
uv run --with pyarrow scripts/extract_frames.py \
    assets/videos/cornellbox/CameraMovement.mp4 -o output/cornellbox
```

produces

```
output/cornellbox/
├── frames/
│   ├── frame_000000.png     # row 0
│   ├── frame_000001.png     # row 1
│   └── …  frame_000249.png  # row 249
├── frames.arrow             # Arrow IPC mapping manifest
└── frames.json              # human-readable sidecar (same data)
```

Run it with no arguments to print flag guidance. Flags: `-o/--out` (default
`output/<video-stem>`), `--format png|jpg` (PNG lossless default), `--url-base`
(served-base prefix for `frame_url`, default `frames`), `--fps` (override the
recorded playback fps — does **not** resample), `--no-arrow` (emit only the JSON
sidecar, no `pyarrow` needed).

## Convention

- **Layout:** `<out>/frames/frame_%06d.<ext>` — a zero-padded, 6-digit,
  **0-based** index.
- **`row N` ↔ `frame N`.** Extraction order *is* the scene-stream order: frame
  `N` maps to Arrow row `N` (`1 row = 1 frame`, matching the stream protocol).
  ffmpeg runs with `-fps_mode passthrough -start_number 0`, so there is exactly
  one still per decoded frame (no drop/dup, no rate conversion) and the index is
  a pure function of decode order.
- **`frame_path`** is the **native** path, *relative to the manifest directory*
  (e.g. `frames/frame_000000.png`) — resolve it against the dir the manifest
  lives in.
- **`frame_url`** is the **browser** URL, relative to a served base
  (`<url-base>/frame_000000.png`) — resolve it against wherever `frames/` is
  served.

## Manifest schema (`frames.arrow`)

Arrow IPC stream, one row per frame:

| Column | Type | Notes |
|---|---|---|
| `row` | `uint32` | 0-based frame/row index (`row == frame N`). |
| `frame_path` | `utf8` | native path, relative to the manifest dir. |
| `frame_url` | `utf8` | browser URL, relative to the served base. |

Schema-level metadata:

| Key | Value |
|---|---|
| `trd.protocol.version` | `0.0.4` |
| `trd.stream.frame_rate` | source fps (e.g. `25.0`) — the intended **playback rate**, matching the stream protocol's [`trd.stream.frame_rate`](protocol/README.md) so display/egress plays back at the right speed. |
| `trd.frames.width` / `trd.frames.height` | frame pixel dimensions. |
| `trd.frames.count` | number of frames. |

`frames.json` mirrors the same data (`width`/`height`/`fps`/`count` + a
`frames[]` array of `{row, frame_path, frame_url}`) for inspection and for tools
that would rather not read Arrow.

## Determinism

Re-extracting the same clip yields **byte-identical** frames and manifest:
frames come straight from the decoder (lossless PNG, `rgb24`), and the manifest
is a pure function of the frame count + probed metadata. This makes the step safe
to cache and to assert on in tests.

## Scope

This slice (#76) covers only the offline extraction + mapping. Consuming the
`frame_path`/`frame_url` reference as a background frame-plane in the scene, the
GPU composite, and the raw egress → GIF are downstream slices
([#63](https://github.com/Hong-Xiang/trd/issues/63),
[#65](https://github.com/Hong-Xiang/trd/issues/65)) of the #62 pipeline.
