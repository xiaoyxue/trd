# trd stream protocol

The **trd stream protocol** defines the columnar wire format the rendering core
consumes and produces. It is an **Apache Arrow IPC stream**: one schema message
followed by N record-batch messages, on stdin (input) and stdout (output).

- **Semantics:** one **params** row = one rendered frame. Mesh/texture/frames
  rows are indexed resources. Output is 1:1 with params, and output batch
  boundaries mirror params batches.
- **Versioning:** the protocol version is carried in schema-level metadata under
  the key `trd.protocol.version` and follows `MAJOR.MINOR.PATCH`. **The renderer
  supports exactly one version — currently `0.0.6` — and hard-rejects any other
  or missing version.** The protocol is deliberately not backward compatible
  (see the [policy](../../AGENTS.md)).
- **Table identity:** every input sub-stream declares `trd.table.kind`; schemas
  are not guessed from column names.
- **Playback rate:** an optional schema-metadata key `trd.stream.frame_rate`
  (float, frames/sec, default **30**) declares the stream's intended playback
  frame rate — like a video file's fps. Front-ends play at this rate (speed =
  fps); `trd-cli`'s rendered image stream copies it through so `scripts/encode.py`
  encodes the GIF/WebP at the same rate.
- **Global conventions:** matrices are **column-major** and right-handed;
  projections target **wgpu clip space** (`z ∈ [0, 1]`); the vertex transform is
  the MVP chain `clip = P · V · M · (position, 1)`.

## Timing model

The animation is a **sequence of frames** (one row = one frame); playback follows
the classic video-player model:

- **`frame_rate` (the stream's fps) sets the speed** — advancing `frame_rate`
  frames per second. A higher fps plays faster. `trd-app` accepts `--fps` to
  override it.
- **Presentation is decoupled from the monitor refresh (vsync).** `trd-app`
  presents at the playback fps by default (non-vsync present mode); pass `--vsync`
  to lock to the refresh rate (Fifo).
- The same content at the same fps plays at the **same speed**.


## Specification

The protocol is **`0.0.6`-only**; there is a single, self-contained spec:

- **[0.0.6](./0.0.6.md)** — `[mesh][texture?][frames?][params]`. Per-mesh
  geometry (`position`/`color`/`uv`/`index`), an optional **texture** table
  (`rgba`), optional inline background **frames** (`frame_bytes` /
  `frame_pixels`), and per-frame params: `model`, a **camera** (CV `k`/`pose` or CG
  `eye`/`target`/`direction`/`up`/`fovy`/`aspect`/`znear`/`zfar`), an instanced
  **draw list** (`draw_mesh`/`draw_model`), and an optional background selected
  by inline `frame_id` or external `frame_path`/`frame_url`.

Earlier iterations were removed; the renderer hard-rejects any version other
than `0.0.6`.

The accepted input version and current output version are defined in
`crates/trd-core/src/protocol.rs` (`SUPPORTED_INPUT_VERSIONS`, `PROTOCOL_VERSION`).
