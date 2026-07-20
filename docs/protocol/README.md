# trd stream protocol

The **trd stream protocol** defines the columnar wire format the rendering core
consumes and produces. It is an **Apache Arrow IPC stream**: one schema message
followed by N record-batch messages, on stdin (input) and stdout (output).

- **Semantics:** `1 row = 1 frame`. The output stream is 1:1 with the input
  stream; output batch boundaries mirror the input batches, so memory stays
  bounded to a single batch in flight.
- **Versioning:** the protocol version is carried in schema-level metadata under
  the key `trd.protocol.version` and follows `MAJOR.MINOR.PATCH`. Minor bumps are
  **additive and backward-compatible** (new optional columns); a decoder accepts
  any version in its supported set and validates a column only if it is present.
- **Playback rate:** an optional schema-metadata key `trd.stream.frame_rate`
  (float, frames/sec, default **30**) declares the stream's intended playback
  frame rate — like a video file's fps. Front-ends play at this rate (speed =
  fps); `trd-cli`'s rendered image stream copies it through so `scripts/encode.py`
  encodes the GIF/WebP at the same rate. It is version-independent (applies to
  0.0.1 and 0.0.2 alike). See the [timing model](#timing-model) below.
- **Global conventions:** matrices are **column-major** and right-handed;
  projections target **wgpu clip space** (`z ∈ [0, 1]`); the vertex transform is
  the MVP chain `clip = P · V · M · (position, 0, 1)`.

## Timing model

The animation is a **sequence of frames** (one row = one frame); playback follows
the classic video-player model:

- **`frame_rate` (the stream's fps) sets the speed** — advancing `frame_rate`
  frames per second. A higher fps plays faster. `trd-app` accepts `--fps` to
  override it.
- **Presentation is decoupled from the monitor refresh (vsync).** `trd-app`
  presents at the playback fps by default (non-vsync present mode); pass `--vsync`
  to lock to the refresh rate (Fifo).
- The same content at the same fps plays at the **same speed** regardless of
  protocol version or whether motion is authored via `theta` (0.0.1) or a `model`
  matrix (0.0.2).


## Versions

| Version | Status | Summary |
|---|---|---|
| [0.0.1](./0.0.1.md) | Superseded | 2D affine per frame (`center`, `size`, `theta`) → RGBA tensor images. |
| [0.0.2](./0.0.2.md) | Superseded | Adds optional `model` (4×4), `k` (3×3 intrinsics), `pose` (4×4) matrix columns (the MVP + camera). Backward-compatible with 0.0.1. |
| [0.0.3](./0.0.3.md) | Superseded | Adds an optional leading **mesh** Arrow stream (`[mesh][params]` framing) with nested-list geometry and the scale-to-fit preview transform, a full **camera** (CV `k`/`pose` or CG `eye`/`target`/`direction`/`up`/`fovy`/`aspect`/`znear`/`zfar`), and a per-frame instanced **draw list** (`draw_mesh`/`draw_model` list columns) for multi-mesh scenes. Backward-compatible with 0.0.2. |
| [0.0.4](./0.0.4.md) | Current | Adds an optional **texture** Arrow stream (`[mesh][texture][params]` framing) — `rgba` interleaved-RGBA `fixed_shape_tensor<UInt8>[H, W, 4]` — an optional per-vertex **`uv`** mesh column, and a **textured** render mode sampling the bound texture (`Rgba8UnormSrgb`, linear, clamp-to-edge). Backward-compatible with 0.0.3. |

See [`CHANGELOG.md`](./CHANGELOG.md) for the per-version deltas.

The accepted input versions and current output version are defined in
`crates/trd-core/src/protocol.rs` (`SUPPORTED_INPUT_VERSIONS`, `PROTOCOL_VERSION`).
