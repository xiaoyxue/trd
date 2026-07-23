# trd stream protocol

The **trd stream protocol** defines the columnar wire format the rendering core
consumes and produces. It is an **Apache Arrow IPC stream**: one schema message
followed by N record-batch messages, on stdin (input) and stdout (output).

- **Semantics:** `1 row = 1 frame`. The output stream is 1:1 with the input
  stream; output batch boundaries mirror the input batches, so memory stays
  bounded to a single batch in flight.
- **Versioning:** the protocol version is carried in schema-level metadata under
  the key `trd.protocol.version` and follows `MAJOR.MINOR.PATCH`. **The renderer
  supports exactly one version at a time — currently `0.0.5` — and hard-rejects
  any other declared version.** Backward compatibility with `0.0.1`–`0.0.4` was
  **deliberately dropped** to simplify the code (see the
  [no-backward-compat policy](../../AGENTS.md)); the older version specs below are
  retained only as historical reference.
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


## Versions

Only **0.0.5** is accepted by the renderer. Earlier versions are **removed** —
their specs remain below as a historical record of how the format evolved.

| Version | Status | Summary |
|---|---|---|
| [0.0.1](./0.0.1.md) | Removed | 2D affine per frame (`center`, `size`, `theta`) → RGBA tensor images. |
| [0.0.2](./0.0.2.md) | Removed | Adds optional `model` (4×4), `k` (3×3 intrinsics), `pose` (4×4) matrix columns (the MVP + camera). |
| [0.0.3](./0.0.3.md) | Removed | Adds an optional leading **mesh** Arrow stream (`[mesh][params]` framing) with nested-list geometry and the scale-to-fit preview transform, a full **camera** (CV `k`/`pose` or CG `eye`/`target`/`direction`/`up`/`fovy`/`aspect`/`znear`/`zfar`), and a per-frame instanced **draw list** (`draw_mesh`/`draw_model` list columns) for multi-mesh scenes. |
| [0.0.4](./0.0.4.md) | Removed | Adds an optional **texture** Arrow stream (`[mesh][texture][params]` framing) — `rgba` interleaved-RGBA `fixed_shape_tensor<UInt8>[H, W, 4]` — an optional per-vertex **`uv`** mesh column, and a **textured** render mode sampling the bound texture (`Rgba8UnormSrgb`, linear, clamp-to-edge). |
| [0.0.5](./0.0.5.md) | **Current (only supported)** | **Mesh-first** `[mesh][texture?][params]`. Adds an optional per-frame **`frame_path`/`frame_url`** (`Utf8`) params column naming a background image, and a `FramePlane` drawable compositing that image beneath the scene (reused `Rgba8UnormSrgb` texture, depth-write off, `Stretch`/`Cover` fit). **Not** backward-compatible with `0.0.1`–`0.0.4`. |

See [`CHANGELOG.md`](./CHANGELOG.md) for the per-version deltas.

The accepted input version and current output version are defined in
`crates/trd-core/src/protocol.rs` (`SUPPORTED_INPUT_VERSIONS`, `PROTOCOL_VERSION`).
