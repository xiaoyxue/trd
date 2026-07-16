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
- **Global conventions:** matrices are **column-major** and right-handed;
  projections target **wgpu clip space** (`z ∈ [0, 1]`); the vertex transform is
  the MVP chain `clip = P · V · M · (position, 0, 1)`.

## Versions

| Version | Status | Summary |
|---|---|---|
| [0.0.1](./0.0.1.md) | Superseded | 2D affine per frame (`center`, `size`, `theta`) → RGBA tensor images. |
| [0.0.2](./0.0.2.md) | Current | Adds optional `model` (4×4), `k` (3×3 intrinsics), `pose` (4×4) matrix columns (the MVP + camera). Backward-compatible with 0.0.1. |

See [`CHANGELOG.md`](./CHANGELOG.md) for the per-version deltas.

The accepted input versions and current output version are defined in
`crates/trd-core/src/protocol.rs` (`SUPPORTED_INPUT_VERSIONS`, `PROTOCOL_VERSION`).
