# trd scene protocol

**The current contract is 0.0.7**, agreed in #367 and implemented through #370:

```text
[params] [mesh?]
```

Each bracket is a complete Apache Arrow IPC stream. Params come first; the
optional mesh table stores only UUID `mesh_id` and original self-contained
`glb` bytes. There are no Arrow OBJ geometry, texture, inline-frame or GLB
path/URL resource tables. Video remains an independent input.

| Topic | Rule |
|---|---|
| Specification | **[0.0.7](0.0.7.md)** and **[machine-readable contract](0.0.7.schema.json)** |
| Params-only | Placement/reference cube without an external mesh resource |
| One/multiple GLBs | One resource row per UUID, shared by corresponding object bindings |
| Rendering subset | [FHC placement/camera/model/binding fields](../../assets/schemas/trd-render-sub-schema.md), plus the retained separate CG/CV adapter |
| Matrices | Tracked source `model` is row-major; CG/CV arrays and internal renderer matrices are column-major. Never silently reinterpret one as the other. |
| Editing/export | Preserve complete source arrays, metadata and batch boundaries. Editor transforms persist in all corresponding existing sparse rows. |
| Video | External decoding/player; missing source row means video-only, not a fabricated placement |
| Output | One planar RGBA image row per params row, preserving params batch boundaries |
| Compatibility | No backward compatibility. Versioned 0.0.7 input must reject other/missing versions; unversioned FHC source ingestion is an explicit adapter, not a legacy fallback. |

## Implementation migration status

**Documentation defines 0.0.7; this is not a claim that the atomic runtime
cutover is complete.** At the `c284478` implementation checkpoint,
`crates/trd-core/src/protocol/mod.rs`, `scripts/protocol_schema.py` and several
legacy producers still stamp `0.0.6`, despite current applications reading the
params/GLB shape. That mismatch remains a release blocker on #367/#370.

The runtime constant, all current producers, fixture metadata and generator must
move together before publishing 0.0.7 acceptance. Do not re-enable old application
paths or merely restamp old mesh/texture/frames payloads. The existing
`protocol_schema.py` still describes the archived format; its `--check` is not
a validator for the new `0.0.7.schema.json` contract yet.

## Timing model

The standalone scene viewer advances params rows at `trd.stream.frame_rate`
(positive frames/second, default 30). Native `--fps` overrides that presentation
rate; vsync remains separate.

The video editor instead uses the external video's actual timeline and the
source's explicit frame identity/PTS mapping. It must not derive a made-up
timestamp or duplicate sparse rows to match the full video frame count.

## Historical specification

[0.0.6](0.0.6.md) and [its schema](0.0.6.schema.json) are retained only as
historical documentation of `[mesh][texture?][frames?][params]`. They are not
the current contract or instructions for a compatibility mode.

Application APIs, placement, all-frame editing and the two-round workflow are
documented in [scene documents](scene-documents.md).
