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

The runtime constant and `scripts/protocol_version.py` both declare **0.0.7**.
Current producers share that Python version; `protocol_schema.py` generates
`0.0.7.schema.json` and checks the committed params/GLB fixtures.
Older declared versions are rejected for both params and mesh tables.

Old mesh/texture/inline-frame/reference producers are removed. OBJ parsing and
image loading remain offline helpers for GLB bundling. The golden generator
authors current params/GLB inputs with external stills; it never changes
expected golden PNGs.

```sh
uv run --with pyarrow python scripts/protocol_schema.py --check
```

Protocol migration is not a compatibility branch. Do not restore an earlier
version or restamp an old resource envelope as current input.

## Timing model

The standalone scene viewer advances params rows at `trd.stream.frame_rate`
(positive frames/second, default 30). Native `--fps` overrides that presentation
rate; vsync remains separate.

The video editor instead uses the external video's actual timeline and the
source's explicit frame identity/PTS mapping. It must not derive a made-up
timestamp or duplicate sparse rows to match the full video frame count.

## Earlier versions

Earlier specifications and schemas are removed from the current tree. Their
history is available in Git, not as another selectable protocol.

Application APIs, placement, all-frame editing and the two-round workflow are
documented in [scene documents](scene-documents.md).
