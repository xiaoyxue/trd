# GPU mesh resource identity

**Design for #366; not implemented yet.** This document fixes the contract for
the later handle migration. The current renderer still publishes mesh-store
indices. No protocol, shader, resource lifetime or rendering behavior changes
with this document.

The goal is narrow: a scene names a mesh by an opaque identity, and the renderer
resolves that identity to its resident resources. Deleting a mesh must never let
an old scene silently draw a replacement.

## Decisions

| Question | Decision |
|---|---|
| D1: manager scope | Evolve `MeshStore` into `MeshResources`, not a generic manager around another store. Manage only caller-supplied, removable meshes. |
| D2: handle and generation | An opaque `MeshId(u64)`, minted once and never reused within its issuing process/module instance. No public slot or per-slot generation. Preserve private slot ordering when batching. |
| D3: wire boundary | `MeshTableIndex` resolves through a CPU-owned registration table in scene assembly. Table validity and GPU residency remain separate errors. |
| D4: dispatch seam | Resolve a handle, then record the primitive. Each `record_*` binds its own complete pass state. |
| D5: PBR slot | Keep one PBR slot per private mesh-storage slot, including holes in the allocation span. No shaded-only allocator in this migration. |
| D6: public surface | Mesh-addressed operations take `MeshId`; wire draws keep table indices. Put GUI identity and appearance beside each object's transform. Do not add unrelated API consolidation. |

## D1: ownership without another forwarding layer

`Renderer` owns `MeshResources`, and `MeshResources` owns the resident `MeshGpu`s.
It replaces the existing store; it does not wrap a second public or private
manager. `MeshGpu` remains the thin geometry/textures/appearance record, including
its explicit destruction behavior. Private slot allocation stays simple.

The following are constraints, not candidates for later migration in this work:

- Gizmos remain enum-addressed: grid plane and selection state already identify
  constant geometry without allocation or stale references.
- Pipelines, uniforms, environment, frame plane, attachments, instances and
  picking remain singleton subsystems owned directly by `Renderer`.
- No `GeometryStore` trait, resource-kind enum with unused variants, generic
  `ResourceArena<T>`, or buffer-access trait is introduced.

The addressing implementation needs a live `MeshId -> private slot` lookup and
reusable slots. Each occupied slot stores its identity with its `MeshGpu`; there
are no parallel metadata vectors whose lengths callers must synchronize.
Removal deletes the lookup entry, destroys the GPU resources and services the
queue exactly as today. Reuse installs a new identity in the vacant slot.

## D2: identity, invalidation and ordering

The public identity is a `Copy + Eq + Hash` integer newtype with private bits:

```rust
pub struct MeshId(u64);
pub struct MeshTableIndex(u32);
```

These are target shapes, not declarations of an already shipped API.
`MeshTableIndex` may expose checked boundary construction and an indexing
accessor. `MeshId` has no public integer constructor, indexing accessor,
`From<u64>`, serialization, or stable external representation.

One internal, device-free issuer allocates monotonically increasing `MeshId`s
for both initial CPU registrations and runtime additions. It is shared across
resource managers in the same process/module instance, safe for concurrent
registration, and must fail explicitly at exhaustion rather than wrap. Wasm
module instances are separate identity domains; transferring handles between
them or persisting them is unsupported. This adds no browser API or platform
conditional to a shared crate.

**No explicit generation is needed because identities never repeat.** The
manager never exposes a slot token; a recycled slot cannot recreate its former
public ID. There is no second long-lived internal handle whose generation must
also be maintained. A per-slot `(index, generation)` by itself was rejected:
two independently created managers can issue the same pair.

Fresh registrations in two renderers have distinct IDs, so passing one
renderer's ID to the other cannot accidentally name its first mesh. Uploading
the *same CPU registration* to two renderers may deliberately preserve its
logical identity: each renderer still needs its own explicit upload. A resource
not uploaded to the receiving renderer fails its residency lookup.

### Sorting is not identity

Today mesh slot order determines submission order within each render mode,
including depth-disabled wireframes and AABBs. Sorting by a newly minted
monotonic ID would change that order after deleting an early slot and refilling
the hole. Widening `Primitive::sort_key` to `u64` alone is therefore insufficient.

During batching, resolve the handle to its private slot and base model. Sort
with the existing `(layer, variation, private geometry slot)` precedence, with
the full `MeshId` as an identity tie-breaker. Non-mesh geometry retains its
existing enum order. The resolved sort key is renderer-private scratch; it never
becomes a field on `Primitive` or `DrawableObject`.

Batch equality continues to compare the complete `Primitive`, including the
complete `MeshId`. Never truncate the ID to `u32` or group on the slot alone.
The current argument-free `Primitive::sort_key` must be split so its
layer/variation policy remains with the taxonomy while storage ordering comes
from resolution. Stable ordering among instances of the same primitive remains
unchanged.

## D3: registration before residency

Three different addresses must not become another overloaded integer:

| Type | Meaning | Owner |
|---|---|---|
| `MeshTableIndex` | A row of the current wire mesh table | Protocol/CPU registration |
| `MeshId` | A registered logical mesh identity | Scene and caller |
| Private mesh slot | A resident record and its PBR uniform offset | `MeshResources` only |

The initial CPU registration table, called `MeshTable` here, owns an ordered
sequence of registered meshes, each pairing its `MeshId` with its CPU `Mesh`.
It mints identities once when the decoded mesh table is registered, not once per
frame. Its row-to-ID view can be borrowed by scene assembly without a device,
queue, `Renderer`, or GPU-manager borrow. This is a CPU asset table, not a mirror
of resident slots.

Construction uploads those same registered entries into `MeshResources`; it
must not mint a different set of IDs while uploading. Existing convenience
constructors can register internally, but must make their CPU row-to-ID binding
available to callers. The explicit registered-input constructor supports the
native reader's pre-device validation/assembly path. Runtime `add_mesh` uses the
same internal issuer and returns a fresh ID directly; it does not append to or
reinterpret a stream's mesh table.

```text
decoded mesh table -> CPU registration: [(MeshId, Mesh), ...]
                           |                         |
                      row-to-ID view            explicit upload
                           |                         |
Draw { MeshTableIndex }    |                   MeshResources
             |             |                 MeshId -> slot -> MeshGpu
             +-------> scene assembly                ^
                           |                         |
              Primitive::Mesh { MeshId, mode } ------+
```

`Scene::try_from_frame(frame, &mesh_table, options, frame_fit)` is the wire
entry. Its shared resolution helper replaces the count-only check. It resolves
the explicit draw list, or implicit mesh row zero, before calling the common
overlay/scene assembler. The native window path must use this same helper,
rather than retain its separate `mesh_id < mesh_count` loop.

`Draw` remains the wire record, with a `MeshTableIndex` rather than a resident
handle. A distinct handle-bearing `ResolvedDraw` feeds `Scene::from_draws` and
picking; GUI-authored draws already have IDs and do not masquerade as wire rows.
Resolve wire mesh-selector options, including `show_local_grid_mesh`, through
the same table at this boundary. Object selection indices remain object/draw
indices, not resource IDs.

### Two validations, two errors

`SceneError::MeshIndexOutOfRange` remains a table-boundary error, reporting the
bad `MeshTableIndex` and table length, not a count of GPU slots. It must be
possible to raise it before any device exists. Preserve the protocol's current
row validation, absent-versus-empty draw-list behavior, and shadow encoding;
this migration does not relax validation of a wire field just because the
selected primitive does not use mesh geometry.

Renderer-side lookup reports `MeshResourceError::NotResident { mesh: MeshId }`
for a removed, foreign or not-yet-uploaded resource. It need not distinguish
these cases by retaining every retired ID forever. A CPU table may still name
a deleted registration; resolution must not silently substitute the next
occupant of its former GPU slot.

Validate all mesh-bearing primitives before target acquisition, uniform writes,
or command encoding. A bad scene fails as a whole; it does not partially draw,
silently skip, or accidentally reuse stale buffers. Apply the same rule to
layered rendering and picking. Public entry points that currently cannot return
this error must become fallible, and every shell must surface it through its
existing error/log/UI path.

## D4: resolve, then record

Resolution supplies a resident resource and its private slot. `record_mesh` and
`record_aabb_box` consume that validated information, not a public array index.
Picking uses the same residency rules and keeps the original draw index for
the returned hit; filtering or batching must not renumber selectable objects.

The dispatch seam stays at `record_*`. Each body binds its own pipeline, bind
groups and instance buffers at entry, relying on no previous record and
restoring nothing at exit. Do not turn this into a trait exposing buffers or
move GPU ownership into the scene.

## D5: deliberately keep derived PBR slots

After handle validation, `private mesh slot == PBR slot` remains intentional.
Every stored mesh can switch to `RenderMode::Shaded` next frame, so allocating
only for currently shaded draws introduces another lifecycle with no measured
benefit for this scope.

`write_pbr` and `record_mesh` obtain the same private slot from the manager;
neither casts a `MeshId` into a dynamic uniform offset. Slot capacity still
covers the allocation span, not just the live mesh count. Initial upload, growth,
hole reuse, removal and appearance edits retain the existing dirty-marking
rules. Texture edits do not dirty PBR uniforms. The slot buffer and its bind
group grow together, and growth rewrites all live slots before the next draw.

An independently allocated PBR slot is deferred until a new resource kind or a
measurement justifies it. It is not a prerequisite for opaque mesh identity.

## D6: target public API and GUI ownership

The current surface in `render/mesh_store.rs` has **13** public methods, not the
11 quoted in the original issue. Map all of them rather than hide unchanged
operations behind a new name:

| Current operation | Target contract |
|---|---|
| `mesh_count()` | Returns the live resource count; never a validation bound or slot capacity. |
| `add_mesh(&Mesh)` | Returns `Result<MeshId, MeshResourceError>`; fresh identity, including when a private hole is reused. |
| `remove_mesh(usize)` | Takes `MeshId`, returns `Result<(), MeshResourceError>`; repeated/stale removal is an explicit error. |
| `set_texture(texture)` | Retains the single-mesh convenience by targeting the initial registered row-zero ID; errors if it is no longer resident, never retargets a replacement. |
| `set_mesh_texture(usize, texture)` | Takes `MeshId` and returns a result. |
| `set_mesh_metallic_roughness_texture(usize, texture)` | Takes `MeshId` and returns a result. |
| `set_mesh_normal_texture(usize, texture)` | Takes `MeshId` and returns a result. |
| `mesh_appearance(usize)` | Takes `MeshId`; returns a result containing the borrowed appearance. |
| `set_appearance(target, appearance)` | `MeshTarget::One(MeshId)` or `All`; returns a result. |
| `set_disney_material(target, material)` | Same typed target/result, through the existing private appearance-edit path. |
| `set_image_based_lighting(target, ibl)` | Same typed target/result and dirty marking. |
| `set_tone_mapping(target, tone_mapping)` | Same typed target/result and dirty marking. |
| `set_pbr_debug_view(target, debug_view)` | Same typed target/result and dirty marking. |

`MeshTarget::All` edits all live records, including the valid empty set.
`MeshTarget::One` never silently ignores a nonresident ID. Keep the existing
private `edit_appearance` as the sole mutation/dirty-marking path; exposing a
generic edit closure or collapsing texture setters is not part of this change.
The internal residency error also propagates through `RenderError`.

**The GUI cannot drop the object-to-mesh relationship.** Replace its parallel
vectors with records, rather than merely renaming `mesh_ids`:

```rust
struct SceneObject {
    mesh: MeshId,
    transform: ObjectTransform,
    mode: RenderMode,
    appearance: MeshAppearance,
}
```

`SceneState::objects` becomes `Vec<SceneObject>`. Registration supplies a real
ID when an object is constructed; scene seeding must not invent `0..n` IDs or
fall back to an object's row number. The UI's selected row indexes these
records. Removing a row does not change surviving resource identities, and
removing its GPU resource is explicit. Where multiple objects share a mesh,
removing one object must not destroy a resource still referenced by another.

This co-location is part of the implementation's caller migration, not a claim
that handles automatically remove bookkeeping. It must cover native/web viewers,
video-editor catalog swaps, appearance editing and pick results together.
Do not generalize this into multi-part model ownership.

## Ordering and delivery

#366 lands this design first. The later implementation is one vertical
handle/lifecycle migration, including its callers and inline/integration tests;
do not merge a half-migrated raw-index/handle API. Align #225's N2 with this
contract: `MeshTableIndex` at the wire boundary and opaque `MeshId` afterwards,
not a publicly constructible `MeshId(u32)` with `.index()`. #225's object-index
and frame-ID work remains separate.

Land that migration **before #222 step 3** so model/submesh grouping can use
stable resource identity. Handles alone do not make `import_glb` accept
multi-primitive assets; grouping and per-part materials remain #222/#160/#161.
New geometry representations (#222 step 2) remain out of scope.

Draft #368 overlaps protocol/scene/delivery callers, #291 overlaps uniform
construction, and #285 overlaps renderer preparation. Reconcile their actual
heads before implementation; do not copy their changes into this design PR.
Protocol `0.0.6`, video-edit document `0.2.0`, Arrow producers and fixtures are
unchanged by the identity migration.

### Implementation acceptance

- CPU-only registration/assembly tests cover independent tables, implicit row
  zero, empty draw lists, bad rows and wire mesh-selector resolution.
- Lifecycle tests cover upload/remove/reuse, repeated removal, stale appearance
  and texture edits, a stale scene, and foreign-manager IDs. Exhaustion fails
  without wrapping. Retired IDs never resolve to replacements.
- Batching tests cover multiple instances sharing one handle, independent
  handles, deletion/reuse ordering and unmodified overlay layer precedence.
- GPU tests exercise PBR buffer growth, hole reuse with a different material,
  render-mode changes, layered draws, AABBs and picking after deletion/reuse.
- GUI coverage exercises deleting the middle of three objects, loading another,
  editing/picking the survivors, and catalog replacement without invented IDs.
- The implementation declares **L3**, runs every required gate on both
  platforms, and records the full matrix/handoff in the PR and issue. Existing
  golden outputs must be **bit-identical**, MSAA on/off and PBR ACES/Reinhard;
  tolerance-based success alone does not establish this stronger requirement.
  Do not regenerate goldens to accommodate re-keying.

This documentation-only PR is **L1**. Its acceptance is the recorded D1-D6
contract and linked issue alignment, not a claim that the runtime migration
or its L3 coverage has already happened.
