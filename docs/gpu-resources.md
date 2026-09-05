# GPU asset resources

`GpuResourceManager` is the renderer's owner of addressable asset residency
(#366). It manages mesh identity, upload, lookup, mutation and removal. It is
not a registry of every wgpu object in the application.

## Ownership

```text
Caller / loader
  CPU Mesh and Texture inputs -- borrowed during upload --> Renderer
  SurfaceTarget / TextureTarget -- borrowed per render --> Renderer

Renderer
  GpuResourceManager
    private live-ID lookup and reusable mesh slots
    MeshGpu
      existing VertexBuffer / IndexBuffer wrappers
      exclusive BoundTexture / BoundMaterialMaps
      base transform and mesh appearance
  existing pipelines, uniforms, gizmos, attachments, picking and frame plane

Scene / Draw / Primitive
  non-owning MeshId + per-instance placement
```

The manager is the single authority for its resident assets. Its private slot
storage is a mechanism, not another resource manager. Mesh-exclusive allocations
remain directly owned by `MeshGpu`; there is no redundant public buffer registry.

CPU geometry is borrowed during upload, not cloned into a retained registration
table. The caller decides whether to retain its source assets. The renderer keeps
only an immutable initial `Vec<MeshId>` in wire-table order; runtime additions
return IDs directly and do not rewrite that original mapping.

Render targets remain caller-owned. Constant gizmos, viewport attachments,
uniforms and other frame subsystems keep their existing owners and update
policies. They do not acquire asset handles merely because they use GPU memory.

## Identity and lifetime

`MeshId` is an opaque, non-owning `Copy` value. The manager mints a fresh identity
for every upload, including when a private slot is reused. Independent renderers
upload the same CPU mesh under different IDs.

The internal issuer never recycles IDs; exhaustion is an error rather than wrap.
There is no public integer-to-ID conversion, storage index accessor or serialized
representation. A removed or foreign ID produces `MeshResourceError::NotResident`
instead of naming another mesh.

Copying a Scene does not retain GPU resources. Removing an object from the GUI
does not automatically remove a mesh still used by another object; callers
request resource removal only when they intend to unload it. A retained stale
Scene then fails explicitly.

Removal invalidates the live lookup, takes the resident record and destroys its
exclusive GPU allocations. Queue servicing preserves the existing release
behavior. This does not promise synchronous physical reclamation: GPU work
already in flight can defer it.

No independent shared-texture/material registry is introduced in this slice.
When actual shared resources require one, dependencies must be explicit and
removal of a resource with live dependants must fail without mutating either
resource. An exclusive binding wrapper must not also independently destroy a
texture owned by a shared registry.

## Reuse the existing wrappers

| Type | Responsibility retained |
|---|---|
| `VertexBuffer<T>` / `IndexBuffer` | Typed geometry layout/count and buffer usage |
| `InstanceBuffer<T>` | Per-frame capacity, growth and upload |
| `BoundUniform` / `BoundSceneSlots` | Buffer/binding consistency and aligned dynamic offsets |
| `BoundTexture` | Exclusive albedo upload, replacement and binding |
| `BoundMaterialMaps` | Linear material maps, normal-map mip generation and group-3 binding |
| `Attachment` / `ViewportAttachment` / `MeshAttachments` | Allocation, resizing and matching MSAA/depth attachments |

The CPU `Texture` trait and `ImageData` remain upload inputs, not GPU-residency
records. Material maps retain their GPU texture/view state rather than whole
CPU image copies. Replacing one map uploads only that map and rebuilds the binding
against the unchanged companion; only the superseded exclusive allocation is
destroyed. Defaults, color-space rules and mip generation stay unchanged.

New shared-buffer suballocation, texture pooling or dense compaction must be
justified by a real requirement. They are not prerequisites for handle safety.
GPU ranges cannot be recycled merely because a CPU slot is free: any later
suballocator must track when the GPU has finished using each range.

## Wire boundary

`WireDraw` contains a `MeshTableIndex`, while runtime `Draw` contains `MeshId`.
The protocol still carries mesh-table rows; its encoding and version do not
change.

Input can be validated before a device exists: mesh rows are checked against
the decoded mesh count, alongside camera/protocol validation. Upload then returns
the initial ID snapshot. Scene assembly resolves rows through that plain slice;
it needs neither a GPU device nor a CPU registration framework.

`RenderOptions` is an ordinary non-generic value. A wire-only grid-mesh row is
passed separately at the stream/scene boundary and resolved to the corresponding
handle before the common overlay assembler runs. This avoids cloning appearance
or environment data merely to convert one selector.

The mapping preserves implicit row zero, absent versus empty draw lists and
shadow row validation. Removing an initial mesh leaves its original row bound to
the invalidated ID, not to a replacement in its former slot. `mesh_count()` is
live resource count, never a wire-table bound.

The GUI keeps `SceneObject { mesh, transform, mode, appearance }`; object rows,
pick results and resource identities remain distinct. There is no numeric-row
fallback when an identity is missing.

## Preparation, ordering and PBR

Validate every mesh-backed primitive before acquiring or writing its target.
Validate all layers before the first layer writes. Picking follows the same
residency rule without renumbering the original draw list.

The manager resolves a handle to its GPU record and private slot; Renderer
prepares instances and records the draw. Each `record_*` binds the pass state it
needs, without depending on another record's preceding bindings.

Keep the existing `(layer, variation, private slot)` ordering with complete
identity in batch equality. Sorting only by the new ID would reorder overlays
when an early slot is reused.

Every mesh can become shaded, so private mesh slots continue to select the
private PBR uniform slots. Allocation spans include holes; growth and reuse
retain their existing dirty-marking behavior. Public IDs are never converted to
uniform offsets. Texture replacement does not dirty material uniforms.

## Scope and validation

This is an ownership and identity migration, not a shader/IBL change. Additional
geometry, multi-part model grouping, independent instance materials, bindless
rendering, global pools and streaming eviction remain separate features.

Downstream GPU-free unit tests use the opt-in `test-support` development feature
to mint nonresident identity fixtures. Production callers cannot construct IDs
from integers; real render integration tests obtain IDs from an actual upload.
The development helper does not register resources or create a second asset API.

The change is **L3**. Required coverage includes source lifetime after upload,
foreign/stale IDs, slot reuse, single-map texture replacement, GUI deletion and
re-addition, picking, PBR growth and whole-layer validation. Run complete L1-L3
on both platforms and keep unavailable physical-device cases explicit.

The reference-only commit `c5af1df` is separate: historical old-algorithm replay
first reproduced all 12 original Stage 2 references exactly, then the references
were updated to the already-established main output. Use
`TRD_STRICT_GOLDENS=1` with the full golden suite; neither shaders, Arrow fixtures
nor tolerances change in this ownership migration.
