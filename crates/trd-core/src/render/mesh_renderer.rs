//! The persistent [`MeshRenderer`]: a decode-once GPU mesh store, instance
//! batching, and the branch-free [`Scene`](super::Scene) encode.
//!
//! The renderer is a composition of a few cohesive parts, each with a single
//! job, so no one struct is a grab-bag of wgpu handles:
//! - [`MeshPass`] — the three mesh pipelines (filled/wireframe/textured) and the
//!   camera `P·V` uniform they share.
//! - [`MeshStore`] — the uploaded [`MeshGpu`]s, the shared axes gizmo, and the
//!   growable per-instance model buffer; also walks a [`Scene`] into draw batches.
//! - [`BoundTexture`](super::BoundTexture) — the mesh albedo sampled by textured
//!   draws (#20).
//! - [`FramePlane`](super::FramePlane) — the background video frame plane (#63).

use std::ops::Range;

use super::bound_texture::BoundTexture;
use super::frame_plane::FramePlane;
use super::*;

use crate::math::Matrix4;
use crate::texture::Texture;

/// An index buffer plus its element count — one `draw_indexed` range.
struct IndexBuf {
    buffer: wgpu::Buffer,
    count: u32,
}

impl IndexBuf {
    fn new(device: &wgpu::Device, label: &str, indices: &[u32]) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let count = u32::try_from(indices.len()).expect("index count exceeds u32::MAX");
        Self { buffer, count }
    }
}

/// A vertex buffer paired with one index buffer: a self-contained indexed draw
/// (e.g. a mesh's AABB box, which carries its own corner vertices). Meshes reuse
/// one vertex buffer for both their filled triangles and wireframe edges, so
/// those keep a shared vertex buffer plus two [`IndexBuf`]s instead.
struct IndexedGeometry {
    vertex_buffer: wgpu::Buffer,
    index: IndexBuf,
}

/// A mesh uploaded to the GPU. Its `vertex_buffer` feeds both the filled
/// `triangles` and the deduped wireframe `edges` (#38); the `aabb` overlay (#42)
/// is a standalone box (own corner vertices + 12-edge `LineList`). `base_model`
/// is the base (preview) transform pre-multiplied beneath every per-frame
/// instance model (`effective = model · base`).
struct MeshGpu {
    vertex_buffer: wgpu::Buffer,
    /// Parallel vertex buffer for the Disney PBR path (`disney.wgsl`): the same
    /// positions + UVs as `vertex_buffer`, but with a derived smooth shading
    /// **normal** in place of the vertex color. Reuses the `triangles` index
    /// buffer. Built once per mesh; only bound by [`RenderMode::Pbr`] draws.
    pbr_vertex_buffer: wgpu::Buffer,
    triangles: IndexBuf,
    edges: IndexBuf,
    aabb: IndexedGeometry,
    base_model: Matrix4,
}

fn upload_mesh(device: &wgpu::Device, mesh: &Mesh, base_model: Matrix4) -> MeshGpu {
    use wgpu::util::DeviceExt;

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("trd mesh vertex buffer"),
        contents: bytemuck::cast_slice(&mesh.vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let triangles = IndexBuf::new(device, "trd mesh index buffer", &mesh.indices);
    let edges = mesh.edge_indices();
    let edges = IndexBuf::new(device, "trd mesh edge buffer", &edges);

    // PBR vertex buffer (#): derive area-weighted smooth normals (the assets have
    // no `vn`) and pack position + normal + UV for `disney.wgsl`, reusing the
    // triangle index buffer above.
    let normals = compute_smooth_normals(&mesh.vertices, &mesh.indices);
    let pbr_vertices: Vec<PbrVertex> = mesh
        .vertices
        .iter()
        .zip(&normals)
        .map(|(v, &normal)| PbrVertex {
            position: v.position,
            normal,
            uv: v.uv,
        })
        .collect();
    let pbr_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("trd mesh pbr vertex buffer"),
        contents: bytemuck::cast_slice(&pbr_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // AABB overlay box: the mesh's own bounding box (mesh-local coords) as 8
    // colored corner vertices + a 12-edge line list. Built once per mesh; drawn
    // only when the scene contains an `AabbBox` for this mesh.
    let aabb_vertices: Vec<Vertex> = mesh
        .aabb()
        .corners()
        .iter()
        .map(|c| Vertex {
            position: c.to_array(),
            color: AABB_COLOR,
            uv: [0.0, 0.0],
        })
        .collect();
    let aabb_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("trd mesh aabb vertex buffer"),
        contents: bytemuck::cast_slice(&aabb_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let aabb_edges = IndexBuf::new(device, "trd mesh aabb edge buffer", &AABB_EDGE_INDICES);

    MeshGpu {
        vertex_buffer,
        pbr_vertex_buffer,
        triangles,
        edges,
        aabb: IndexedGeometry {
            vertex_buffer: aabb_vertex_buffer,
            index: aabb_edges,
        },
        base_model,
    }
}

fn create_instance_buffer(device: &wgpu::Device, capacity: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("trd mesh instance buffer"),
        size: capacity as u64 * std::mem::size_of::<InstanceRaw>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Binds `vertex_buffer` at slot 0 and `index`, then draws it over `instances`
/// (the per-instance model buffer stays bound at slot 1). Pipeline + group
/// bindings are the caller's responsibility.
fn draw_indexed(
    pass: &mut wgpu::RenderPass,
    vertex_buffer: &wgpu::Buffer,
    index: &IndexBuf,
    instances: Range<u32>,
) {
    pass.set_vertex_buffer(0, vertex_buffer.slice(..));
    pass.set_index_buffer(index.buffer.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..index.count, 0, instances);
}

/// Which geometry a [`DrawCommand`] binds. The `usize` is a mesh id (index into
/// [`MeshStore::meshes`]) for the mesh kinds, or a [`GridPlane::index`] for
/// `Grid`; `Axes` uses the shared gizmo geometry.
enum DrawKind {
    /// Filled triangles of a mesh (its triangle index buffer + filled pipeline).
    Filled(usize),
    /// Textured triangles of a mesh (triangle index buffer + textured pipeline,
    /// sampling the bound texture at each vertex UV) (#20).
    Textured(usize),
    /// Disney **PBR** triangles of a mesh (its dedicated position+normal+UV
    /// vertex buffer + `disney.wgsl` pipeline, lit by the virtual light rig and
    /// the bound HDR environment map). Reuses the triangle index buffer.
    Pbr(usize),
    /// Edge lines of a mesh (its deduped edge index buffer + line pipeline).
    Wireframe(usize),
    /// A mesh's AABB box (its precomputed corner geometry + line pipeline).
    Aabb(usize),
    /// A coordinate-plane grid (the shared per-plane grid vertex buffer indexed
    /// by [`GridPlane::index`], non-indexed line draw).
    Grid(usize),
    /// A contact / blob **grounding shadow** (the shared shadow quad geometry,
    /// non-indexed triangle draw, alpha-blended over the frame plane).
    Shadow,
    /// The coordinate-axes gizmo (shared vertex buffer, non-indexed line draw).
    Axes,
}

/// One instanced draw recorded while walking a [`Scene`]: the geometry to bind
/// ([`DrawKind`]) and the contiguous instance-buffer range to draw it over.
struct DrawCommand {
    kind: DrawKind,
    start: u32,
    count: u32,
}

/// Appends `bucket`'s instance models to `instances` and, when non-empty,
/// records a [`DrawCommand`] over the appended range. Grouping same-geometry
/// instances into one range preserves GPU instancing.
fn push_command(
    instances: &mut Vec<InstanceRaw>,
    commands: &mut Vec<DrawCommand>,
    kind: DrawKind,
    bucket: &[InstanceRaw],
) {
    if bucket.is_empty() {
        return;
    }
    let start = instances.len() as u32;
    instances.extend_from_slice(bucket);
    commands.push(DrawCommand {
        kind,
        start,
        count: bucket.len() as u32,
    });
}

/// The result of walking a [`Scene`] once ([`MeshStore::build_batches`]): the
/// flattened per-instance models, the [`DrawCommand`]s over them (already in
/// draw order), and the singleton background frame-plane fit (if any).
struct Batches {
    instances: Vec<InstanceRaw>,
    commands: Vec<DrawCommand>,
    frame_fit: Option<FrameFit>,
}

/// The three mesh pipelines sharing one bind-group layout, plus the camera
/// (`P·V`) uniform buffer + bind group they all bind at group 0. Filled and
/// wireframe share one explicit layout so a single camera bind group is valid
/// whichever [`RenderMode`] is active; the textured pipeline adds the albedo
/// texture at group 1.
struct MeshPass {
    filled: wgpu::RenderPipeline,
    wireframe: wgpu::RenderPipeline,
    textured: wgpu::RenderPipeline,
    /// The contact / blob grounding-shadow pipeline (alpha-blended, depth-write
    /// off); shares the untextured camera bind-group layout (group 0).
    shadow: wgpu::RenderPipeline,
    /// The Disney PBR pipeline (`disney.wgsl`): group 0 = [`pbr_uniform`], group 1
    /// = the bound albedo texture, group 2 = the HDR environment map.
    pbr: wgpu::RenderPipeline,
    /// The per-frame `PbrUniform` (camera `P·V` + world pos, material, lights),
    /// rewritten each `encode`; bound as group 0 by the PBR pipeline.
    pbr_uniform: wgpu::Buffer,
    pbr_bind_group: wgpu::BindGroup,
    camera_uniform: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
}

impl MeshPass {
    /// Constructs a `MeshPass` for `format`, building all pipelines over their
    /// bind-group layouts at `sample_count`× MSAA. `texture_layout` is the albedo
    /// texture's group-1 layout (from [`BoundTexture::layout`]), shared by the
    /// textured and PBR pipelines; `env_layout` is the PBR pipeline's group-2
    /// environment-map layout (from [`BoundEnv::layout`]). Every pipeline in the
    /// pass shares the one `sample_count` (`1` = no MSAA, single-sample).
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        texture_layout: &wgpu::BindGroupLayout,
        env_layout: &wgpu::BindGroupLayout,
        sample_count: u32,
    ) -> Self {
        // One explicit bind-group layout shared by both untextured pipelines, so
        // the single camera bind group is valid whichever RenderMode is active.
        let camera_layout = create_mesh_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trd mesh pipeline layout"),
            bind_group_layouts: &[Some(&camera_layout)],
            immediate_size: 0,
        });
        let filled = create_mesh_pipeline_with(
            device,
            format,
            &pipeline_layout,
            wgpu::PrimitiveTopology::TriangleList,
            Some(solid_depth_stencil()),
            sample_count,
        );
        let wireframe = create_mesh_pipeline_with(
            device,
            format,
            &pipeline_layout,
            wgpu::PrimitiveTopology::LineList,
            Some(overlay_depth_stencil()),
            sample_count,
        );
        // Contact / blob grounding-shadow pipeline (#110 follow-up): shares the
        // untextured camera layout (group 0), alpha-blended, depth-write off.
        let shadow = create_shadow_pipeline(device, format, &pipeline_layout, sample_count);
        // Textured pipeline (#20): group 0 = the shared camera uniform, group 1 =
        // the bound albedo texture + sampler.
        let textured_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("trd textured pipeline layout"),
                bind_group_layouts: &[Some(&camera_layout), Some(texture_layout)],
                immediate_size: 0,
            });
        let textured =
            create_textured_pipeline(device, format, &textured_pipeline_layout, sample_count);
        // Disney PBR pipeline (#): group 0 = the PbrUniform, group 1 = the shared
        // albedo texture layout, group 2 = the HDR environment map. Its group-0
        // layout differs from the camera layout, so the encode arm restores the
        // camera bind group after each PBR draw.
        let pbr_layout = create_pbr_bind_group_layout(device);
        let pbr_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trd pbr pipeline layout"),
            bind_group_layouts: &[Some(&pbr_layout), Some(texture_layout), Some(env_layout)],
            immediate_size: 0,
        });
        let pbr = create_pbr_pipeline(device, format, &pbr_pipeline_layout, sample_count);
        // The PbrUniform buffer is (re)written every frame; seed it with a neutral
        // material so an unconfigured PBR draw still renders something sane.
        let pbr_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trd pbr uniform"),
            size: std::mem::size_of::<PbrUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pbr_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trd pbr bind group"),
            layout: &pbr_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: pbr_uniform.as_entire_binding(),
            }],
        });
        // Identity params ignore the viewport (no intrinsics); each frame's
        // `write_camera` supplies the real target dimensions.
        let (camera_uniform, camera_bind_group) = create_view_proj_binding(
            device,
            &camera_layout,
            FrameParams::IDENTITY,
            Viewport {
                width: 1,
                height: 1,
            },
        );
        Self {
            filled,
            wireframe,
            textured,
            shadow,
            pbr,
            pbr_uniform,
            pbr_bind_group,
            camera_uniform,
            camera_bind_group,
        }
    }

    /// Rewrites the camera `P·V` uniform for this frame's `params`/`viewport`.
    fn write_camera(&self, queue: &wgpu::Queue, params: FrameParams, viewport: Viewport) {
        write_view_proj(queue, &self.camera_uniform, params, viewport);
    }

    /// Rewrites the Disney PBR uniform (camera `P·V` + world position, material,
    /// light rig, env gate) for this frame.
    fn write_pbr(
        &self,
        queue: &wgpu::Queue,
        params: FrameParams,
        viewport: Viewport,
        material: &PbrMaterial,
        use_env: bool,
    ) {
        let uniform = PbrUniform::new(
            params.view_proj_matrix(viewport).to_cols_array(),
            params.camera_position(),
            material,
            use_env,
        );
        queue.write_buffer(&self.pbr_uniform, 0, bytemuck::bytes_of(&uniform));
    }
}

/// The decode-once geometry store: the uploaded [`MeshGpu`]s (referenced by a
/// scene's mesh ids), the shared coordinate-axes gizmo vertices, and the
/// growable per-instance model-matrix buffer. Also walks a [`Scene`] into
/// [`Batches`], the one place mesh base models are applied.
struct MeshStore {
    meshes: Vec<MeshGpu>,
    /// The coordinate-axes gizmo geometry (six `LineList` vertices); each
    /// [`DrawableObject::CoordinateAxes`] draws it under its own model, supplied
    /// through the shared instance buffer.
    axes_vertex_buffer: wgpu::Buffer,
    /// The coordinate-plane grid geometry, one `LineList` vertex buffer per
    /// [`GridPlane`] (indexed by [`GridPlane::index`]): XY, XZ, YZ. Each
    /// [`DrawableObject::PlaneGrid`] draws the buffer for its plane under its own
    /// model, supplied through the shared instance buffer.
    grid_vertex_buffers: [wgpu::Buffer; 3],
    /// The contact / blob **grounding-shadow** quad geometry (six `TriangleList`
    /// vertices, a unit XY quad); each [`DrawableObject::BlobShadow`] draws it
    /// under its own model through the shared instance buffer, alpha-blended.
    shadow_vertex_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    instance_capacity: u32,
}

impl MeshStore {
    /// Constructs a `MeshStore`, uploading each mesh with its base (preview)
    /// model and sizing the instance buffer to at least one instance.
    fn new(device: &wgpu::Device, meshes: &[Mesh], base_models: &[Matrix4]) -> Self {
        use wgpu::util::DeviceExt;

        let gpu_meshes = meshes
            .iter()
            .zip(base_models)
            .map(|(mesh, &base)| upload_mesh(device, mesh, base))
            .collect();
        let instance_capacity = (meshes.len() as u32).max(1);
        let instance_buffer = create_instance_buffer(device, instance_capacity);

        // Coordinate-axes gizmo: six LineList vertices at the world origin. Each
        // CoordinateAxes drawable draws them under its own model, supplied via
        // the shared instance buffer (so the gizmo is not tied to a fixed model).
        let axes_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trd axes vertex buffer"),
            contents: bytemuck::cast_slice(&axes_vertices()),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Coordinate-plane grids: one LineList vertex buffer per plane (XY/XZ/YZ),
        // spanning the unit model-space square. Each PlaneGrid drawable draws its
        // plane's buffer under its own model via the shared instance buffer.
        let grid_buffer = |plane: GridPlane| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("trd grid vertex buffer"),
                contents: bytemuck::cast_slice(&grid_vertices(plane)),
                usage: wgpu::BufferUsages::VERTEX,
            })
        };
        let grid_vertex_buffers = [
            grid_buffer(GridPlane::Xy),
            grid_buffer(GridPlane::Xz),
            grid_buffer(GridPlane::Yz),
        ];

        // Contact / blob grounding-shadow quad: six TriangleList vertices (a unit
        // XY quad). Each BlobShadow drawable draws them under its own model via
        // the shared instance buffer, alpha-blended over the frame plane.
        let shadow_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("trd shadow vertex buffer"),
            contents: bytemuck::cast_slice(&blob_shadow_vertices()),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            meshes: gpu_meshes,
            axes_vertex_buffer,
            grid_vertex_buffers,
            shadow_vertex_buffer,
            instance_buffer,
            instance_capacity,
        }
    }

    fn len(&self) -> usize {
        self.meshes.len()
    }

    /// Walks `scene` once, bucketing each drawable's instance model by the
    /// geometry it draws (its base model pre-multiplied in, `effective = model ·
    /// base`), then flattens the buckets into one instance list + ordered
    /// [`DrawCommand`]s. Draw order: grounding shadows, filled, textured, PBR,
    /// grids, wireframe, AABB boxes, then axes — so the blob shadow sits under the
    /// opaque meshes, which precede the line overlays, and the plane grid sits
    /// beneath the wireframe/axes gizmos drawn over it. Out-of-range mesh ids are
    /// skipped.
    fn build_batches(&self, scene: &[DrawableObject]) -> Batches {
        let mesh_count = self.meshes.len();
        let mut filled: Vec<Vec<InstanceRaw>> = vec![Vec::new(); mesh_count];
        let mut textured: Vec<Vec<InstanceRaw>> = vec![Vec::new(); mesh_count];
        let mut pbr: Vec<Vec<InstanceRaw>> = vec![Vec::new(); mesh_count];
        let mut wireframe: Vec<Vec<InstanceRaw>> = vec![Vec::new(); mesh_count];
        let mut aabb: Vec<Vec<InstanceRaw>> = vec![Vec::new(); mesh_count];
        // One instance bucket per grid plane (XY/XZ/YZ), keyed by GridPlane::index.
        let mut grid: [Vec<InstanceRaw>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        // Contact / blob grounding-shadow instances (shared quad geometry).
        let mut shadow: Vec<InstanceRaw> = Vec::new();
        let mut axes: Vec<InstanceRaw> = Vec::new();
        // The background frame plane is a singleton overlay (there is one bound
        // frame texture); the last FramePlane in the scene wins its fit.
        let mut frame_fit: Option<FrameFit> = None;

        for object in scene {
            match *object {
                DrawableObject::Mesh {
                    mesh_id,
                    model,
                    mode,
                } => {
                    let Some(mesh) = self.meshes.get(mesh_id as usize) else {
                        continue;
                    };
                    let effective = Matrix4::from_cols_array(&model) * mesh.base_model;
                    let instance = InstanceRaw {
                        model: effective.to_cols_array(),
                    };
                    match mode {
                        RenderMode::Filled => filled[mesh_id as usize].push(instance),
                        RenderMode::Textured => textured[mesh_id as usize].push(instance),
                        RenderMode::Pbr => pbr[mesh_id as usize].push(instance),
                        RenderMode::Wireframe => wireframe[mesh_id as usize].push(instance),
                        // A Shadow draw is emitted as DrawableObject::BlobShadow by
                        // build_scene, never as a Mesh — so this arm is unreachable;
                        // skip defensively rather than panic.
                        RenderMode::Shadow => {}
                    }
                }
                DrawableObject::AabbBox { mesh_id, model } => {
                    let Some(mesh) = self.meshes.get(mesh_id as usize) else {
                        continue;
                    };
                    let effective = Matrix4::from_cols_array(&model) * mesh.base_model;
                    aabb[mesh_id as usize].push(InstanceRaw {
                        model: effective.to_cols_array(),
                    });
                }
                DrawableObject::CoordinateAxes { model } => {
                    axes.push(InstanceRaw { model });
                }
                DrawableObject::PlaneGrid { plane, model } => {
                    grid[plane.index()].push(InstanceRaw { model });
                }
                DrawableObject::BlobShadow { model } => {
                    shadow.push(InstanceRaw { model });
                }
                DrawableObject::FramePlane { fit } => {
                    frame_fit = Some(fit);
                }
            }
        }

        // Flatten every instance model into one buffer, recording a draw command
        // per non-empty group in the layered draw order.
        let mut instances: Vec<InstanceRaw> = Vec::with_capacity(scene.len());
        let mut commands: Vec<DrawCommand> = Vec::new();
        // Grounding shadows first (right after the background frame plane) so the
        // opaque content meshes composite on top and only the surrounding rim
        // darkens the floor.
        push_command(&mut instances, &mut commands, DrawKind::Shadow, &shadow);
        for (id, bucket) in filled.iter().enumerate() {
            push_command(&mut instances, &mut commands, DrawKind::Filled(id), bucket);
        }
        for (id, bucket) in textured.iter().enumerate() {
            push_command(
                &mut instances,
                &mut commands,
                DrawKind::Textured(id),
                bucket,
            );
        }
        for (id, bucket) in pbr.iter().enumerate() {
            push_command(&mut instances, &mut commands, DrawKind::Pbr(id), bucket);
        }
        for (plane, bucket) in grid.iter().enumerate() {
            push_command(&mut instances, &mut commands, DrawKind::Grid(plane), bucket);
        }
        for (id, bucket) in wireframe.iter().enumerate() {
            push_command(
                &mut instances,
                &mut commands,
                DrawKind::Wireframe(id),
                bucket,
            );
        }
        for (id, bucket) in aabb.iter().enumerate() {
            push_command(&mut instances, &mut commands, DrawKind::Aabb(id), bucket);
        }
        push_command(&mut instances, &mut commands, DrawKind::Axes, &axes);

        Batches {
            instances,
            commands,
            frame_fit,
        }
    }

    /// Uploads the flattened instance models, growing the buffer (to the next
    /// power of two) when the frame needs more instances than it holds.
    fn upload_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[InstanceRaw],
    ) {
        if instances.len() as u32 > self.instance_capacity {
            self.instance_capacity = (instances.len() as u32).next_power_of_two();
            self.instance_buffer = create_instance_buffer(device, self.instance_capacity);
        }
        if !instances.is_empty() {
            queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
        }
    }
}

/// Persistent indexed mesh renderer. A composition of a [`MeshPass`] (pipelines
/// and camera uniform), a [`MeshStore`] (decode-once geometry and instance
/// buffer), a [`BoundTexture`] (mesh albedo, #20), and a [`FramePlane`]
/// (background video frame, #63), plus a viewport-sized depth attachment. Each
/// [`encode`](Self::encode) draws a frame's [`Scene`] — an ordered list of
/// [`DrawableObject`]s — grouping instances by geometry so each buffer is drawn
/// once over a contiguous instance range. The renderer holds no mode/overlay
/// state; what to draw is entirely the scene.
pub struct MeshRenderer {
    pass: MeshPass,
    texture: BoundTexture,
    /// The bound HDR environment map reflected by [`RenderMode::Pbr`] draws.
    env: BoundEnv,
    /// The Disney material applied globally to every [`RenderMode::Pbr`] draw.
    pbr_material: PbrMaterial,
    store: MeshStore,
    frame_plane: FramePlane,
    /// The mesh pass's depth attachment, (re)created lazily in `encode` to match
    /// the viewport. Gives solid (filled/textured) meshes real z-occlusion.
    depth: Option<DepthTarget>,
    /// The mesh pass's multisampled color attachment ([`sample_count`](Self::sample_count)×),
    /// (re)created lazily in `encode` to match the viewport. The pass renders into
    /// it and resolves into the caller's single-sample `view`, so every front-end
    /// gets anti-aliased edges transparently. `None` when MSAA is disabled
    /// (`sample_count == 1`): the pass then renders straight into `view`.
    msaa: Option<MsaaColorTarget>,
    /// The mesh pass's MSAA sample count — `4` (the default,
    /// [`MSAA_SAMPLE_COUNT`]) for anti-aliased edges, or `1` to render
    /// single-sampled (no MSAA). Fixed at construction because every pipeline +
    /// the depth/color attachments must share it.
    sample_count: u32,
    /// The color format the pipelines were built for; the MSAA color target must
    /// be created with the same format.
    format: wgpu::TextureFormat,
    /// Retained so `encode` can grow GPU resources on demand without the caller
    /// threading a `&Device` through every call (`wgpu::Device` is a cheap `Arc`).
    device: wgpu::Device,
    /// The object-id **picking** pipeline (`picking.wgsl`): renders each drawn
    /// object in a flat id color into a single-sample linear target, reused by
    /// [`encode_picking`](Self::encode_picking). Built once (its own bind-group
    /// layout is structurally the camera layout, so `camera_bind_group` binds it).
    pick_pipeline: wgpu::RenderPipeline,
    /// Per-instance [`PickInstanceRaw`] buffer for the picking pass (model +
    /// id color), grown on demand like the mesh instance buffer.
    pick_instances: wgpu::Buffer,
    pick_instance_capacity: u32,
}

impl MeshRenderer {
    /// Constructs a `MeshRenderer` that derives each mesh's base (preview) model
    /// automatically via [`Mesh::preview_transform`]
    /// ([`crate::DEFAULT_PREVIEW_TARGET`]) — center + uniform scale-to-fit — so an
    /// arbitrary-unit asset renders centered at a reasonable size. A convenience
    /// constructor over [`new`](Self::new); shared by the headless
    /// [`crate::run_stream`]/`BatchRenderer` and the windowed `trd-app`.
    pub fn auto_fit(device: &wgpu::Device, format: wgpu::TextureFormat, meshes: &[Mesh]) -> Self {
        let base_models: Vec<Matrix4> = meshes
            .iter()
            .map(|mesh| {
                mesh.preview_transform(crate::DEFAULT_PREVIEW_TARGET)
                    .matrix()
            })
            .collect();
        Self::new(device, format, meshes, &base_models)
    }

    /// Constructs a `MeshRenderer` over one or more meshes, each paired with an
    /// explicit base (preview) model that is pre-multiplied beneath every
    /// per-frame instance model (`effective = model · base`). This is the primary
    /// constructor; [`auto_fit`](Self::auto_fit) derives the base models for you.
    /// A frame's [`Scene`] references these meshes by id (row index). The mesh
    /// pass renders at [`MSAA_SAMPLE_COUNT`]×; use
    /// [`with_sample_count`](Self::with_sample_count) to override (e.g. `1` = no
    /// MSAA).
    ///
    /// Panics if `meshes` is empty or `meshes`/`base_models` differ in length.
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        meshes: &[Mesh],
        base_models: &[Matrix4],
    ) -> Self {
        Self::with_sample_count(device, format, meshes, base_models, MSAA_SAMPLE_COUNT)
    }

    /// Like [`new`](Self::new), but with an explicit mesh-pass MSAA
    /// `sample_count`: `4` ([`MSAA_SAMPLE_COUNT`]) for anti-aliased edges, or `1`
    /// to render single-sampled (no MSAA — aliased edges, the raw rasterized
    /// coverage). All pipelines and the depth/color attachments are built for this
    /// count.
    ///
    /// Panics if `meshes` is empty, `meshes`/`base_models` differ in length, or
    /// `sample_count` is 0.
    pub fn with_sample_count(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        meshes: &[Mesh],
        base_models: &[Matrix4],
        sample_count: u32,
    ) -> Self {
        assert!(
            !meshes.is_empty(),
            "MeshRenderer requires at least one mesh"
        );
        assert_eq!(
            meshes.len(),
            base_models.len(),
            "meshes and base_models must have equal length"
        );
        assert!(sample_count >= 1, "sample_count must be >= 1");

        let texture = BoundTexture::new(device);
        let env = BoundEnv::new(device);
        let pass = MeshPass::new(device, format, texture.layout(), env.layout(), sample_count);
        let store = MeshStore::new(device, meshes, base_models);
        let frame_plane = FramePlane::new(device, format, sample_count);

        // The picking pipeline: a group-0 camera uniform (structurally identical
        // to the mesh camera layout, so `pass.camera_bind_group` binds it) + the
        // per-instance id color, single-sampled into PICK_FORMAT.
        let pick_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trd picking pipeline layout"),
            bind_group_layouts: &[Some(&create_mesh_bind_group_layout(device))],
            immediate_size: 0,
        });
        let pick_pipeline = create_picking_pipeline(device, &pick_layout);
        let pick_instance_capacity = (meshes.len() as u32).max(1);
        let pick_instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trd pick instance buffer"),
            size: pick_instance_capacity as u64 * std::mem::size_of::<PickInstanceRaw>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pass,
            texture,
            env,
            pbr_material: PbrMaterial::default(),
            store,
            frame_plane,
            depth: None,
            msaa: None,
            sample_count,
            format,
            device: device.clone(),
            pick_pipeline,
            pick_instances,
            pick_instance_capacity,
        }
    }

    /// The number of meshes this renderer can draw; valid mesh ids in a
    /// [`DrawableObject::Mesh`]/[`DrawableObject::AabbBox`] are in
    /// `0..mesh_count()`.
    pub fn mesh_count(&self) -> usize {
        self.store.len()
    }

    /// Binds `texture` as the albedo sampled by [`RenderMode::Textured`] meshes
    /// (#20). The image is (re)uploaded lazily on the next
    /// [`encode`](Self::encode). Until set, the bound texture is 1×1 white.
    pub fn set_texture(&mut self, texture: &dyn Texture) {
        self.texture.set(texture);
    }

    /// Sets the Disney [`PbrMaterial`] applied to every [`RenderMode::Pbr`] draw
    /// (the material is global — one per render invocation). Takes effect on the
    /// next [`encode`](Self::encode).
    pub fn set_pbr_material(&mut self, material: PbrMaterial) {
        self.pbr_material = material;
    }

    /// Binds `env` as the equirectangular HDR environment map reflected by
    /// [`RenderMode::Pbr`] draws. The probe is (re)uploaded lazily on the next
    /// [`encode`](Self::encode). Until set, PBR draws use no environment
    /// reflection (a 1×1 black probe keeps the bind group valid).
    pub fn set_env_map(&mut self, env: EnvMapData) {
        self.env.set(env);
    }

    /// Uploads `rgba` (tightly-packed, row-major `height`×`width`×4) as the
    /// **background frame texture** (#63) sampled by a
    /// [`DrawableObject::FramePlane`]. Delegates to [`FramePlane::upload_rgba`],
    /// which reuses the GPU texture across same-resolution frames.
    ///
    /// Panics if `rgba.len() != width * height * 4` or either dimension is zero.
    pub fn update_frame_texture_rgba(
        &mut self,
        queue: &wgpu::Queue,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) {
        self.frame_plane
            .upload_rgba(&self.device, queue, rgba, width, height);
    }

    /// Whether a background frame texture is currently bound (so a
    /// [`DrawableObject::FramePlane`] would render).
    pub fn has_frame_texture(&self) -> bool {
        self.frame_plane.is_bound()
    }

    /// Encodes one frame's [`Scene`] — an ordered list of [`DrawableObject`]s —
    /// under the shared camera `P·V` uniform. `viewport` gives the target's pixel
    /// dimensions, used to project camera intrinsics (`FrameParams::k`).
    ///
    /// The steps read top-to-bottom: set the camera, walk the scene into
    /// per-geometry instance batches, upload them, size the depth buffer, then
    /// record the pass — the background frame plane first (depth-write off) so
    /// the mesh scene z-composites on top, then each batched draw. Instances are
    /// grouped by geometry so each buffer is drawn once over a contiguous range.
    /// Out-of-range `mesh_id`s are skipped (callers should validate first).
    pub fn encode(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        params: FrameParams,
        scene: &[DrawableObject],
        viewport: Viewport,
    ) {
        // 1. Camera P·V for this frame.
        self.pass.write_camera(queue, params, viewport);
        // 1b. Disney PBR uniform for this frame (camera P·V + world pos, the
        //     global material, and whether an HDR probe is bound). Cheap; written
        //     unconditionally so a PBR draw always has a current uniform.
        self.pass.write_pbr(
            queue,
            params,
            viewport,
            &self.pbr_material,
            self.env.has_env(),
        );

        // 2. Walk the scene once into per-geometry instance batches, then upload
        //    the flattened instance models (growing the buffer if needed).
        let batches = self.store.build_batches(scene);
        self.store
            .upload_instances(&self.device, queue, &batches.instances);

        // 3. Match the depth + (when MSAA is on) color attachments to the viewport
        //    (solid meshes z-occlude; the multisampled color, if any, is resolved
        //    into `view`).
        self.ensure_depth(viewport);
        self.ensure_msaa(viewport);

        // 4. Background frame-plane fit for this viewport (no-op if the scene has
        //    no FramePlane or no frame texture is bound yet).
        if let Some(fit) = batches.frame_fit {
            self.frame_plane.write_fit(queue, fit, viewport);
        }

        // 5. (Re)upload the bound albedo texture on first use / after set_texture
        //    (#20) and the HDR environment map (after set_env_map): encode is where
        //    a GPU queue is available.
        let texture_bind_group = self.texture.ensure_uploaded(&self.device, queue);
        let env_bind_group = self.env.ensure_uploaded(&self.device, queue);

        // 6. Record the pass. With MSAA (`sample_count > 1`) the mesh pass renders
        //    into the multisampled color attachment and resolves into the caller's
        //    single-sample `view`, so every front-end (offscreen CLI, native
        //    window, wasm canvas) gets anti-aliased edges with no API change.
        //    Without MSAA (`sample_count == 1`) there is no MSAA target — the pass
        //    renders straight into `view` (no resolve).
        let depth_view = &self.depth.as_ref().expect("depth set in step 3").view;
        let color_attachment = match self.msaa.as_ref() {
            Some(msaa) => wgpu::RenderPassColorAttachment {
                view: &msaa.view,
                depth_slice: None,
                resolve_target: Some(view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            },
            None => wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            },
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("trd mesh pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // Draw the background frame plane first (#63): its own pipeline + group-0
        // bind, depth-write off, so the mesh scene composites on top. Only when
        // the scene requested one (and a frame texture is bound).
        if batches.frame_fit.is_some() {
            self.frame_plane.draw(&mut pass);
        }

        // The camera bind group (group 0) and the instance buffer (slot 1) stay
        // bound across every mesh draw; each command only swaps pipeline +
        // geometry (and, for textured, the group-1 albedo texture).
        pass.set_bind_group(0, &self.pass.camera_bind_group, &[]);
        pass.set_vertex_buffer(1, self.store.instance_buffer.slice(..));
        for command in &batches.commands {
            let range = command.start..command.start + command.count;
            match command.kind {
                DrawKind::Filled(id) => {
                    let mesh = &self.store.meshes[id];
                    pass.set_pipeline(&self.pass.filled);
                    draw_indexed(&mut pass, &mesh.vertex_buffer, &mesh.triangles, range);
                }
                DrawKind::Textured(id) => {
                    let mesh = &self.store.meshes[id];
                    pass.set_pipeline(&self.pass.textured);
                    pass.set_bind_group(1, texture_bind_group, &[]);
                    draw_indexed(&mut pass, &mesh.vertex_buffer, &mesh.triangles, range);
                }
                DrawKind::Pbr(id) => {
                    let mesh = &self.store.meshes[id];
                    pass.set_pipeline(&self.pass.pbr);
                    // group 0 = PbrUniform (differs from the camera layout),
                    // group 1 = albedo, group 2 = HDR environment map.
                    pass.set_bind_group(0, &self.pass.pbr_bind_group, &[]);
                    pass.set_bind_group(1, texture_bind_group, &[]);
                    pass.set_bind_group(2, env_bind_group, &[]);
                    draw_indexed(&mut pass, &mesh.pbr_vertex_buffer, &mesh.triangles, range);
                    // Restore group 0 = camera for the following non-PBR draws
                    // (their pipelines' group-0 layout is the camera uniform).
                    pass.set_bind_group(0, &self.pass.camera_bind_group, &[]);
                }
                DrawKind::Wireframe(id) => {
                    let mesh = &self.store.meshes[id];
                    pass.set_pipeline(&self.pass.wireframe);
                    draw_indexed(&mut pass, &mesh.vertex_buffer, &mesh.edges, range);
                }
                DrawKind::Aabb(id) => {
                    let mesh = &self.store.meshes[id];
                    pass.set_pipeline(&self.pass.wireframe);
                    draw_indexed(&mut pass, &mesh.aabb.vertex_buffer, &mesh.aabb.index, range);
                }
                DrawKind::Grid(plane) => {
                    pass.set_pipeline(&self.pass.wireframe);
                    pass.set_vertex_buffer(0, self.store.grid_vertex_buffers[plane].slice(..));
                    pass.draw(0..GRID_VERTEX_COUNT, range);
                }
                DrawKind::Shadow => {
                    pass.set_pipeline(&self.pass.shadow);
                    pass.set_vertex_buffer(0, self.store.shadow_vertex_buffer.slice(..));
                    pass.draw(0..SHADOW_VERTEX_COUNT, range);
                }
                DrawKind::Axes => {
                    pass.set_pipeline(&self.pass.wireframe);
                    pass.set_vertex_buffer(0, self.store.axes_vertex_buffer.slice(..));
                    pass.draw(0..AXES_VERTEX_COUNT, range);
                }
            }
        }
    }

    /// Encodes the **object-id picking pass** (#141): renders each `draws` entry's
    /// mesh in a flat color encoding its **index** (the same 0-based order the
    /// caller placed them), single-sampled and depth-tested into `color_view`
    /// (cleared to id `0` = background) with `depth_view`. No lighting, no
    /// texture, no MSAA — so the pixel under the cursor reads back to an exact id
    /// via [`PickInstanceRaw::decode`]. `color_view` must be a [`PICK_FORMAT`]
    /// (linear) target and `depth_view` a [`DEPTH_FORMAT`] attachment of the same
    /// size. Out-of-range mesh ids and `Shadow` draws are skipped, but the index
    /// mapping is preserved (a skipped draw's index simply never appears).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_picking(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        params: FrameParams,
        draws: &[Draw],
        viewport: Viewport,
    ) {
        // Camera P·V for this frame (writes the shared camera uniform bound by
        // `camera_bind_group`, which is layout-compatible with the pick pipeline).
        self.pass.write_camera(queue, params, viewport);

        // Build one pick instance per drawable object, carrying its index color.
        // Keep the draw index as the id even when an entry is skipped, so a decoded
        // id maps straight back to `draws[index]`.
        let mut instances: Vec<PickInstanceRaw> = Vec::with_capacity(draws.len());
        let mut records: Vec<(usize, u32)> = Vec::with_capacity(draws.len());
        for (index, draw) in draws.iter().enumerate() {
            if draw.mode == Some(RenderMode::Shadow) {
                continue;
            }
            let Some(mesh) = self.store.meshes.get(draw.mesh_id as usize) else {
                continue;
            };
            let effective = Matrix4::from_cols_array(&draw.model) * mesh.base_model;
            let slot = instances.len() as u32;
            instances.push(PickInstanceRaw::new(
                effective.to_cols_array(),
                index as u32,
            ));
            records.push((draw.mesh_id as usize, slot));
        }

        // Grow + upload the pick instance buffer.
        if instances.len() as u32 > self.pick_instance_capacity {
            self.pick_instance_capacity = (instances.len() as u32).next_power_of_two();
            self.pick_instances = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("trd pick instance buffer"),
                size: self.pick_instance_capacity as u64
                    * std::mem::size_of::<PickInstanceRaw>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !instances.is_empty() {
            queue.write_buffer(&self.pick_instances, 0, bytemuck::cast_slice(&instances));
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("trd picking pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Clear to id 0 (background).
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pick_pipeline);
        pass.set_bind_group(0, &self.pass.camera_bind_group, &[]);
        pass.set_vertex_buffer(1, self.pick_instances.slice(..));
        for (mesh_id, slot) in records {
            let mesh = &self.store.meshes[mesh_id];
            draw_indexed(
                &mut pass,
                &mesh.vertex_buffer,
                &mesh.triangles,
                slot..slot + 1,
            );
        }
    }

    /// Ensures the depth attachment matches `viewport` (each dimension clamped to
    /// ≥ 1) at the renderer's [`sample_count`](Self::sample_count) (the depth
    /// sample count must match the color attachment), recreating it only when the
    /// target size changes.
    fn ensure_depth(&mut self, viewport: Viewport) {
        let dw = viewport.width.max(1);
        let dh = viewport.height.max(1);
        if self
            .depth
            .as_ref()
            .is_none_or(|d| d.width != dw || d.height != dh)
        {
            self.depth = Some(create_depth_target(&self.device, dw, dh, self.sample_count));
        }
    }

    /// Ensures the multisampled color attachment matches `viewport` (each
    /// dimension clamped to ≥ 1) at the renderer's
    /// [`sample_count`](Self::sample_count) and color `format`, recreating it only
    /// when the target size changes. When MSAA is disabled (`sample_count == 1`)
    /// no MSAA target is needed — the pass renders straight into the caller's
    /// single-sample `view` — so this clears it to `None`.
    fn ensure_msaa(&mut self, viewport: Viewport) {
        if self.sample_count <= 1 {
            self.msaa = None;
            return;
        }
        let dw = viewport.width.max(1);
        let dh = viewport.height.max(1);
        if self
            .msaa
            .as_ref()
            .is_none_or(|m| m.width != dw || m.height != dh)
        {
            self.msaa = Some(create_msaa_color_target(
                &self.device,
                self.format,
                dw,
                dh,
                self.sample_count,
            ));
        }
    }
}
