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
/// [`MeshStore::meshes`]); `Axes` uses the shared gizmo geometry.
enum DrawKind {
    /// Filled triangles of a mesh (its triangle index buffer + filled pipeline).
    Filled(usize),
    /// Textured triangles of a mesh (triangle index buffer + textured pipeline,
    /// sampling the bound texture at each vertex UV) (#20).
    Textured(usize),
    /// Edge lines of a mesh (its deduped edge index buffer + line pipeline).
    Wireframe(usize),
    /// A mesh's AABB box (its precomputed corner geometry + line pipeline).
    Aabb(usize),
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
    camera_uniform: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
}

impl MeshPass {
    /// Constructs a `MeshPass` for `format`, building all three pipelines over a
    /// shared camera bind-group layout. `texture_layout` is the albedo texture's
    /// group-1 layout (from [`BoundTexture::layout`]), needed by the textured
    /// pipeline's layout.
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        texture_layout: &wgpu::BindGroupLayout,
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
            MSAA_SAMPLE_COUNT,
        );
        let wireframe = create_mesh_pipeline_with(
            device,
            format,
            &pipeline_layout,
            wgpu::PrimitiveTopology::LineList,
            Some(overlay_depth_stencil()),
            MSAA_SAMPLE_COUNT,
        );
        // Textured pipeline (#20): group 0 = the shared camera uniform, group 1 =
        // the bound albedo texture + sampler.
        let textured_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("trd textured pipeline layout"),
                bind_group_layouts: &[Some(&camera_layout), Some(texture_layout)],
                immediate_size: 0,
            });
        let textured =
            create_textured_pipeline(device, format, &textured_pipeline_layout, MSAA_SAMPLE_COUNT);
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
            camera_uniform,
            camera_bind_group,
        }
    }

    /// Rewrites the camera `P·V` uniform for this frame's `params`/`viewport`.
    fn write_camera(&self, queue: &wgpu::Queue, params: FrameParams, viewport: Viewport) {
        write_view_proj(queue, &self.camera_uniform, params, viewport);
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

        Self {
            meshes: gpu_meshes,
            axes_vertex_buffer,
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
    /// [`DrawCommand`]s. Draw order: filled, textured, wireframe, AABB boxes,
    /// then axes — so opaque meshes precede the line overlays that composite on
    /// top. Out-of-range mesh ids are skipped.
    fn build_batches(&self, scene: &[DrawableObject]) -> Batches {
        let mesh_count = self.meshes.len();
        let mut filled: Vec<Vec<InstanceRaw>> = vec![Vec::new(); mesh_count];
        let mut textured: Vec<Vec<InstanceRaw>> = vec![Vec::new(); mesh_count];
        let mut wireframe: Vec<Vec<InstanceRaw>> = vec![Vec::new(); mesh_count];
        let mut aabb: Vec<Vec<InstanceRaw>> = vec![Vec::new(); mesh_count];
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
                        RenderMode::Wireframe => wireframe[mesh_id as usize].push(instance),
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
                DrawableObject::FramePlane { fit } => {
                    frame_fit = Some(fit);
                }
            }
        }

        // Flatten every instance model into one buffer, recording a draw command
        // per non-empty group in the layered draw order.
        let mut instances: Vec<InstanceRaw> = Vec::with_capacity(scene.len());
        let mut commands: Vec<DrawCommand> = Vec::new();
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
    store: MeshStore,
    frame_plane: FramePlane,
    /// The mesh pass's depth attachment, (re)created lazily in `encode` to match
    /// the viewport. Gives solid (filled/textured) meshes real z-occlusion.
    depth: Option<DepthTarget>,
    /// The mesh pass's multisampled color attachment ([`MSAA_SAMPLE_COUNT`]×),
    /// (re)created lazily in `encode` to match the viewport. The pass renders into
    /// it and resolves into the caller's single-sample `view`, so every front-end
    /// gets anti-aliased edges transparently.
    msaa: Option<MsaaColorTarget>,
    /// The color format the pipelines were built for; the MSAA color target must
    /// be created with the same format.
    format: wgpu::TextureFormat,
    /// Retained so `encode` can grow GPU resources on demand without the caller
    /// threading a `&Device` through every call (`wgpu::Device` is a cheap `Arc`).
    device: wgpu::Device,
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
    /// A frame's [`Scene`] references these meshes by id (row index).
    ///
    /// Panics if `meshes` is empty or `meshes`/`base_models` differ in length.
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        meshes: &[Mesh],
        base_models: &[Matrix4],
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

        let texture = BoundTexture::new(device);
        let pass = MeshPass::new(device, format, texture.layout());
        let store = MeshStore::new(device, meshes, base_models);
        let frame_plane = FramePlane::new(device, format);

        Self {
            pass,
            texture,
            store,
            frame_plane,
            depth: None,
            msaa: None,
            format,
            device: device.clone(),
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

        // 2. Walk the scene once into per-geometry instance batches, then upload
        //    the flattened instance models (growing the buffer if needed).
        let batches = self.store.build_batches(scene);
        self.store
            .upload_instances(&self.device, queue, &batches.instances);

        // 3. Match the depth + MSAA color attachments to the viewport (solid
        //    meshes z-occlude; the multisampled color is resolved into `view`).
        self.ensure_depth(viewport);
        self.ensure_msaa(viewport);

        // 4. Background frame-plane fit for this viewport (no-op if the scene has
        //    no FramePlane or no frame texture is bound yet).
        if let Some(fit) = batches.frame_fit {
            self.frame_plane.write_fit(queue, fit, viewport);
        }

        // 5. (Re)upload the bound albedo texture on first use / after set_texture
        //    (#20): encode is where a GPU queue is available.
        let texture_bind_group = self.texture.ensure_uploaded(&self.device, queue);

        // 6. Record the pass. The mesh pass renders into the multisampled color
        //    attachment and resolves into the caller's single-sample `view`, so
        //    every front-end (offscreen CLI, native window, wasm canvas) gets
        //    anti-aliased edges with no API change.
        let depth_view = &self.depth.as_ref().expect("depth set in step 3").view;
        let msaa_view = &self.msaa.as_ref().expect("msaa set in step 3").view;
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("trd mesh pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: msaa_view,
                depth_slice: None,
                resolve_target: Some(view),
                ops: wgpu::Operations {
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
                DrawKind::Axes => {
                    pass.set_pipeline(&self.pass.wireframe);
                    pass.set_vertex_buffer(0, self.store.axes_vertex_buffer.slice(..));
                    pass.draw(0..AXES_VERTEX_COUNT, range);
                }
            }
        }
    }

    /// Ensures the depth attachment matches `viewport` (each dimension clamped to
    /// ≥ 1) at [`MSAA_SAMPLE_COUNT`] (the depth sample count must match the color
    /// attachment), recreating it only when the target size changes.
    fn ensure_depth(&mut self, viewport: Viewport) {
        let dw = viewport.width.max(1);
        let dh = viewport.height.max(1);
        if self
            .depth
            .as_ref()
            .is_none_or(|d| d.width != dw || d.height != dh)
        {
            self.depth = Some(create_depth_target(&self.device, dw, dh, MSAA_SAMPLE_COUNT));
        }
    }

    /// Ensures the multisampled color attachment matches `viewport` (each
    /// dimension clamped to ≥ 1) at [`MSAA_SAMPLE_COUNT`] and the renderer's
    /// color `format`, recreating it only when the target size changes.
    fn ensure_msaa(&mut self, viewport: Viewport) {
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
                MSAA_SAMPLE_COUNT,
            ));
        }
    }
}
