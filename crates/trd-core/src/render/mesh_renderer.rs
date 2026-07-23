//! The persistent [`MeshRenderer`]: decode-once GPU mesh store, instance
//! batching, and the branch-free [`Scene`](super::Scene) encode.

use super::*;

use crate::math::Matrix4;
use crate::texture::{ImageData, Texture};

/// A mesh uploaded to the GPU: its vertex buffer, the filled **triangle** index
/// buffer, the deduped **edge** (`LineList`) index buffer for wireframe (#38),
/// and the base (preview) model pre-multiplied beneath every per-frame instance
/// model.
struct MeshGpu {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    edge_buffer: wgpu::Buffer,
    edge_count: u32,
    /// AABB overlay (#42): 8 corner vertices (mesh-local coords, [`AABB_COLOR`])
    /// and their 12-edge `LineList` index buffer, drawn beneath the same
    /// per-instance model as the mesh so the box tracks it exactly.
    aabb_vertex_buffer: wgpu::Buffer,
    aabb_edge_buffer: wgpu::Buffer,
    aabb_edge_count: u32,
    base_model: Matrix4,
}

fn upload_mesh(device: &wgpu::Device, mesh: &Mesh, base_model: Matrix4) -> MeshGpu {
    use wgpu::util::DeviceExt;

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("trd mesh vertex buffer"),
        contents: bytemuck::cast_slice(&mesh.vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("trd mesh index buffer"),
        contents: bytemuck::cast_slice(&mesh.indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let index_count = u32::try_from(mesh.indices.len()).expect("mesh index count exceeds u32::MAX");

    let edges = mesh.edge_indices();
    let edge_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("trd mesh edge buffer"),
        contents: bytemuck::cast_slice(&edges),
        usage: wgpu::BufferUsages::INDEX,
    });
    let edge_count = u32::try_from(edges.len()).expect("mesh edge index count exceeds u32::MAX");

    // AABB overlay box: the mesh's own bounding box (mesh-local coords) as 8
    // colored corner vertices + a 12-edge line list. Built once per mesh; drawn
    // only when the renderer's `show_aabb` is set.
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
    let aabb_edge_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("trd mesh aabb edge buffer"),
        contents: bytemuck::cast_slice(&AABB_EDGE_INDICES),
        usage: wgpu::BufferUsages::INDEX,
    });
    let aabb_edge_count = AABB_EDGE_INDICES.len() as u32;

    MeshGpu {
        vertex_buffer,
        index_buffer,
        index_count,
        edge_buffer,
        edge_count,
        aabb_vertex_buffer,
        aabb_edge_buffer,
        aabb_edge_count,
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

/// Which geometry a [`DrawCommand`] binds. The `usize` is a mesh id (index into
/// [`MeshRenderer::meshes`]); `Axes` uses the renderer's shared gizmo geometry.
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

/// Persistent indexed mesh renderer. Owns a filled (`TriangleList`) and a
/// wireframe (`LineList`) pipeline sharing one bind-group layout, a camera
/// (`P·V`) uniform buffer + bind group, a decode-once store of GPU meshes (each
/// with a base/preview model + triangle, edge and AABB-box index buffers), the
/// shared coordinate-axes gizmo geometry, and a growable per-instance
/// model-matrix buffer. Each [`MeshRenderer::encode`] draws a frame's
/// [`Scene`] — an ordered list of [`DrawableObject`]s — grouping instances by
/// geometry so each buffer is drawn once over a contiguous instance range. The
/// renderer holds no mode/overlay state: what to draw is entirely the scene.
///
/// The **background frame texture** (#63) is a second, separately-updated texture
/// binding (`frame_texture`): the mesh albedo above arrives inside the Arrow
/// scene channel and skins the meshes, while this one is uploaded at the boundary
/// from `frame_path`/`frame_url` and skins a [`DrawableObject::FramePlane`]. It is
/// reused across frames (grown only on a resolution change) so per-frame updates
/// never reallocate.
struct FrameTextureGpu {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    /// `vec4` fit uniform (`uv_scale.xy` + padding), rewritten each frame from the
    /// [`FrameFit`] + texture/viewport aspect.
    fit_uniform: wgpu::Buffer,
    width: u32,
    height: u32,
}

pub struct MeshRenderer {
    pipeline: wgpu::RenderPipeline,
    wireframe_pipeline: wgpu::RenderPipeline,
    /// Textured pipeline (#20): draws filled triangles sampling the bound
    /// texture at each vertex UV.
    textured_pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Group-1 layout for the bound texture + sampler (kept to rebuild the bind
    /// group when [`set_texture`](MeshRenderer::set_texture) swaps the image).
    texture_bind_group_layout: wgpu::BindGroupLayout,
    /// The bound texture's group-1 bind group; `None` until `texture_image` is
    /// (re)uploaded on the next `encode` (which supplies the GPU queue).
    texture_bind_group: Option<wgpu::BindGroup>,
    /// The RGBA8 image uploaded as the bound texture (default: 1x1 white, the
    /// identity albedo).
    texture_image: ImageData,
    meshes: Vec<MeshGpu>,
    /// The coordinate-axes gizmo geometry (six `LineList` vertices); each
    /// [`DrawableObject::CoordinateAxes`] draws it under its own model, supplied
    /// through the shared instance buffer.
    axes_vertex_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    instance_capacity: u32,
    /// The mesh pass's depth attachment, (re)created lazily in `encode` to match
    /// the viewport. Gives solid (filled/textured) meshes real z-occlusion.
    depth: Option<DepthTarget>,
    /// Retained so `encode` can grow the instance buffer on demand without the
    /// caller threading a `&Device` through every call (`wgpu::Device` is a
    /// cheap `Arc` handle).
    device: wgpu::Device,
    /// The background frame-plane pipeline (#63): a fullscreen textured quad drawn
    /// first (depth-write off) beneath the mesh scene.
    frame_plane_pipeline: wgpu::RenderPipeline,
    /// Group-0 layout for the background frame texture + sampler + fit uniform.
    frame_bind_group_layout: wgpu::BindGroupLayout,
    /// Linear, clamp-to-edge sampler shared by every background frame texture.
    frame_sampler: wgpu::Sampler,
    /// The reused background frame texture + its bind group; `None` until the
    /// first [`update_frame_texture_rgba`](Self::update_frame_texture_rgba). A
    /// [`DrawableObject::FramePlane`] is skipped while this is `None`.
    frame_texture: Option<FrameTextureGpu>,
}

impl MeshRenderer {
    /// Constructs a `MeshRenderer` that derives each mesh's base (preview) model
    /// automatically via [`Mesh::preview_transform`]
    /// ([`crate::DEFAULT_PREVIEW_TARGET`]) — center + uniform scale-to-fit — so an
    /// arbitrary-unit asset renders centered at a reasonable size. A convenience
    /// constructor over [`new`](Self::new); shared by the headless
    /// [`crate::run_stream`]/`BatchRenderer` and the windowed `trd-app`.
    pub fn auto_fit(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        meshes: &[Mesh],
    ) -> Self {
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
        use wgpu::util::DeviceExt;

        assert!(
            !meshes.is_empty(),
            "MeshRenderer requires at least one mesh"
        );
        assert_eq!(
            meshes.len(),
            base_models.len(),
            "meshes and base_models must have equal length"
        );

        // One explicit bind-group layout shared by both pipelines, so the single
        // params bind group is valid whichever RenderMode is active.
        let bind_group_layout = create_mesh_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trd mesh pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = create_mesh_pipeline_with(
            device,
            format,
            &pipeline_layout,
            wgpu::PrimitiveTopology::TriangleList,
            Some(solid_depth_stencil()),
        );
        let wireframe_pipeline = create_mesh_pipeline_with(
            device,
            format,
            &pipeline_layout,
            wgpu::PrimitiveTopology::LineList,
            Some(overlay_depth_stencil()),
        );
        // Textured pipeline (#20): group 0 = the shared view-proj uniform, group
        // 1 = the bound texture + sampler.
        let texture_bind_group_layout = create_texture_bind_group_layout(device);
        let textured_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("trd textured pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout), Some(&texture_bind_group_layout)],
                immediate_size: 0,
            });
        let textured_pipeline = create_textured_pipeline(device, format, &textured_pipeline_layout);
        // Background frame-plane pipeline (#63): group 0 = the frame texture +
        // sampler + fit uniform, no vertex buffers. Its own bind-group layout,
        // separate from the mesh albedo texture (different update rate).
        let frame_bind_group_layout = create_frame_bind_group_layout(device);
        let frame_plane_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("trd frame plane pipeline layout"),
                bind_group_layouts: &[Some(&frame_bind_group_layout)],
                immediate_size: 0,
            });
        let frame_plane_pipeline =
            create_frame_plane_pipeline(device, format, &frame_plane_pipeline_layout);
        let frame_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("trd frame plane sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        // The identity params ignore the viewport (no intrinsics); each `encode`
        // supplies the real target dimensions.
        let (uniform, bind_group) = create_view_proj_binding(
            device,
            &bind_group_layout,
            FrameParams::IDENTITY,
            Viewport {
                width: 1,
                height: 1,
            },
        );
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
            pipeline,
            wireframe_pipeline,
            textured_pipeline,
            uniform,
            bind_group,
            texture_bind_group_layout,
            texture_bind_group: None,
            texture_image: ImageData {
                width: 1,
                height: 1,
                rgba: vec![255, 255, 255, 255],
            },
            meshes: gpu_meshes,
            axes_vertex_buffer,
            instance_buffer,
            instance_capacity,
            depth: None,
            device: device.clone(),
            frame_plane_pipeline,
            frame_bind_group_layout,
            frame_sampler,
            frame_texture: None,
        }
    }

    /// The number of meshes this renderer can draw; valid mesh ids in a
    /// [`DrawableObject::Mesh`]/[`DrawableObject::AabbBox`] are in
    /// `0..mesh_count()`.
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Binds `texture` as the source sampled by [`RenderMode::Textured`] meshes
    /// (#20). The image is (re)uploaded lazily on the next
    /// [`encode`](Self::encode) (which supplies the GPU queue). Until set, the
    /// bound texture is 1x1 white (the identity albedo).
    pub fn set_texture(&mut self, texture: &dyn Texture) {
        self.texture_image = texture.to_image();
        self.texture_bind_group = None;
    }

    /// Uploads `rgba` (tightly-packed, row-major `height`×`width`×4) as the
    /// **background frame texture** (#63) sampled by a
    /// [`DrawableObject::FramePlane`]. The GPU texture is **reused** across
    /// frames — it is (re)created only when the dimensions change, so streaming a
    /// fixed-resolution video allocates once and every later frame is a plain
    /// `queue.write_texture` into the same texture (no per-frame realloc). The
    /// texture is `Rgba8UnormSrgb` (linearized on sample) and carries **no
    /// mipmaps** (a near-fullscreen background samples ~1:1, and per-frame mip
    /// regeneration would dominate the update cost).
    ///
    /// Panics if `rgba.len() != width * height * 4` or either dimension is zero.
    pub fn update_frame_texture_rgba(
        &mut self,
        queue: &wgpu::Queue,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) {
        assert!(
            width > 0 && height > 0,
            "frame texture dimensions must be non-zero"
        );
        assert_eq!(
            rgba.len(),
            width as usize * height as usize * 4,
            "frame texture rgba length must be width*height*4"
        );

        let needs_new = self
            .frame_texture
            .as_ref()
            .is_none_or(|ft| ft.width != width || ft.height != height);
        if needs_new {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("trd frame texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let fit_uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("trd frame fit uniform"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("trd frame plane bind group"),
                layout: &self.frame_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.frame_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: fit_uniform.as_entire_binding(),
                    },
                ],
            });
            self.frame_texture = Some(FrameTextureGpu {
                texture,
                bind_group,
                fit_uniform,
                width,
                height,
            });
        }

        let ft = self
            .frame_texture
            .as_ref()
            .expect("frame texture set above");
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &ft.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Whether a background frame texture is currently bound (so a
    /// [`DrawableObject::FramePlane`] would render).
    pub fn has_frame_texture(&self) -> bool {
        self.frame_texture.is_some()
    }

    /// Encodes one frame's [`Scene`] — an ordered list of [`DrawableObject`]s —
    /// under the shared camera `P·V` uniform. `viewport` gives the target's pixel
    /// dimensions, used to project camera intrinsics (`FrameParams::k`).
    ///
    /// Instances are grouped by geometry so each buffer is drawn once over a
    /// contiguous instance range: [`DrawableObject::Mesh`] by `(mesh_id, mode)`
    /// (its model pre-multiplied over the mesh base model, `effective = model ·
    /// base`), [`DrawableObject::AabbBox`] by `mesh_id` (same `model · base` as
    /// the mesh it boxes), and [`DrawableObject::CoordinateAxes`] under its own
    /// model. Gizmo overlays (AABB boxes, axes) and wireframes are composited
    /// after all solid geometry and drawn depth-`Always`/no-write, so they stay
    /// visible on top even though solid meshes now z-occlude via a depth buffer.
    ///
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
        write_view_proj(queue, &self.uniform, params, viewport);

        // (Re)upload the bound texture on first use / after `set_texture` (#20):
        // `encode` is where a GPU queue is available.
        if self.texture_bind_group.is_none() {
            self.texture_bind_group = Some(upload_texture(
                &self.device,
                queue,
                &self.texture_bind_group_layout,
                &self.texture_image,
            ));
        }

        // Walk the scene once, bucketing each drawable's instance model by the
        // geometry it draws so same-geometry instances share one draw call.
        let mesh_count = self.meshes.len();
        let mut filled: Vec<Vec<InstanceRaw>> = vec![Vec::new(); mesh_count];
        let mut textured: Vec<Vec<InstanceRaw>> = vec![Vec::new(); mesh_count];
        let mut wireframe: Vec<Vec<InstanceRaw>> = vec![Vec::new(); mesh_count];
        let mut aabb: Vec<Vec<InstanceRaw>> = vec![Vec::new(); mesh_count];
        let mut axes: Vec<InstanceRaw> = Vec::new();
        // The background frame plane is a singleton overlay (there is one bound
        // frame texture); the last FramePlane in the scene wins its fit.
        let mut frame_plane: Option<FrameFit> = None;

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
                    frame_plane = Some(fit);
                }
            }
        }

        // Flatten every instance model into one buffer, recording a draw command
        // per non-empty group. Order = filled meshes, wireframe meshes, then the
        // gizmo overlays (AABB boxes, then axes) on top.
        let mut instances: Vec<InstanceRaw> = Vec::with_capacity(scene.len());
        let mut commands: Vec<DrawCommand> = Vec::new();
        for (mesh_id, bucket) in filled.iter().enumerate() {
            push_command(
                &mut instances,
                &mut commands,
                DrawKind::Filled(mesh_id),
                bucket,
            );
        }
        for (mesh_id, bucket) in textured.iter().enumerate() {
            push_command(
                &mut instances,
                &mut commands,
                DrawKind::Textured(mesh_id),
                bucket,
            );
        }
        for (mesh_id, bucket) in wireframe.iter().enumerate() {
            push_command(
                &mut instances,
                &mut commands,
                DrawKind::Wireframe(mesh_id),
                bucket,
            );
        }
        for (mesh_id, bucket) in aabb.iter().enumerate() {
            push_command(
                &mut instances,
                &mut commands,
                DrawKind::Aabb(mesh_id),
                bucket,
            );
        }
        push_command(&mut instances, &mut commands, DrawKind::Axes, &axes);

        if instances.len() as u32 > self.instance_capacity {
            self.instance_capacity = (instances.len() as u32).next_power_of_two();
            self.instance_buffer = create_instance_buffer(&self.device, self.instance_capacity);
        }
        if !instances.is_empty() {
            queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));
        }

        // Ensure the depth attachment matches the viewport (solid meshes need it
        // for z-occlusion; recreated only when the target size changes).
        let dw = viewport.width.max(1);
        let dh = viewport.height.max(1);
        if self
            .depth
            .as_ref()
            .is_none_or(|d| d.width != dw || d.height != dh)
        {
            self.depth = Some(create_depth_target(&self.device, dw, dh));
        }
        let depth_view = &self.depth.as_ref().unwrap().view;

        // Background frame plane (#63): compute + upload its centered fit scale for
        // the current viewport before the pass, so the fullscreen quad samples the
        // reused frame texture with the right crop/fill. Skipped when no FramePlane
        // is in the scene or no frame texture has been uploaded yet.
        if let (Some(fit), Some(ft)) = (frame_plane, self.frame_texture.as_ref()) {
            let scale =
                frame_fit_uv_scale(fit, ft.width, ft.height, viewport.width, viewport.height);
            let fit_data: [f32; 4] = [scale[0], scale[1], 0.0, 0.0];
            queue.write_buffer(&ft.fit_uniform, 0, bytemuck::cast_slice(&fit_data));
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("trd mesh pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
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
        // bind (texture/sampler/fit), depth-write off, so it fills color under the
        // cleared depth and the mesh scene z-composites on top. Only when a frame
        // texture is bound.
        if frame_plane.is_some() {
            if let Some(ft) = self.frame_texture.as_ref() {
                pass.set_pipeline(&self.frame_plane_pipeline);
                pass.set_bind_group(0, &ft.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        for command in &commands {
            let range = command.start..command.start + command.count;
            match command.kind {
                DrawKind::Filled(mesh_id) => {
                    let mesh = &self.meshes[mesh_id];
                    pass.set_pipeline(&self.pipeline);
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, range);
                }
                DrawKind::Textured(mesh_id) => {
                    let mesh = &self.meshes[mesh_id];
                    pass.set_pipeline(&self.textured_pipeline);
                    // group 0 (view-proj) stays bound from before the loop; bind
                    // the texture as group 1 (uploaded above, always Some here).
                    if let Some(texture) = self.texture_bind_group.as_ref() {
                        pass.set_bind_group(1, texture, &[]);
                    }
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.index_count, 0, range);
                }
                DrawKind::Wireframe(mesh_id) => {
                    let mesh = &self.meshes[mesh_id];
                    pass.set_pipeline(&self.wireframe_pipeline);
                    pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(mesh.edge_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.edge_count, 0, range);
                }
                DrawKind::Aabb(mesh_id) => {
                    let mesh = &self.meshes[mesh_id];
                    pass.set_pipeline(&self.wireframe_pipeline);
                    pass.set_vertex_buffer(0, mesh.aabb_vertex_buffer.slice(..));
                    pass.set_index_buffer(
                        mesh.aabb_edge_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    pass.draw_indexed(0..mesh.aabb_edge_count, 0, range);
                }
                DrawKind::Axes => {
                    pass.set_pipeline(&self.wireframe_pipeline);
                    pass.set_vertex_buffer(0, self.axes_vertex_buffer.slice(..));
                    pass.draw(0..AXES_VERTEX_COUNT, range);
                }
            }
        }
    }
}
