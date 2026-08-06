//! The persistent [`MeshRenderer`]: a decode-once GPU mesh store, instance
//! batching, and [`Scene`](super::Scene) encoding.
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

use super::bound_material_maps::BoundMaterialMaps;
use super::bound_texture::BoundTexture;
use super::env_background::{EnvBackground, EnvBackgroundSettings};
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

/// A self-contained non-indexed draw.
struct VertexGeometry {
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
}

impl VertexGeometry {
    fn new<T: bytemuck::Pod>(device: &wgpu::Device, label: &str, vertices: &[T]) -> Self {
        use wgpu::util::DeviceExt;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let vertex_count = u32::try_from(vertices.len()).expect("vertex count exceeds u32::MAX");
        Self {
            vertex_buffer,
            vertex_count,
        }
    }
}

/// A mesh uploaded to the GPU. Its `vertex_buffer` feeds both the filled
/// `triangles` and the deduped wireframe `edges` (#38); the `aabb` overlay (#42)
/// is a standalone box of 12 screen-space-expanded edge quads. `base_model` is
/// the base (preview) transform pre-multiplied beneath every per-frame instance
/// model (`effective = model · base`).
struct MeshGpu {
    vertex_buffer: wgpu::Buffer,
    /// Parallel vertex buffer for the Disney PBR path (`disney.wgsl`): the same
    /// positions + UVs as `vertex_buffer`, but with a derived smooth shading
    /// **normal** in place of the vertex color. Reuses the `triangles` index
    /// buffer. Built once per mesh; only bound by [`RenderMode::Pbr`] draws.
    pbr_vertex_buffer: wgpu::Buffer,
    triangles: IndexBuf,
    edges: IndexBuf,
    aabb: VertexGeometry,
    base_model: Matrix4,
    /// This mesh's **own** albedo texture (group 1), so a multi-object scene skins
    /// each object with its own diffuse (#141). Defaults to 1×1 white (identity
    /// albedo) until [`set`](BoundTexture::set) via `set_mesh_texture`.
    texture: BoundTexture,
    material_maps: BoundMaterialMaps,
}

impl MeshGpu {
    fn filled(&self) -> (&wgpu::Buffer, &IndexBuf) {
        (&self.vertex_buffer, &self.triangles)
    }

    fn pbr(&self) -> (&wgpu::Buffer, &IndexBuf) {
        (&self.pbr_vertex_buffer, &self.triangles)
    }

    fn wireframe(&self) -> (&wgpu::Buffer, &IndexBuf) {
        (&self.vertex_buffer, &self.edges)
    }

    fn aabb(&self) -> &VertexGeometry {
        &self.aabb
    }
}

fn upload_mesh(
    device: &wgpu::Device,
    mesh: &Mesh,
    base_model: Matrix4,
    texture_layout: &wgpu::BindGroupLayout,
    material_maps_layout: &wgpu::BindGroupLayout,
) -> MeshGpu {
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
    let normals = mesh
        .shading
        .as_ref()
        .filter(|shading| shading.normals.len() == mesh.vertices.len())
        .map(|shading| shading.normals.clone())
        .unwrap_or_else(|| compute_smooth_normals(&mesh.vertices, &mesh.indices));
    let tangents = mesh
        .shading
        .as_ref()
        .filter(|shading| shading.tangents.len() == mesh.vertices.len())
        .map(|shading| shading.tangents.clone())
        .unwrap_or_else(|| compute_tangents(&mesh.vertices, &mesh.indices, &normals));
    let pbr_vertices: Vec<PbrVertex> = mesh
        .vertices
        .iter()
        .zip(&normals)
        .zip(&tangents)
        .map(|((v, &normal), &tangent)| PbrVertex {
            position: v.position,
            normal,
            uv: v.uv,
            tangent,
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
    let aabb_corners = mesh.aabb().corners().map(|corner| corner.to_array());
    let aabb_vertices = aabb_line_vertices(&aabb_corners);

    MeshGpu {
        vertex_buffer,
        pbr_vertex_buffer,
        triangles,
        edges,
        aabb: VertexGeometry::new(device, "trd mesh aabb line buffer", &aabb_vertices),
        base_model,
        texture: BoundTexture::with_layout(texture_layout.clone()),
        material_maps: BoundMaterialMaps::with_layout(material_maps_layout.clone()),
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
    (vertex_buffer, index): (&wgpu::Buffer, &IndexBuf),
    instances: Range<u32>,
) {
    pass.set_vertex_buffer(0, vertex_buffer.slice(..));
    pass.set_index_buffer(index.buffer.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..index.count, 0, instances);
}

fn draw_vertices(pass: &mut wgpu::RenderPass, geometry: &VertexGeometry, instances: Range<u32>) {
    pass.set_vertex_buffer(0, geometry.vertex_buffer.slice(..));
    pass.draw(0..geometry.vertex_count, instances);
}

/// Which geometry a [`DrawCommand`] binds. The `usize` is a mesh id (index into
/// [`MeshStore::meshes`]) for the mesh kinds, or a [`GridPlane::index`] for
/// `Grid`; `Axes` uses the shared gizmo geometry. Variants are declared in their
/// layered draw order; the derived ordering is the batching order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum DrawKind {
    /// A contact / blob **grounding shadow** (the shared shadow quad geometry,
    /// non-indexed triangle draw, alpha-blended over the frame plane).
    Shadow,
    /// Filled triangles of a mesh (its triangle index buffer + filled pipeline).
    Filled(usize),
    /// Textured triangles of a mesh (triangle index buffer + textured pipeline,
    /// sampling the bound texture at each vertex UV) (#20).
    Textured(usize),
    /// Disney **PBR** triangles of a mesh (its dedicated position+normal+UV
    /// vertex buffer + `disney.wgsl` pipeline, lit by the virtual light rig and
    /// the bound HDR environment map). Reuses the triangle index buffer.
    Pbr(usize),
    /// A coordinate-plane grid (the shared per-plane grid vertex buffer indexed
    /// by [`GridPlane::index`], non-indexed line draw).
    Grid(usize),
    /// Edge lines of a mesh (its deduped edge index buffer + line pipeline).
    Wireframe(usize),
    /// A mesh's AABB box (its precomputed corner geometry + line pipeline).
    Aabb(usize),
    /// The coordinate-axes gizmo (shared vertex buffer, non-indexed line draw).
    Axes,
}

/// One instanced draw recorded while walking a [`Scene`]: the geometry to bind
/// ([`DrawKind`]) and the contiguous instance-buffer range to draw it over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DrawCommand {
    kind: DrawKind,
    start: u32,
    count: u32,
}

/// The result of walking a [`Scene`] once: the flattened per-instance models,
/// the [`DrawCommand`]s over them (already in draw order), and the singleton
/// background frame-plane fit (if any).
struct Batches {
    instances: Vec<InstanceRaw>,
    commands: Vec<DrawCommand>,
    frame_fit: Option<FrameFit>,
    environment_background: Option<([f32; 3], Tonemap)>,
}

/// Walks `scene` once into a flat draw list, stable-sorts by [`DrawKind`], then
/// groups equal runs into instanced commands. Out-of-range mesh ids are skipped;
/// the last background frame plane wins.
fn build_batches(
    scene: &[DrawableObject],
    mut mesh_base_model: impl FnMut(usize) -> Option<Matrix4>,
) -> Batches {
    let mut draws: Vec<(DrawKind, InstanceRaw)> = Vec::with_capacity(scene.len());
    let mut frame_fit = None;
    let mut environment_background = None;

    for object in scene {
        let (kind, model) = match *object {
            DrawableObject::Mesh {
                mesh_id,
                model,
                mode,
            } => {
                let mesh_id = mesh_id as usize;
                let Some(base_model) = mesh_base_model(mesh_id) else {
                    continue;
                };
                let kind = match mode {
                    RenderMode::Filled => DrawKind::Filled(mesh_id),
                    RenderMode::Textured => DrawKind::Textured(mesh_id),
                    RenderMode::Pbr => DrawKind::Pbr(mesh_id),
                    RenderMode::Wireframe => DrawKind::Wireframe(mesh_id),
                    RenderMode::Shadow => continue,
                };
                let effective = Matrix4::from_cols_array(&model) * base_model;
                (kind, effective.to_cols_array())
            }
            DrawableObject::AabbBox { mesh_id, model } => {
                let mesh_id = mesh_id as usize;
                let Some(base_model) = mesh_base_model(mesh_id) else {
                    continue;
                };
                let effective = Matrix4::from_cols_array(&model) * base_model;
                (DrawKind::Aabb(mesh_id), effective.to_cols_array())
            }
            DrawableObject::CoordinateAxes { model } => (DrawKind::Axes, model),
            DrawableObject::PlaneGrid { plane, model } => (DrawKind::Grid(plane.index()), model),
            DrawableObject::BlobShadow { model } => (DrawKind::Shadow, model),
            DrawableObject::EnvironmentBackground {
                rotation,
                exposure,
                blur,
                tonemap,
            } => {
                environment_background = Some(([rotation, exposure, blur], tonemap));
                continue;
            }
            DrawableObject::FramePlane { fit } => {
                frame_fit = Some(fit);
                continue;
            }
        };
        draws.push((kind, InstanceRaw { model }));
    }

    draws.sort_by_key(|(kind, _)| *kind);

    let mut instances = Vec::with_capacity(draws.len());
    let mut commands = Vec::new();
    for run in draws.chunk_by(|a, b| a.0 == b.0) {
        let start = instances.len() as u32;
        instances.extend(run.iter().map(|(_, instance)| *instance));
        commands.push(DrawCommand {
            kind: run[0].0,
            start,
            count: run.len() as u32,
        });
    }

    Batches {
        instances,
        commands,
        frame_fit,
        environment_background,
    }
}

/// The mesh and gizmo pipelines plus their camera/material bindings. Filled,
/// wireframe, arrowheads, and textured rendering share the camera layout;
/// expanded gizmo lines use a viewport-aware group-0 uniform.
struct MeshPass {
    filled: wgpu::RenderPipeline,
    wireframe: wgpu::RenderPipeline,
    /// Screen-space expanded, alpha-feathered AABB/axes/grid line pipeline.
    gizmo_line: wgpu::RenderPipeline,
    /// Unlit overlay triangles for coordinate-axis arrowheads.
    gizmo_solid: wgpu::RenderPipeline,
    textured: wgpu::RenderPipeline,
    /// The contact / blob grounding-shadow pipeline (alpha-blended, depth-write
    /// off); shares the untextured camera bind-group layout (group 0).
    shadow: wgpu::RenderPipeline,
    /// The Disney PBR pipeline (`disney.wgsl`): group 0 = [`pbr_uniform`], group 1
    /// = the bound albedo texture, group 2 = the HDR environment map.
    pbr: wgpu::RenderPipeline,
    /// The per-object `PbrUniform` buffer: `mesh_count` [`pbr_stride`]-spaced
    /// slots (each carries the shared camera/lights + that mesh's material),
    /// rewritten each `encode`; a draw binds its slot via a dynamic offset.
    pbr_uniform: wgpu::Buffer,
    pbr_bind_group: wgpu::BindGroup,
    /// The 256-aligned byte stride between adjacent `PbrUniform` slots (the
    /// `min_uniform_buffer_offset_alignment`-rounded `size_of::<PbrUniform>()`),
    /// so `slot i` lives at `i * pbr_stride`.
    pbr_stride: u64,
    camera_uniform: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    gizmo_uniform: wgpu::Buffer,
    gizmo_bind_group: wgpu::BindGroup,
}

struct PbrInputs<'a> {
    materials: &'a [DisneyMaterial],
    ibl: &'a [ImageBasedLighting],
    tone_mappings: &'a [ToneMapping],
    debug_views: &'a [PbrDebugView],
    lighting: Lighting,
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
        material_maps_layout: &wgpu::BindGroupLayout,
        env_layout: &wgpu::BindGroupLayout,
        sample_count: u32,
        mesh_count: usize,
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
        let gizmo_solid = create_mesh_pipeline_with(
            device,
            format,
            &pipeline_layout,
            wgpu::PrimitiveTopology::TriangleList,
            Some(overlay_depth_stencil()),
            sample_count,
        );
        let gizmo_layout = create_gizmo_bind_group_layout(device);
        let gizmo_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("trd gizmo pipeline layout"),
                bind_group_layouts: &[Some(&gizmo_layout)],
                immediate_size: 0,
            });
        let gizmo_line =
            create_gizmo_line_pipeline(device, format, &gizmo_pipeline_layout, sample_count);
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
            bind_group_layouts: &[
                Some(&pbr_layout),
                Some(texture_layout),
                Some(env_layout),
                Some(material_maps_layout),
            ],
            immediate_size: 0,
        });
        let pbr = create_pbr_pipeline(device, format, &pbr_pipeline_layout, sample_count);
        // The per-object PbrUniform buffer: one 256-aligned slot per mesh, each
        // rewritten every frame with the shared camera/lights + that mesh's
        // material; a PBR draw selects its slot via a dynamic offset.
        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let pbr_stride = (std::mem::size_of::<PbrUniform>() as u64).next_multiple_of(align);
        let pbr_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trd pbr uniform"),
            size: pbr_stride * mesh_count.max(1) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pbr_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trd pbr bind group"),
            layout: &pbr_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                // A single-slot window; the dynamic offset picks which slot.
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &pbr_uniform,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<PbrUniform>() as u64),
                }),
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
        let (gizmo_uniform, gizmo_bind_group) = create_gizmo_binding(
            device,
            &gizmo_layout,
            FrameParams::IDENTITY,
            Viewport {
                width: 1,
                height: 1,
            },
        );
        Self {
            filled,
            wireframe,
            gizmo_line,
            gizmo_solid,
            textured,
            shadow,
            pbr,
            pbr_uniform,
            pbr_bind_group,
            pbr_stride,
            camera_uniform,
            camera_bind_group,
            gizmo_uniform,
            gizmo_bind_group,
        }
    }

    /// Rewrites the camera `P·V` uniform for this frame's `params`/`viewport`.
    fn write_camera(&self, queue: &wgpu::Queue, params: FrameParams, viewport: Viewport) {
        write_view_proj(queue, &self.camera_uniform, params, viewport);
        write_gizmo_params(queue, &self.gizmo_uniform, params, viewport);
    }

    /// Rewrites the Disney PBR uniform **slots** for this frame: one slot per
    /// mesh (`materials[i]` → slot `i` at `i * pbr_stride`), each carrying the
    /// shared camera `P·V` + world position + light rig, this mesh's material, and
    /// the env gate. A PBR draw then binds its object's material via a dynamic
    /// offset. `materials` is indexed by mesh id.
    fn write_pbr(
        &self,
        queue: &wgpu::Queue,
        params: FrameParams,
        viewport: Viewport,
        inputs: PbrInputs<'_>,
        use_env: bool,
    ) {
        debug_assert_eq!(inputs.materials.len(), inputs.ibl.len());
        debug_assert_eq!(inputs.materials.len(), inputs.tone_mappings.len());
        debug_assert_eq!(inputs.materials.len(), inputs.debug_views.len());
        let view_proj = params.view_proj_matrix(viewport).to_cols_array();
        let camera_pos = params.camera_position();
        for (i, (((material, ibl), tone_mapping), debug_view)) in inputs
            .materials
            .iter()
            .zip(inputs.ibl)
            .zip(inputs.tone_mappings)
            .zip(inputs.debug_views)
            .enumerate()
        {
            let uniform = PbrUniform::new(
                view_proj,
                camera_pos,
                PbrUniformInputs {
                    material,
                    lighting: inputs.lighting,
                    ibl: *ibl,
                    tone_mapping: *tone_mapping,
                    debug_view: *debug_view,
                    use_env,
                },
            );
            queue.write_buffer(
                &self.pbr_uniform,
                i as u64 * self.pbr_stride,
                bytemuck::bytes_of(&uniform),
            );
        }
    }
}

/// The decode-once geometry store: the uploaded [`MeshGpu`]s (referenced by a
/// scene's mesh ids), the shared coordinate-axes gizmo vertices, and the
/// growable per-instance model-matrix buffer.
struct MeshStore {
    meshes: Vec<MeshGpu>,
    /// The coordinate-axis shafts and cone arrowheads.
    axes_lines: VertexGeometry,
    axes_heads: VertexGeometry,
    /// The coordinate-plane grid geometry, one expanded-line buffer per
    /// [`GridPlane`] (indexed by [`GridPlane::index`]): XY, XZ, YZ. Each
    /// [`DrawableObject::PlaneGrid`] draws the buffer for its plane under its own
    /// model, supplied through the shared instance buffer.
    grid_lines: [VertexGeometry; 3],
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
    fn new(
        device: &wgpu::Device,
        meshes: &[Mesh],
        base_models: &[Matrix4],
        texture_layout: &wgpu::BindGroupLayout,
        material_maps_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        use wgpu::util::DeviceExt;

        let gpu_meshes = meshes
            .iter()
            .zip(base_models)
            .map(|(mesh, &base)| {
                upload_mesh(device, mesh, base, texture_layout, material_maps_layout)
            })
            .collect();
        let instance_capacity = (meshes.len() as u32).max(1);
        let instance_buffer = create_instance_buffer(device, instance_capacity);

        let axes_lines = VertexGeometry::new(device, "trd axes line buffer", &axes_line_vertices());
        let axes_heads =
            VertexGeometry::new(device, "trd axes arrow buffer", &axes_arrow_vertices());

        // Coordinate-plane grids: one expanded-line vertex buffer per plane
        // (XY/XZ/YZ), drawn under each PlaneGrid object's model.
        let grid_lines = [
            VertexGeometry::new(
                device,
                "trd xy grid line buffer",
                &grid_line_vertices(GridPlane::Xy),
            ),
            VertexGeometry::new(
                device,
                "trd xz grid line buffer",
                &grid_line_vertices(GridPlane::Xz),
            ),
            VertexGeometry::new(
                device,
                "trd yz grid line buffer",
                &grid_line_vertices(GridPlane::Yz),
            ),
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
            axes_lines,
            axes_heads,
            grid_lines,
            shadow_vertex_buffer,
            instance_buffer,
            instance_capacity,
        }
    }

    fn len(&self) -> usize {
        self.meshes.len()
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
    /// The bound HDR environment map reflected by [`RenderMode::Pbr`] draws.
    env: BoundEnv,
    env_background: EnvBackground,
    /// The Disney material of **each** mesh (indexed by mesh id) applied to its
    /// [`RenderMode::Pbr`] draws (#141) — so a multi-object scene can give every
    /// object its own metallic/roughness/base_color.
    pbr_materials: Vec<DisneyMaterial>,
    /// Per-object environment reflection gains, parallel to `pbr_materials`.
    pbr_ibl: Vec<ImageBasedLighting>,
    /// Per-object output transforms, parallel to `pbr_materials`.
    pbr_tone_mappings: Vec<ToneMapping>,
    pbr_debug_views: Vec<PbrDebugView>,
    /// Scene light rig controls shared by every PBR object.
    lighting: Lighting,
    store: MeshStore,
    frame_plane: FramePlane,
    /// The mesh pass's depth attachment, (re)created lazily in `encode` to match
    /// the viewport. Gives solid (filled/textured) meshes real z-occlusion.
    depth: Option<DepthTarget>,
    /// The mesh pass's multisampled color attachment ([`sample_count`](Self::sample_count)×),
    /// (re)created lazily in `encode` to match the viewport. The pass renders into
    /// it and resolves into the caller's single-sample `view`, so every front-end
    /// gets multisampled mesh/arrowhead edges transparently. Gizmo lines add
    /// analytic AA separately. `None` when MSAA is disabled (`sample_count == 1`):
    /// the pass then renders straight into `view`.
    msaa: Option<MsaaColorTarget>,
    /// The mesh pass's MSAA sample count — `4` (the default,
    /// [`MSAA_SAMPLE_COUNT`]) for multisampled edges, or `1` for single-sample
    /// rasterization. Fixed at construction because every pipeline plus the
    /// depth/color attachments must share it.
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
    /// `sample_count`: `4` ([`MSAA_SAMPLE_COUNT`]) for multisampled edges, or `1`
    /// to render single-sampled. Gizmo lines retain their shader-based analytic AA
    /// at `1`; mesh silhouettes and hardware wireframes do not. All pipelines and
    /// the depth/color attachments are built for this count.
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

        // One shared group-1 albedo layout for the textured/PBR pipelines and
        // every per-mesh [`BoundTexture`] (each object skins with its own diffuse).
        let texture_layout = create_texture_bind_group_layout(device);
        let material_maps_layout = BoundMaterialMaps::create_layout(device);
        let env = BoundEnv::new(device);
        let env_background = EnvBackground::new(device, format, env.layout(), sample_count);
        let pass = MeshPass::new(
            device,
            format,
            &texture_layout,
            &material_maps_layout,
            env.layout(),
            sample_count,
            meshes.len(),
        );
        let store = MeshStore::new(
            device,
            meshes,
            base_models,
            &texture_layout,
            &material_maps_layout,
        );
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
            env,
            env_background,
            pbr_materials: vec![DisneyMaterial::default(); meshes.len()],
            pbr_ibl: vec![ImageBasedLighting::default(); meshes.len()],
            pbr_tone_mappings: vec![ToneMapping::default(); meshes.len()],
            pbr_debug_views: vec![PbrDebugView::default(); meshes.len()],
            lighting: Lighting::default(),
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

    /// Binds `texture` as the albedo of **mesh 0** — the single-mesh /
    /// wire-protocol default sampled by [`RenderMode::Textured`]/[`RenderMode::Pbr`]
    /// draws (#20). For a multi-object scene, skin each object with
    /// [`set_mesh_texture`](Self::set_mesh_texture). The image is (re)uploaded
    /// lazily on the next [`encode`](Self::encode); until set it is 1×1 white.
    pub fn set_texture(&mut self, texture: &dyn Texture) {
        self.set_mesh_texture(0, texture);
    }

    /// Binds `texture` as the albedo of mesh `mesh_id` — so a multi-object scene
    /// skins each object with its **own** diffuse (#141). Out-of-range ids are
    /// ignored. The image uploads lazily on the next [`encode`](Self::encode).
    pub fn set_mesh_texture(&mut self, mesh_id: usize, texture: &dyn Texture) {
        if let Some(mesh) = self.store.meshes.get_mut(mesh_id) {
            mesh.texture.set(texture);
        }
    }

    /// Binds a glTF metallic-roughness map (G=roughness, B=metallic).
    pub fn set_mesh_metallic_roughness_texture(&mut self, mesh_id: usize, texture: &dyn Texture) {
        if let Some(mesh) = self.store.meshes.get_mut(mesh_id) {
            mesh.material_maps.set_metallic_roughness(texture);
        }
    }

    /// Binds a tangent-space glTF normal map.
    pub fn set_mesh_normal_texture(&mut self, mesh_id: usize, texture: &dyn Texture) {
        if let Some(mesh) = self.store.meshes.get_mut(mesh_id) {
            mesh.material_maps.set_normal(texture);
        }
    }

    /// Sets the [`DisneyMaterial`] of **every** mesh — the single-mesh / global
    /// default. For a multi-object scene, give each object its own material with
    /// [`set_mesh_disney_material`](Self::set_mesh_disney_material). Takes effect on the
    /// next [`encode`](Self::encode).
    pub fn set_disney_material(&mut self, material: DisneyMaterial) {
        for m in &mut self.pbr_materials {
            *m = material.clone();
        }
    }

    /// Sets the [`DisneyMaterial`] of mesh `mesh_id` only (#141) — so each
    /// object in a multi-object scene has its own metallic/roughness/base_color.
    /// Out-of-range ids are ignored. Takes effect on the next
    /// [`encode`](Self::encode).
    pub fn set_mesh_disney_material(&mut self, mesh_id: usize, material: DisneyMaterial) {
        if let Some(m) = self.pbr_materials.get_mut(mesh_id) {
            *m = material;
        }
    }

    /// Sets scene lighting controls shared by every PBR object.
    pub fn set_lighting(&mut self, lighting: Lighting) {
        self.lighting = lighting;
    }

    /// Sets image-based-lighting controls for every PBR object.
    pub fn set_image_based_lighting(&mut self, ibl: ImageBasedLighting) {
        self.pbr_ibl.fill(ibl);
    }

    /// Sets image-based-lighting controls for one PBR object.
    pub fn set_mesh_image_based_lighting(&mut self, mesh_id: usize, ibl: ImageBasedLighting) {
        if let Some(current) = self.pbr_ibl.get_mut(mesh_id) {
            *current = ibl;
        }
    }

    /// Sets the per-object output transform of every PBR object.
    pub fn set_tone_mapping(&mut self, tone_mapping: ToneMapping) {
        self.pbr_tone_mappings.fill(tone_mapping);
    }

    /// Sets the output transform of one PBR object.
    pub fn set_mesh_tone_mapping(&mut self, mesh_id: usize, tone_mapping: ToneMapping) {
        if let Some(current) = self.pbr_tone_mappings.get_mut(mesh_id) {
            *current = tone_mapping;
        }
    }

    /// Selects a diagnostic PBR output for one mesh.
    pub fn set_mesh_pbr_debug_view(&mut self, mesh_id: usize, debug_view: PbrDebugView) {
        if let Some(current) = self.pbr_debug_views.get_mut(mesh_id) {
            *current = debug_view;
        }
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
        // 1b. Disney PBR uniform slots for this frame — one per mesh (each carries
        //     the shared camera/lights + that mesh's material, #141). Written
        //     unconditionally so a PBR draw always has a current material slot.
        self.pass.write_pbr(
            queue,
            params,
            viewport,
            PbrInputs {
                materials: &self.pbr_materials,
                ibl: &self.pbr_ibl,
                tone_mappings: &self.pbr_tone_mappings,
                debug_views: &self.pbr_debug_views,
                lighting: self.lighting,
            },
            self.env.has_env(),
        );

        // 2. Walk the scene once into per-geometry instance batches, then upload
        //    the flattened instance models (growing the buffer if needed).
        let batches = build_batches(scene, |mesh_id| {
            self.store.meshes.get(mesh_id).map(|mesh| mesh.base_model)
        });
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

        // 5. (Re)upload each mesh's **own** albedo texture (#141: per-object
        //    diffuse) + the HDR environment map, on first use / after a set —
        //    encode is where a GPU queue is available. Uploads happen up front so
        //    the render loop below only *reads* each mesh's group-1 bind group.
        for mesh in &mut self.store.meshes {
            mesh.texture.ensure_uploaded(&self.device, queue);
            mesh.material_maps.ensure_uploaded(&self.device, queue);
        }
        let env_bind_group = self.env.ensure_uploaded(&self.device, queue);
        if let Some(([rotation, exposure, blur], tonemap)) = batches.environment_background {
            self.env_background.write(
                queue,
                params,
                viewport,
                EnvBackgroundSettings {
                    rotation,
                    exposure,
                    blur,
                    tonemap,
                },
            );
        }

        // 6. Record the pass. With MSAA (`sample_count > 1`) the mesh pass renders
        //    into the multisampled color attachment and resolves into the caller's
        //    single-sample `view`, so every front-end (offscreen CLI, native
        //    window, wasm canvas) gets multisampled mesh/arrowhead edges.
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

        if batches.environment_background.is_some() {
            self.env_background.draw(&mut pass, env_bind_group);
        }

        // Draw the background frame plane first (#63): its own pipeline + group-0
        // bind, depth-write off, so the mesh scene composites on top. Only when
        // the scene requested one (and a frame texture is bound).
        if batches.frame_fit.is_some() {
            self.frame_plane.draw(&mut pass);
        }

        // The instance buffer (slot 1) stays bound across every draw. Most
        // commands use the camera bind group; expanded lines briefly swap in the
        // viewport-aware gizmo bind group.
        pass.set_bind_group(0, &self.pass.camera_bind_group, &[]);
        pass.set_vertex_buffer(1, self.store.instance_buffer.slice(..));
        for command in &batches.commands {
            let range = command.start..command.start + command.count;
            match command.kind {
                DrawKind::Filled(id) => {
                    let mesh = &self.store.meshes[id];
                    pass.set_pipeline(&self.pass.filled);
                    draw_indexed(&mut pass, mesh.filled(), range);
                }
                DrawKind::Textured(id) => {
                    let mesh = &self.store.meshes[id];
                    pass.set_pipeline(&self.pass.textured);
                    pass.set_bind_group(1, mesh.texture.bind_group(), &[]);
                    draw_indexed(&mut pass, mesh.filled(), range);
                }
                DrawKind::Pbr(id) => {
                    let mesh = &self.store.meshes[id];
                    pass.set_pipeline(&self.pass.pbr);
                    // group 0 = this mesh's PbrUniform slot (selected by a dynamic
                    // offset), group 1 = this mesh's albedo, group 2 = HDR env map.
                    let offset = (id as u64 * self.pass.pbr_stride) as u32;
                    pass.set_bind_group(0, &self.pass.pbr_bind_group, &[offset]);
                    pass.set_bind_group(1, mesh.texture.bind_group(), &[]);
                    pass.set_bind_group(2, env_bind_group, &[]);
                    pass.set_bind_group(3, mesh.material_maps.bind_group(), &[]);
                    draw_indexed(&mut pass, mesh.pbr(), range);
                    // Restore group 0 = camera for the following non-PBR draws
                    // (their pipelines' group-0 layout is the camera uniform).
                    pass.set_bind_group(0, &self.pass.camera_bind_group, &[]);
                }
                DrawKind::Wireframe(id) => {
                    let mesh = &self.store.meshes[id];
                    pass.set_pipeline(&self.pass.wireframe);
                    draw_indexed(&mut pass, mesh.wireframe(), range);
                }
                DrawKind::Aabb(id) => {
                    let mesh = &self.store.meshes[id];
                    pass.set_pipeline(&self.pass.gizmo_line);
                    pass.set_bind_group(0, &self.pass.gizmo_bind_group, &[]);
                    draw_vertices(&mut pass, mesh.aabb(), range);
                    pass.set_bind_group(0, &self.pass.camera_bind_group, &[]);
                }
                DrawKind::Grid(plane) => {
                    pass.set_pipeline(&self.pass.gizmo_line);
                    pass.set_bind_group(0, &self.pass.gizmo_bind_group, &[]);
                    draw_vertices(&mut pass, &self.store.grid_lines[plane], range);
                    pass.set_bind_group(0, &self.pass.camera_bind_group, &[]);
                }
                DrawKind::Shadow => {
                    pass.set_pipeline(&self.pass.shadow);
                    pass.set_vertex_buffer(0, self.store.shadow_vertex_buffer.slice(..));
                    pass.draw(0..SHADOW_VERTEX_COUNT, range);
                }
                DrawKind::Axes => {
                    pass.set_pipeline(&self.pass.gizmo_line);
                    pass.set_bind_group(0, &self.pass.gizmo_bind_group, &[]);
                    draw_vertices(&mut pass, &self.store.axes_lines, range.clone());
                    pass.set_pipeline(&self.pass.gizmo_solid);
                    pass.set_bind_group(0, &self.pass.camera_bind_group, &[]);
                    draw_vertices(&mut pass, &self.store.axes_heads, range);
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
            draw_indexed(&mut pass, mesh.filled(), slot..slot + 1);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn model(tag: f32) -> [f32; 16] {
        let mut model = Matrix4::IDENTITY.to_cols_array();
        model[12] = tag;
        model
    }

    fn mesh(mesh_id: u32, tag: f32, mode: RenderMode) -> DrawableObject {
        DrawableObject::Mesh {
            mesh_id,
            model: model(tag),
            mode,
        }
    }

    #[test]
    fn batches_in_layer_order_and_preserves_equal_kind_order() {
        let scene = [
            DrawableObject::CoordinateAxes { model: model(80.0) },
            mesh(1, 61.0, RenderMode::Wireframe),
            DrawableObject::FramePlane {
                fit: FrameFit::Stretch,
            },
            mesh(1, 12.0, RenderMode::Filled),
            mesh(0, 30.0, RenderMode::Pbr),
            DrawableObject::BlobShadow { model: model(1.0) },
            DrawableObject::AabbBox {
                mesh_id: 1,
                model: model(71.0),
            },
            DrawableObject::PlaneGrid {
                plane: GridPlane::Yz,
                model: model(52.0),
            },
            mesh(0, 10.0, RenderMode::Filled),
            mesh(0, 20.0, RenderMode::Textured),
            DrawableObject::PlaneGrid {
                plane: GridPlane::Xy,
                model: model(50.0),
            },
            mesh(0, 11.0, RenderMode::Filled),
            DrawableObject::AabbBox {
                mesh_id: 0,
                model: model(70.0),
            },
            mesh(1, 31.0, RenderMode::Pbr),
            mesh(99, 99.0, RenderMode::Filled),
            mesh(0, 98.0, RenderMode::Shadow),
            DrawableObject::FramePlane {
                fit: FrameFit::Cover,
            },
        ];
        let base_models = [Matrix4::IDENTITY, Matrix4::IDENTITY];

        let batches = build_batches(&scene, |mesh_id| base_models.get(mesh_id).copied());
        let commands = batches
            .commands
            .iter()
            .map(|command| (command.kind, command.start, command.count))
            .collect::<Vec<_>>();

        assert_eq!(
            commands,
            [
                (DrawKind::Shadow, 0, 1),
                (DrawKind::Filled(0), 1, 2),
                (DrawKind::Filled(1), 3, 1),
                (DrawKind::Textured(0), 4, 1),
                (DrawKind::Pbr(0), 5, 1),
                (DrawKind::Pbr(1), 6, 1),
                (DrawKind::Grid(0), 7, 1),
                (DrawKind::Grid(2), 8, 1),
                (DrawKind::Wireframe(1), 9, 1),
                (DrawKind::Aabb(0), 10, 1),
                (DrawKind::Aabb(1), 11, 1),
                (DrawKind::Axes, 12, 1),
            ]
        );
        assert_eq!(
            batches
                .instances
                .iter()
                .map(|instance| instance.model[12])
                .collect::<Vec<_>>(),
            [1.0, 10.0, 11.0, 12.0, 20.0, 30.0, 31.0, 50.0, 52.0, 61.0, 70.0, 71.0, 80.0,]
        );
        assert_eq!(batches.frame_fit, Some(FrameFit::Cover));
    }
}
