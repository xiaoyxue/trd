//! Shared, platform-agnostic parametric triangle rendering.
//!
//! [`render_triangle`] draws the hello-triangle transformed by [`FrameParams`]
//! into the given texture view. [`MeshRenderer`] draws the same triangle through
//! the vertex/index-buffer path used by the native batch renderer.

use crate::math::{Matrix4, Point3, Transform, Vector3};

/// Default clip near/far planes used when deriving a projection from camera
/// intrinsics `K`. The hello-triangle is authored on the `z = 0` plane, so the
/// exact values only need to bracket it; they are renderer constants until the
/// camera slice (#18) makes them configurable.
pub(crate) const DEFAULT_NEAR: f32 = 0.1;
pub(crate) const DEFAULT_FAR: f32 = 1000.0;

/// RGB color of the optional AABB overlay box (bright green), chosen to stand
/// out against the default white mesh. See [`MeshRenderer::set_show_aabb`].
pub(crate) const AABB_COLOR: [f32; 3] = [0.0, 1.0, 0.0];

/// The 12 edges of an axis-aligned box as a `LineList` index buffer, indexing
/// the 8 corners in the order produced by [`crate::math::Aabb3::corners`]
/// (bit 0 = x, bit 1 = y, bit 2 = z of `(lo, hi)`): 4 bottom (`z=lo`) edges, 4
/// top (`z=hi`) edges, then the 4 vertical edges.
pub(crate) const AABB_EDGE_INDICES: [u32; 24] = [
    0, 1, 1, 3, 3, 2, 2, 0, // bottom face (z = lo)
    4, 5, 5, 7, 7, 6, 6, 4, // top face (z = hi)
    0, 4, 1, 5, 2, 6, 3, 7, // vertical edges
];

/// RGB colors of the coordinate-axes overlay gizmo (#42): X = red, Y = green,
/// Z = blue — the conventional right-handed axis coloring. See
/// [`MeshRenderer::set_show_axes`].
pub(crate) const AXES_COLORS: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// World-space length of each coordinate axis in the overlay gizmo. The mesh
/// preview transform ([`crate::Mesh::preview_transform`]) fits a mesh's largest
/// extent to [`crate::mesh::DEFAULT_PREVIEW_TARGET`] world units (so a centered
/// mesh spans about `[-1, 1]` on its largest axis); a length of `1.5` reaches
/// from the world origin out past that half-extent, keeping the axis tips
/// visible just outside the silhouette.
pub(crate) const AXES_LENGTH: f32 = 1.5;

/// Number of `LineList` vertices in the coordinate-axes gizmo (three lines →
/// six vertices), drawn non-indexed. See [`axes_vertices`].
pub(crate) const AXES_VERTEX_COUNT: u32 = 6;

/// The six vertices of the coordinate-axes gizmo as a `LineList`: three lines
/// from the world origin along +X, +Y, +Z, each colored per [`AXES_COLORS`].
/// Drawn non-indexed (`draw(0..6, ..)`) under the camera `P·V` with an identity
/// per-instance model, so the gizmo marks the world origin/frame.
pub(crate) const fn axes_vertices() -> [Vertex; 6] {
    [
        Vertex {
            position: [0.0, 0.0, 0.0],
            color: AXES_COLORS[0],
        },
        Vertex {
            position: [AXES_LENGTH, 0.0, 0.0],
            color: AXES_COLORS[0],
        },
        Vertex {
            position: [0.0, 0.0, 0.0],
            color: AXES_COLORS[1],
        },
        Vertex {
            position: [0.0, AXES_LENGTH, 0.0],
            color: AXES_COLORS[1],
        },
        Vertex {
            position: [0.0, 0.0, 0.0],
            color: AXES_COLORS[2],
        },
        Vertex {
            position: [0.0, 0.0, AXES_LENGTH],
            color: AXES_COLORS[2],
        },
    ]
}

/// Per-frame transform parameters for the triangle.
///
/// The base triangle vertices `p_i` are transformed by the full MVP chain
/// `clip = P · V · M · (p_i, 0, 1)` in the vertex shader, where:
/// - **M** (model) is [`FrameParams::model`] if present, else synthesized from
///   the 2D affine `center`/`size`/`theta` as
///   `translate(center) · rotate_z(theta) · scale(size)` (reproducing the
///   original `p' = center + R(theta) · (size ⊙ p_i)`).
/// - **V** (view) is the camera-from-world transform, resolved (in precedence
///   order) from the **CV** pose [`FrameParams::pose`] (view = `inverse(pose)`),
///   else the **CG** look-at ([`FrameParams::eye`] + [`FrameParams::target`] or
///   [`FrameParams::direction`], with [`FrameParams::up`]), else identity.
/// - **P** (projection) is derived (in precedence order) from the **CV**
///   intrinsics [`FrameParams::k`] + viewport, else the **CG** perspective recipe
///   ([`FrameParams::fovy`] + [`FrameParams::aspect`]/[`FrameParams::znear`]/
///   [`FrameParams::zfar`]), else identity.
///
/// **CV wins over CG** (a well-formed stream carries only one form; mixing is
/// rejected at decode as a conflicting camera form). Any matrix/param that would
/// be identity is simply omitted (its column is absent), so a stream with no
/// camera columns has `P = V = I` and `M` is the 2D affine — byte-for-byte the
/// protocol `0.0.1` result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameParams {
    /// Triangle centroid in NDC; `(0,0)` is screen center.
    pub center: [f32; 2],
    /// Per-axis scale; `(1,1)` is the base triangle.
    pub size: [f32; 2],
    /// Rotation in radians, counter-clockwise.
    pub theta: f32,
    /// Optional explicit 4×4 **model** matrix, column-major (16 floats). When
    /// `Some`, it supersedes `center`/`size`/`theta`.
    pub model: Option<[f32; 16]>,
    /// Optional camera **intrinsics** `K` (**CV** form): a 3×3 pinhole matrix,
    /// column-major (9 floats). `Some` derives the projection; `None` falls back
    /// to the CG projection recipe or identity.
    pub k: Option<[f32; 9]>,
    /// Optional camera **pose** (**CV** form, world-from-camera): a 4×4 matrix,
    /// column-major (16 floats). The view matrix is its inverse; `None` falls
    /// back to the CG look-at or identity.
    pub pose: Option<[f32; 16]>,
    /// Optional camera **eye**/position (**CG** form): world-space `[x, y, z]`.
    pub eye: Option<[f32; 3]>,
    /// Optional CG look-at **target** point: world-space `[x, y, z]`. Takes
    /// precedence over [`FrameParams::direction`] when both are present.
    pub target: Option<[f32; 3]>,
    /// Optional CG forward **direction** the camera looks along from `eye`
    /// (`target = eye + direction`); an alternative to [`FrameParams::target`].
    pub direction: Option<[f32; 3]>,
    /// Optional CG **up** vector; defaults to `+Y` when absent.
    pub up: Option<[f32; 3]>,
    /// Optional CG vertical **field of view** in radians.
    pub fovy: Option<f32>,
    /// Optional CG **aspect** ratio (width/height); defaults to the viewport's.
    pub aspect: Option<f32>,
    /// Optional CG near clip plane; defaults to [`DEFAULT_NEAR`].
    pub znear: Option<f32>,
    /// Optional CG far clip plane; defaults to [`DEFAULT_FAR`].
    pub zfar: Option<f32>,
}

/// A malformed camera specification detected at decode time. Mapped by each
/// decoder to its stream/protocol error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraFormError {
    /// Both the CV form (`k`/`pose`) and the CG form (`eye`/`target`/
    /// `direction`/`fovy`) are present; a stream must use exactly one.
    Conflicting,
    /// The CG form is incomplete: an `eye` without a look `target`/`direction`,
    /// or a look `target`/`direction` without an `eye`.
    Incomplete,
}

/// The render target's pixel dimensions, needed to turn pixel-space camera
/// intrinsics `K` into a clip-space projection.
///
/// The viewport is a **size** (not a matrix baked into the MVP): it supplies the
/// pixel units that `K`'s `fx,fy,cx,cy` live in and the `aspect` ratio for a
/// projection. The NDC→pixel mapping (including the y-flip) is applied by the
/// render target / readback, matching the shipped pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

impl Viewport {
    /// The width/height aspect ratio (`>= 1` for landscape), guarding a zero
    /// height by treating each dimension as at least one pixel.
    #[inline]
    pub fn aspect(self) -> f32 {
        self.width.max(1) as f32 / self.height.max(1) as f32
    }
}

impl FrameParams {
    /// The identity transform: centered, unit scale, no rotation, no camera.
    pub const IDENTITY: FrameParams = FrameParams {
        center: [0.0, 0.0],
        size: [1.0, 1.0],
        theta: 0.0,
        model: None,
        k: None,
        pose: None,
        eye: None,
        target: None,
        direction: None,
        up: None,
        fovy: None,
        aspect: None,
        znear: None,
        zfar: None,
    };

    /// The effective 4×4 model matrix: the explicit [`FrameParams::model`] if
    /// present, else the 2D affine synthesized from `center`/`size`/`theta`.
    /// Used by front-ends to place the default single instance of mesh 0 when a
    /// frame carries no explicit instanced draw list.
    pub fn model_matrix(&self) -> Matrix4 {
        match self.model {
            Some(cols) => Matrix4::from_cols_array(&cols),
            None => model_from_2d_affine(self.center, self.size, self.theta),
        }
    }

    /// The view matrix `camera-from-world`, resolved in precedence order:
    /// **CV** `inverse(pose)`, else **CG** look-at (`eye` → `target` or
    /// `eye + direction`, with `up` defaulting to `+Y`), else identity.
    pub(crate) fn view_matrix(&self) -> Matrix4 {
        // CV form wins over CG.
        if let Some(cols) = self.pose {
            return Matrix4::from_cols_array(&cols).inverse();
        }
        if let Some(eye) = self.eye {
            let eye = Point3::new(eye[0], eye[1], eye[2]);
            let up = self
                .up
                .map(|u| Vector3::new(u[0], u[1], u[2]))
                .unwrap_or(Vector3::Y);
            // A look-at `target` takes precedence over a forward `direction`.
            let target = if let Some(t) = self.target {
                Point3::new(t[0], t[1], t[2])
            } else if let Some(d) = self.direction {
                eye + Vector3::new(d[0], d[1], d[2])
            } else {
                // Incomplete CG form (rejected at decode); be lenient here.
                return Matrix4::IDENTITY;
            };
            return Transform::look_at_rh(eye, target, up).matrix();
        }
        Matrix4::IDENTITY
    }

    /// The projection matrix, resolved in precedence order: **CV** intrinsics
    /// `K` + viewport, else **CG** perspective (`fovy`, `aspect` defaulting to
    /// the viewport's, `znear`/`zfar` defaulting to [`DEFAULT_NEAR`]/
    /// [`DEFAULT_FAR`]), else identity.
    pub(crate) fn projection_matrix(&self, viewport: Viewport) -> Matrix4 {
        if let Some(k) = self.k {
            return projection_from_intrinsics(k, viewport);
        }
        if let Some(fovy) = self.fovy {
            let aspect = self.aspect.unwrap_or_else(|| viewport.aspect());
            let near = self.znear.unwrap_or(DEFAULT_NEAR);
            let far = self.zfar.unwrap_or(DEFAULT_FAR);
            return Transform::perspective_rh(fovy, aspect, near, far).matrix();
        }
        Matrix4::IDENTITY
    }

    /// Validates the camera specification: exactly one of the CV (`k`/`pose`)
    /// and CG (`eye`/`target`/`direction`/`fovy`) forms, and a complete CG
    /// look-at (`eye` iff a look `target`/`direction`). A stream with no camera
    /// columns (all `None`) is valid (identity camera).
    pub(crate) fn check_camera_form(&self) -> Result<(), CameraFormError> {
        let cv = self.k.is_some() || self.pose.is_some();
        let look = self.target.is_some() || self.direction.is_some();
        let cg = self.eye.is_some() || look || self.fovy.is_some();
        if cv && cg {
            return Err(CameraFormError::Conflicting);
        }
        if self.eye.is_some() != look {
            return Err(CameraFormError::Incomplete);
        }
        Ok(())
    }

    /// The full clip transform `P · V · M` for a given viewport.
    pub(crate) fn clip_transform(&self, viewport: Viewport) -> Matrix4 {
        self.clip_transform_with_base(viewport, Matrix4::IDENTITY)
    }

    /// The clip transform `P · V · M · base`, where `base` is a model-space
    /// pre-transform applied before the per-frame model. Used by the mesh path to
    /// apply a mesh's [`crate::Mesh::preview_transform`] (center + scale-to-fit)
    /// beneath the per-frame `model` (e.g. a turntable rotation), so an
    /// arbitrary-unit asset renders centered and at a reasonable size.
    pub(crate) fn clip_transform_with_base(&self, viewport: Viewport, base: Matrix4) -> Matrix4 {
        self.projection_matrix(viewport) * self.view_matrix() * self.model_matrix() * base
    }

    /// The camera-only transform `P · V` for a given viewport, used by the
    /// instanced mesh path where each drawn instance supplies its own model
    /// matrix (`clip = P · V · M · p`).
    pub(crate) fn view_proj_matrix(&self, viewport: Viewport) -> Matrix4 {
        self.projection_matrix(viewport) * self.view_matrix()
    }
}

/// Builds the 2D-affine model matrix `translate(center) · rotate_z(theta) ·
/// scale(size)` (z untouched), the `0.0.1` transform expressed as a [`Matrix4`].
pub(crate) fn model_from_2d_affine(center: [f32; 2], size: [f32; 2], theta: f32) -> Matrix4 {
    Matrix4::from_translation(Vector3::new(center[0], center[1], 0.0))
        * Matrix4::from_rotation_z(theta)
        * Matrix4::from_scale(Vector3::new(size[0], size[1], 1.0))
}

/// Builds a right-handed, wgpu-clip-space (`z ∈ [0, 1]`) perspective projection
/// from a pinhole intrinsics matrix `K` (column-major: `fx = k[0]`, skew
/// `s = k[3]`, `fy = k[4]`, `cx = k[6]`, `cy = k[7]`) and the target viewport.
///
/// Conventions (to be validated visually / refined in the camera slice #18):
/// `K` shares NDC orientation (x right, y up, camera looking down `-z`); a
/// centered principal point (`cx = W/2`, `cy = H/2`) with square pixels and no
/// skew reduces to [`glam::Mat4::perspective_rh`]. Skew `s` shears the
/// projection (couples camera-`y` into clip-`x`). `near`/`far` are
/// [`DEFAULT_NEAR`]/[`DEFAULT_FAR`]. This is the exact inverse of
/// [`crate::Camera::to_intrinsics`], so `K ⇄ projection` round-trips losslessly.
pub(crate) fn projection_from_intrinsics(k: [f32; 9], viewport: Viewport) -> Matrix4 {
    let fx = k[0];
    let s = k[3];
    let fy = k[4];
    let cx = k[6];
    let cy = k[7];
    let w = viewport.width.max(1) as f32;
    let h = viewport.height.max(1) as f32;
    let (n, f) = (DEFAULT_NEAR, DEFAULT_FAR);

    // Column-major: each row below is one column of the matrix.
    Matrix4::from_cols_array(&[
        2.0 * fx / w,
        0.0,
        0.0,
        0.0,
        2.0 * s / w,
        2.0 * fy / h,
        0.0,
        0.0,
        2.0 * cx / w - 1.0,
        2.0 * cy / h - 1.0,
        f / (n - f),
        -1.0,
        0.0,
        0.0,
        (f * n) / (n - f),
        0.0,
    ])
}

/// GPU uniform matching the WGSL `Params` layout: a single column-major 4×4
/// matrix (64 bytes). The triangle path stores the full clip transform
/// `P · V · M`; the instanced mesh path stores the camera-only `P · V` (each
/// instance supplies its own model matrix).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
    transform: [f32; 16],
}

impl Uniform {
    fn from_params(params: FrameParams, viewport: Viewport) -> Self {
        Uniform {
            transform: params.clip_transform(viewport).to_cols_array(),
        }
    }

    fn view_proj(params: FrameParams, viewport: Viewport) -> Self {
        Uniform {
            transform: params.view_proj_matrix(viewport).to_cols_array(),
        }
    }
}

/// A mesh vertex consumed by `mesh.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 12,
            shader_location: 1,
        },
    ];

    /// Returns the vertex buffer layout expected by `mesh.wgsl`.
    pub const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// Per-instance model matrix fed to `mesh.wgsl` as four `vec4` instance
/// attributes (shader locations 2-5, column-major, 64-byte stride).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [f32; 16],
}

impl InstanceRaw {
    const ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 0,
            shader_location: 2,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 16,
            shader_location: 3,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 32,
            shader_location: 4,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 48,
            shader_location: 5,
        },
    ];

    /// Returns the per-instance buffer layout expected by `mesh.wgsl`.
    const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// Canonical indexed mesh container.
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    /// The legacy hello-triangle expressed as a 3-vertex indexed mesh.
    pub fn hello_triangle() -> Self {
        Self {
            vertices: vec![
                Vertex {
                    position: [0.0, 0.5, 0.0],
                    color: [1.0, 0.0, 0.0],
                },
                Vertex {
                    position: [-0.5, -0.5, 0.0],
                    color: [0.0, 1.0, 0.0],
                },
                Vertex {
                    position: [0.5, -0.5, 0.0],
                    color: [0.0, 0.0, 1.0],
                },
            ],
            indices: vec![0, 1, 2],
        }
    }
}

/// Builds the triangle render pipeline for `format` using an auto bind-group
/// layout (group 0, binding 0 = the params uniform).
pub fn create_triangle_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("triangle.wgsl"));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("trd triangle pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Builds the indexed mesh render pipeline for `format` using an auto bind-group
/// layout (group 0, binding 0 = the params uniform), drawn as filled triangles.
pub fn create_mesh_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("trd mesh pipeline layout"),
        bind_group_layouts: &[Some(&create_mesh_bind_group_layout(device))],
        immediate_size: 0,
    });
    create_mesh_pipeline_with(
        device,
        format,
        &layout,
        wgpu::PrimitiveTopology::TriangleList,
    )
}

/// The explicit bind-group layout shared by every mesh pipeline (group 0,
/// binding 0 = the camera `P·V` uniform, vertex-stage visible). Making it
/// explicit (rather than auto-derived per pipeline) lets the filled and
/// wireframe pipelines share **one** layout, so a single params bind group is
/// valid for both regardless of the active [`RenderMode`].
pub(crate) fn create_mesh_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("trd mesh bind group layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// Builds an indexed mesh pipeline for `format` and `topology` (filled
/// `TriangleList` or wireframe `LineList`) over the shared explicit `layout`.
/// Both topologies use the same `mesh.wgsl` (the vertex shader only transforms
/// positions; line rasterization needs no extra WebGPU features).
fn create_mesh_pipeline_with(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    layout: &wgpu::PipelineLayout,
    topology: wgpu::PrimitiveTopology,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("mesh.wgsl"));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("trd mesh pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[Some(Vertex::layout()), Some(InstanceRaw::layout())],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(format.into())],
        }),
        primitive: wgpu::PrimitiveState {
            topology,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Creates the params uniform buffer + bind group for `pipeline`, initialised
/// to `params` for the given `viewport` (needed to project camera intrinsics).
pub(crate) fn create_params_binding(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    params: FrameParams,
    viewport: Viewport,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    use wgpu::util::DeviceExt;
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("trd params uniform"),
        contents: bytemuck::bytes_of(&Uniform::from_params(params, viewport)),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("trd params bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    (buffer, bind_group)
}

/// Creates the camera `P·V` uniform buffer + bind group over an **explicit**
/// bind-group layout (shared by the filled and wireframe mesh pipelines),
/// initialised to `params`'s view-projection for `viewport`. Used by
/// [`MeshRenderer`], whose two pipelines must share one bind group.
pub(crate) fn create_view_proj_binding(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: FrameParams,
    viewport: Viewport,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    use wgpu::util::DeviceExt;
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("trd view-proj uniform"),
        contents: bytemuck::bytes_of(&Uniform::view_proj(params, viewport)),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("trd view-proj bind group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    (buffer, bind_group)
}
/// [`MeshRenderer`], which reuse one uniform buffer across frames instead of
/// rebuilding it.
pub(crate) fn write_params(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    params: FrameParams,
    viewport: Viewport,
) {
    queue.write_buffer(
        buffer,
        0,
        bytemuck::bytes_of(&Uniform::from_params(params, viewport)),
    );
}

/// Like [`write_params`] but writes the camera-only `P · V` transform (the
/// instanced mesh path supplies each model matrix per instance).
pub(crate) fn write_view_proj(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    params: FrameParams,
    viewport: Viewport,
) {
    queue.write_buffer(
        buffer,
        0,
        bytemuck::bytes_of(&Uniform::view_proj(params, viewport)),
    );
}

/// Draws the transformed triangle into `view`, clearing to black first.
///
/// `width`/`height` are the target's pixel dimensions, used to project camera
/// intrinsics (`FrameParams::k`); they have no effect on the identity/2D-affine
/// path. Builds a fresh pipeline and uniform each call; intended for one-shot
/// callers. The batch renderer reuses a persistent pipeline.
pub fn render_triangle(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    view: &wgpu::TextureView,
    format: wgpu::TextureFormat,
    params: FrameParams,
    width: u32,
    height: u32,
) {
    let pipeline = create_triangle_pipeline(device, format);
    let (_buffer, bind_group) =
        create_params_binding(device, &pipeline, params, Viewport { width, height });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("trd triangle encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("trd triangle pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
    queue.submit(Some(encoder.finish()));
}

/// Persistent triangle renderer: owns one pipeline, uniform buffer, and bind
/// group, and encodes a single frame into a caller-provided encoder and view.
///
/// `encode` never creates a command encoder, submits, acquires a surface, or
/// presents; those belong to the target adapter (CLI readback or canvas).
pub struct TriangleRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl TriangleRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let pipeline = create_triangle_pipeline(device, format);
        // The identity params ignore the viewport (no intrinsics); each `encode`
        // supplies the real target dimensions.
        let (uniform, bind_group) = create_params_binding(
            device,
            &pipeline,
            FrameParams::IDENTITY,
            Viewport {
                width: 1,
                height: 1,
            },
        );
        Self {
            pipeline,
            uniform,
            bind_group,
        }
    }

    /// Encodes one frame. `width`/`height` are the target's pixel dimensions,
    /// used to project camera intrinsics (`FrameParams::k`); they have no effect
    /// on the identity/2D-affine path.
    pub fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        params: FrameParams,
        width: u32,
        height: u32,
    ) {
        write_params(queue, &self.uniform, params, Viewport { width, height });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("trd triangle pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// How a [`MeshRenderer`] rasterizes its meshes: solid filled triangles, or an
/// edge **wireframe** (`LineList` over the derived [`crate::Mesh::edge_indices`]
/// buffer). Default is [`RenderMode::Filled`]; wireframe (#38) is opt-in via
/// [`MeshRenderer::set_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderMode {
    /// Draw triangles filled (the mesh's triangle index buffer).
    #[default]
    Filled,
    /// Draw only triangle edges as lines (the deduped edge index buffer).
    Wireframe,
}

/// A single instance placement decoded from a frame's protocol draw list
/// (`draw_mesh` / `draw_model`): which mesh to draw (index into the leading mesh
/// table) and the per-instance model matrix (column-major), applied beneath that
/// mesh's base (preview) model. This is the *wire* representation; the renderer
/// composes it (plus core gizmos) into a [`Scene`] of [`DrawableObject`]s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Draw {
    pub mesh_id: u32,
    pub model: [f32; 16],
}

/// The base interface for every primitive the renderer can draw (#41). A
/// `DrawableObject` is a light, `Copy` handle: geometry (GPU buffers) is owned
/// once by the renderer's decode-once store (meshes keyed by id, plus the shared
/// gizmo geometry), and each variant carries only *which* primitive to draw and
/// its per-frame model. The renderer and [`Scene`] only ever see
/// `DrawableObject`s and never special-case a concrete primitive type.
///
/// Wireframe is a render *mode* of the [`DrawableObject::Mesh`] primitive (not a
/// separate variant); the coordinate axes and the AABB box are genuinely
/// distinct line-topology primitives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DrawableObject {
    /// A decoded mesh (id = row index in the leading mesh table) placed by
    /// `model` and drawn in `mode` (filled or wireframe). `model` is the
    /// per-frame draw model; the renderer pre-multiplies the mesh's base
    /// (preview) model beneath it (`effective = model · base`).
    Mesh {
        mesh_id: u32,
        model: [f32; 16],
        mode: RenderMode,
    },
    /// The axis-aligned bounding-box wireframe of mesh `mesh_id` (#42), placed by
    /// the same `model` as the mesh instance it boxes (the renderer applies that
    /// mesh's base model beneath `model` too), so the box tracks the mesh
    /// exactly. Reuses the mesh's precomputed corner geometry.
    AabbBox { mesh_id: u32, model: [f32; 16] },
    /// The world-orientation coordinate gizmo (#42): three lines from the origin
    /// along +X/+Y/+Z, colored red/green/blue. Placed by `model` (identity marks
    /// the world origin); not tied to any mesh, so no base model is applied.
    CoordinateAxes { model: [f32; 16] },
}

/// A frame's ordered list of [`DrawableObject`]s the renderer walks and encodes
/// under the one shared camera `P·V` uniform. The wire authors the mesh draws
/// (the protocol 0.0.3 draw list); the core adds gizmo drawables (axes, AABB
/// boxes). A single-mesh frame is the degenerate one-element scene — the
/// renderer always iterates a `Scene`, with no single-object special case.
pub type Scene = Vec<DrawableObject>;

/// Builds a per-frame [`Scene`] from a wire `draws` list plus the render `mode`
/// and overlay flags. Each [`Draw`] becomes one [`DrawableObject::Mesh`] in
/// `mode`; with `show_aabb`, each also emits a tracking
/// [`DrawableObject::AabbBox`]; with `show_axes`, one origin
/// [`DrawableObject::CoordinateAxes`] is appended. The order (all meshes, then
/// all boxes, then axes) matches the renderer's draw buckets so output is
/// pixel-identical to the pre-scene, flag-driven path.
///
/// Shared by the native ([`crate::run_stream`]) and wasm front-ends so neither
/// branches per primitive type: both author the same ordered `Scene` and hand
/// it to [`MeshRenderer::encode`].
pub fn build_scene(draws: &[Draw], mode: RenderMode, show_aabb: bool, show_axes: bool) -> Scene {
    let mut scene = Vec::with_capacity(draws.len() * (1 + usize::from(show_aabb)) + 1);
    for draw in draws {
        scene.push(DrawableObject::Mesh {
            mesh_id: draw.mesh_id,
            model: draw.model,
            mode,
        });
    }
    if show_aabb {
        for draw in draws {
            scene.push(DrawableObject::AabbBox {
                mesh_id: draw.mesh_id,
                model: draw.model,
            });
        }
    }
    if show_axes {
        scene.push(DrawableObject::CoordinateAxes {
            model: Matrix4::IDENTITY.to_cols_array(),
        });
    }
    scene
}

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
pub struct MeshRenderer {
    pipeline: wgpu::RenderPipeline,
    wireframe_pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    meshes: Vec<MeshGpu>,
    /// The coordinate-axes gizmo geometry (six `LineList` vertices); each
    /// [`DrawableObject::CoordinateAxes`] draws it under its own model, supplied
    /// through the shared instance buffer.
    axes_vertex_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    instance_capacity: u32,
    /// Retained so `encode` can grow the instance buffer on demand without the
    /// caller threading a `&Device` through every call (`wgpu::Device` is a
    /// cheap `Arc` handle).
    device: wgpu::Device,
}

impl MeshRenderer {
    /// Builds a single-mesh renderer with no base model (vertices are drawn in
    /// their own coordinates). Use [`MeshRenderer::with_base_model`] to apply a
    /// preview/normalization transform beneath the per-frame model.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, mesh: &Mesh) -> Self {
        Self::with_base_model(device, format, mesh, Matrix4::IDENTITY)
    }

    /// Like [`MeshRenderer::new`] but pre-multiplies `base_model` beneath every
    /// frame's model — used to apply a mesh's [`crate::Mesh::preview_transform`]
    /// (center + scale-to-fit).
    pub fn with_base_model(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        mesh: &Mesh,
        base_model: Matrix4,
    ) -> Self {
        Self::with_meshes(device, format, std::slice::from_ref(mesh), &[base_model])
    }

    /// Like [`with_meshes`](Self::with_meshes) but derives each mesh's base
    /// (preview) model automatically via [`Mesh::preview_transform`]
    /// ([`crate::DEFAULT_PREVIEW_TARGET`]) — center + uniform scale-to-fit — so an
    /// arbitrary-unit asset renders centered at a reasonable size. Shared by the
    /// headless [`crate::run_stream`]/`BatchRenderer` and the windowed `trd-app`.
    pub fn with_meshes_preview(
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
        Self::with_meshes(device, format, meshes, &base_models)
    }

    /// Builds a renderer over several meshes, each with its own base (preview)
    /// model. A frame's [`Scene`] references these meshes by id (row index).
    ///
    /// Panics if `meshes` is empty or `meshes`/`base_models` differ in length.
    pub fn with_meshes(
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
        );
        let wireframe_pipeline = create_mesh_pipeline_with(
            device,
            format,
            &pipeline_layout,
            wgpu::PrimitiveTopology::LineList,
        );
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
            uniform,
            bind_group,
            meshes: gpu_meshes,
            axes_vertex_buffer,
            instance_buffer,
            instance_capacity,
            device: device.clone(),
        }
    }

    /// The number of meshes this renderer can draw; valid mesh ids in a
    /// [`DrawableObject::Mesh`]/[`DrawableObject::AabbBox`] are in
    /// `0..mesh_count()`.
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
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
    /// model. Gizmo overlays (AABB boxes, axes) are composited after all mesh
    /// geometry so they stay visible (this path has no depth buffer).
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

        // Walk the scene once, bucketing each drawable's instance model by the
        // geometry it draws so same-geometry instances share one draw call.
        let mesh_count = self.meshes.len();
        let mut filled: Vec<Vec<InstanceRaw>> = vec![Vec::new(); mesh_count];
        let mut wireframe: Vec<Vec<InstanceRaw>> = vec![Vec::new(); mesh_count];
        let mut aabb: Vec<Vec<InstanceRaw>> = vec![Vec::new(); mesh_count];
        let mut axes: Vec<InstanceRaw> = Vec::new();

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
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Point3, Rotation, Transform};
    use approx::assert_abs_diff_eq;
    use glam::{Mat4, Vec3, Vec4};

    #[test]
    fn uniform_layout_matches_wgsl_params() {
        // One column-major 4x4 f32 matrix = 64 bytes.
        assert_eq!(std::mem::size_of::<Uniform>(), 64);
        let viewport = Viewport {
            width: 8,
            height: 4,
        };
        assert_eq!(
            Uniform::from_params(FrameParams::IDENTITY, viewport).transform,
            Matrix4::IDENTITY.to_cols_array()
        );
    }

    #[test]
    fn vertex_layout_matches_wgsl_inputs() {
        assert_eq!(std::mem::size_of::<Vertex>(), 24);
        assert_eq!(std::mem::align_of::<Vertex>(), 4);

        let layout = Vertex::layout();
        assert_eq!(layout.array_stride, 24);
        assert_eq!(layout.step_mode, wgpu::VertexStepMode::Vertex);
        assert_eq!(layout.attributes.len(), 2);
        assert_eq!(layout.attributes[0].offset, 0);
        assert_eq!(layout.attributes[0].shader_location, 0);
        assert_eq!(layout.attributes[0].format, wgpu::VertexFormat::Float32x3);
        assert_eq!(layout.attributes[1].offset, 12);
        assert_eq!(layout.attributes[1].shader_location, 1);
        assert_eq!(layout.attributes[1].format, wgpu::VertexFormat::Float32x3);
    }

    #[test]
    fn hello_triangle_mesh_matches_shader_constants() {
        let mesh = Mesh::hello_triangle();
        assert_eq!(
            mesh.vertices,
            vec![
                Vertex {
                    position: [0.0, 0.5, 0.0],
                    color: [1.0, 0.0, 0.0],
                },
                Vertex {
                    position: [-0.5, -0.5, 0.0],
                    color: [0.0, 1.0, 0.0],
                },
                Vertex {
                    position: [0.5, -0.5, 0.0],
                    color: [0.0, 0.0, 1.0],
                },
            ]
        );
        assert_eq!(mesh.indices, vec![0, 1, 2]);
    }

    #[test]
    fn identity_params_produce_identity_model() {
        assert_eq!(FrameParams::IDENTITY.model_matrix(), Matrix4::IDENTITY);
    }

    #[test]
    fn explicit_model_supersedes_2d_affine() {
        // A `model` column value is used verbatim, regardless of center/size/theta.
        let cols: [f32; 16] = std::array::from_fn(|i| i as f32 + 1.0);
        let params = FrameParams {
            center: [9.0, 9.0],
            size: [9.0, 9.0],
            theta: 9.0,
            model: Some(cols),
            ..FrameParams::IDENTITY
        };
        assert_eq!(params.model_matrix(), Matrix4::from_cols_array(&cols));
    }

    #[test]
    fn synthesized_model_reproduces_2d_affine_transform() {
        // The synthesized model must map base vertices exactly like the legacy
        // `p' = center + R(theta) * (size ⊙ p)` formula.
        let center = [0.1_f32, -0.2];
        let size = [0.5_f32, 0.75];
        let theta = 1.25_f32;
        let model = Transform::from_matrix(model_from_2d_affine(center, size, theta));

        for base in [[0.0_f32, 0.5], [-0.5, -0.5], [0.5, -0.5]] {
            let scaled = [base[0] * size[0], base[1] * size[1]];
            let (s, c) = theta.sin_cos();
            let expected = [
                center[0] + c * scaled[0] - s * scaled[1],
                center[1] + s * scaled[0] + c * scaled[1],
            ];
            let got = model.transform_point(Point3::new(base[0], base[1], 0.0));
            assert!(
                (got.x() - expected[0]).abs() < 1e-6,
                "x: {got:?} {expected:?}"
            );
            assert!(
                (got.y() - expected[1]).abs() < 1e-6,
                "y: {got:?} {expected:?}"
            );
            assert!(got.z().abs() < 1e-6, "z should stay 0: {got:?}");
        }
    }

    #[test]
    fn view_matrix_is_pose_inverse() {
        let pose = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0))
            * Mat4::from_rotation_y(0.7)
            * Mat4::from_rotation_x(-0.3);
        let params = FrameParams {
            pose: Some(pose.to_cols_array()),
            ..FrameParams::IDENTITY
        };
        assert_abs_diff_eq!(
            params.view_matrix().into_inner(),
            pose.inverse(),
            epsilon = 1e-5
        );
        // No pose => identity view.
        assert_eq!(FrameParams::IDENTITY.view_matrix(), Matrix4::IDENTITY);
    }

    #[test]
    fn cg_view_matches_look_at() {
        // The CG `eye`/`target`/`up` form resolves to the same view matrix as a
        // direct look-at.
        let params = FrameParams {
            eye: Some([2.0, 3.0, 5.0]),
            target: Some([0.1, 0.2, 0.3]),
            up: Some([0.0, 1.0, 0.0]),
            fovy: Some(0.9),
            ..FrameParams::IDENTITY
        };
        let expected = Transform::look_at_rh(
            Point3::new(2.0, 3.0, 5.0),
            Point3::new(0.1, 0.2, 0.3),
            Vector3::Y,
        )
        .matrix();
        assert_abs_diff_eq!(
            params.view_matrix().into_inner(),
            expected.into_inner(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn cg_direction_matches_target_at_eye_plus_direction() {
        // `direction` resolves to `target = eye + direction`; `up` defaults to +Y.
        let eye = [1.0, 2.0, 3.0];
        let dir = [0.0, 0.0, -1.0];
        let via_dir = FrameParams {
            eye: Some(eye),
            direction: Some(dir),
            ..FrameParams::IDENTITY
        };
        let via_target = FrameParams {
            eye: Some(eye),
            target: Some([1.0, 2.0, 2.0]),
            ..FrameParams::IDENTITY
        };
        assert_abs_diff_eq!(
            via_dir.view_matrix().into_inner(),
            via_target.view_matrix().into_inner(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn cv_pose_wins_over_cg_view() {
        // Even if CG `eye`/`target` are present, a CV `pose` takes precedence for
        // the view matrix (a well-formed stream never mixes them; this pins the
        // resolution order).
        let pose = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)) * Mat4::from_rotation_y(0.4);
        let params = FrameParams {
            pose: Some(pose.to_cols_array()),
            eye: Some([9.0, 9.0, 9.0]),
            target: Some([0.0, 0.0, 0.0]),
            ..FrameParams::IDENTITY
        };
        assert_abs_diff_eq!(
            params.view_matrix().into_inner(),
            pose.inverse(),
            epsilon = 1e-5
        );
    }

    #[test]
    fn cg_projection_matches_perspective() {
        let viewport = Viewport {
            width: 800,
            height: 600,
        };
        let params = FrameParams {
            fovy: Some(0.9),
            aspect: Some(1.5),
            znear: Some(0.5),
            zfar: Some(50.0),
            eye: Some([0.0, 0.0, 1.0]),
            target: Some([0.0, 0.0, 0.0]),
            ..FrameParams::IDENTITY
        };
        let expected = Transform::perspective_rh(0.9, 1.5, 0.5, 50.0).matrix();
        assert_abs_diff_eq!(
            params.projection_matrix(viewport).into_inner(),
            expected.into_inner(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn cg_projection_defaults_aspect_and_clip_planes() {
        // `aspect` defaults to the viewport's; `znear`/`zfar` to the renderer's.
        let viewport = Viewport {
            width: 800,
            height: 400,
        };
        let params = FrameParams {
            fovy: Some(0.8),
            eye: Some([0.0, 0.0, 1.0]),
            target: Some([0.0, 0.0, 0.0]),
            ..FrameParams::IDENTITY
        };
        let expected =
            Transform::perspective_rh(0.8, viewport.aspect(), DEFAULT_NEAR, DEFAULT_FAR).matrix();
        assert_abs_diff_eq!(
            params.projection_matrix(viewport).into_inner(),
            expected.into_inner(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn camera_form_validation() {
        // No camera columns: valid (identity camera).
        assert_eq!(FrameParams::IDENTITY.check_camera_form(), Ok(()));
        // CV form (k + pose): valid.
        assert_eq!(
            FrameParams {
                k: Some([0.0; 9]),
                pose: Some([0.0; 16]),
                ..FrameParams::IDENTITY
            }
            .check_camera_form(),
            Ok(())
        );
        // CG look-at (eye + target): valid.
        assert_eq!(
            FrameParams {
                eye: Some([0.0; 3]),
                target: Some([0.0; 3]),
                fovy: Some(1.0),
                ..FrameParams::IDENTITY
            }
            .check_camera_form(),
            Ok(())
        );
        // CG forward (eye + direction): valid.
        assert_eq!(
            FrameParams {
                eye: Some([0.0; 3]),
                direction: Some([0.0, 0.0, -1.0]),
                ..FrameParams::IDENTITY
            }
            .check_camera_form(),
            Ok(())
        );
        // Mixing CV and CG: rejected.
        assert_eq!(
            FrameParams {
                k: Some([0.0; 9]),
                eye: Some([0.0; 3]),
                target: Some([0.0; 3]),
                ..FrameParams::IDENTITY
            }
            .check_camera_form(),
            Err(CameraFormError::Conflicting)
        );
        // `eye` without a look target/direction: incomplete.
        assert_eq!(
            FrameParams {
                eye: Some([0.0; 3]),
                ..FrameParams::IDENTITY
            }
            .check_camera_form(),
            Err(CameraFormError::Incomplete)
        );
        // Look `target` without an `eye`: incomplete.
        assert_eq!(
            FrameParams {
                target: Some([0.0; 3]),
                ..FrameParams::IDENTITY
            }
            .check_camera_form(),
            Err(CameraFormError::Incomplete)
        );
    }

    #[test]
    fn centered_square_intrinsics_match_glam_perspective() {
        // A centered principal point with square pixels must reduce to glam's
        // right-handed perspective (fov_y from fy).
        let viewport = Viewport {
            width: 800,
            height: 600,
        };
        let (w, h) = (viewport.width as f32, viewport.height as f32);
        let f = 500.0_f32; // fx = fy
        let k = [f, 0.0, 0.0, 0.0, f, 0.0, w / 2.0, h / 2.0, 1.0];

        let got = projection_from_intrinsics(k, viewport);
        let fov_y = 2.0 * (h / (2.0 * f)).atan();
        let expected = Mat4::perspective_rh(fov_y, w / h, DEFAULT_NEAR, DEFAULT_FAR);
        assert_abs_diff_eq!(got.into_inner(), expected, epsilon = 1e-4);
    }

    #[test]
    fn principal_axis_projects_to_principal_point() {
        // A camera-space point straight ahead (on the optical axis, -z) lands at
        // the principal point in NDC (0,0 for a centered K).
        let viewport = Viewport {
            width: 640,
            height: 480,
        };
        let (w, h) = (viewport.width as f32, viewport.height as f32);
        let k = [400.0, 0.0, 0.0, 0.0, 400.0, 0.0, w / 2.0, h / 2.0, 1.0];
        let p = projection_from_intrinsics(k, viewport);
        let clip = p.into_inner() * Vec4::new(0.0, 0.0, -5.0, 1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(ndc.x.abs() < 1e-5 && ndc.y.abs() < 1e-5, "ndc = {ndc:?}");
    }

    #[test]
    fn no_camera_clip_transform_equals_model() {
        // Without pose/intrinsics, P = V = I so the clip transform is the model.
        let params = FrameParams {
            center: [0.2, -0.3],
            size: [0.5, 0.5],
            theta: 0.4,
            ..FrameParams::IDENTITY
        };
        let viewport = Viewport {
            width: 256,
            height: 256,
        };
        assert_abs_diff_eq!(
            params.clip_transform(viewport).into_inner(),
            params.model_matrix().into_inner(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn math_transform_reproduces_2d_affine_model() {
        // The typed `math::Transform` API rebuilds the legacy
        // `translate · rotate_z · scale` model matrix that drives the GPU
        // uniform — the quaternion `Transform` path matches the direct-trig
        // `Matrix4` path within tolerance.
        let center = [0.2_f32, -0.3];
        let size = [0.5_f32, 0.75];
        let theta = 0.4_f32;

        // `a.then(b) == b * a`, so this is translate · rotate_z · scale.
        let t = Transform::from_scale(Vector3::new(size[0], size[1], 1.0))
            .then(Transform::from_rotation(Rotation::from_rotation_z(theta)))
            .then(Transform::from_translation(Vector3::new(
                center[0], center[1], 0.0,
            )));

        let expected = model_from_2d_affine(center, size, theta);
        assert_abs_diff_eq!(
            t.matrix().into_inner(),
            expected.into_inner(),
            epsilon = 1e-6
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn render_with_readback(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        encode: impl FnOnce(&wgpu::Queue, &mut wgpu::CommandEncoder, &wgpu::TextureView),
    ) -> Vec<u8> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("trd render test target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let unpadded = width * 4;
        let padded_bytes_per_row = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trd render test readback"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("trd render test encoder"),
        });
        encode(queue, &mut encoder, &view);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        rx.recv()
            .expect("map_async callback dropped")
            .expect("GPU readback failed");

        let pixels = {
            let mapped = slice.get_mapped_range().expect("buffer mapped after poll");
            crate::tightly_pack_rgba(&mapped, width, height, padded_bytes_per_row)
                .expect("GPU row unpack failed")
        };
        staging.unmap();
        pixels
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn test_device() -> (wgpu::Device, wgpu::Queue) {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("GPU adapter required");
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("trd mesh continuity test device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("request_device failed")
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    #[cfg(not(target_arch = "wasm32"))]
    fn mesh_renderer_matches_triangle_renderer_pixels() {
        let (device, queue) = pollster::block_on(test_device());
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let (width, height) = (64, 64);
        let triangle = TriangleRenderer::new(&device, format);
        let mut mesh = MeshRenderer::new(&device, format, &Mesh::hello_triangle());

        let triangle_pixels = render_with_readback(
            &device,
            &queue,
            format,
            width,
            height,
            |queue, encoder, view| {
                triangle.encode(queue, encoder, view, FrameParams::IDENTITY, width, height);
            },
        );
        let scene = [DrawableObject::Mesh {
            mesh_id: 0,
            model: Matrix4::IDENTITY.to_cols_array(),
            mode: RenderMode::Filled,
        }];
        let mesh_pixels = render_with_readback(
            &device,
            &queue,
            format,
            width,
            height,
            |queue, encoder, view| {
                mesh.encode(
                    queue,
                    encoder,
                    view,
                    FrameParams::IDENTITY,
                    &scene,
                    Viewport { width, height },
                );
            },
        );

        assert_eq!(mesh_pixels, triangle_pixels);
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    #[cfg(not(target_arch = "wasm32"))]
    fn mesh_renderer_draws_multiple_instances() {
        let (device, queue) = pollster::block_on(test_device());
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let (width, height) = (64, 64);
        let mut mesh = MeshRenderer::new(&device, format, &Mesh::hello_triangle());

        // One centered instance vs. two instances translated to opposite sides.
        let single = [DrawableObject::Mesh {
            mesh_id: 0,
            model: Matrix4::IDENTITY.to_cols_array(),
            mode: RenderMode::Filled,
        }];
        let single_px = render_with_readback(&device, &queue, format, width, height, |q, e, v| {
            mesh.encode(
                q,
                e,
                v,
                FrameParams::IDENTITY,
                &single,
                Viewport { width, height },
            );
        });

        let two = [
            DrawableObject::Mesh {
                mesh_id: 0,
                model: Matrix4::from_translation(Vector3::new(-0.4, 0.0, 0.0)).to_cols_array(),
                mode: RenderMode::Filled,
            },
            DrawableObject::Mesh {
                mesh_id: 0,
                model: Matrix4::from_translation(Vector3::new(0.4, 0.0, 0.0)).to_cols_array(),
                mode: RenderMode::Filled,
            },
        ];
        let two_px = render_with_readback(&device, &queue, format, width, height, |q, e, v| {
            mesh.encode(
                q,
                e,
                v,
                FrameParams::IDENTITY,
                &two,
                Viewport { width, height },
            );
        });

        assert_ne!(
            single_px, two_px,
            "two translated instances must differ from one centered instance"
        );

        // The two-instance frame must have colored pixels in both the left and
        // right thirds of the image (one triangle each).
        let has_color_in = |xs: std::ops::Range<u32>| {
            xs.into_iter().any(|x| {
                (0..height).any(|y| {
                    let i = ((y * width + x) * 4) as usize;
                    two_px[i] > 0 || two_px[i + 1] > 0 || two_px[i + 2] > 0
                })
            })
        };
        assert!(has_color_in(0..width / 3), "left instance is missing");
        assert!(
            has_color_in(2 * width / 3..width),
            "right instance is missing"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    #[cfg(not(target_arch = "wasm32"))]
    fn mesh_renderer_wireframe_lights_edges_only() {
        let (device, queue) = pollster::block_on(test_device());
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let (width, height) = (64, 64);
        let mut mesh = MeshRenderer::new(&device, format, &Mesh::hello_triangle());
        let model = Matrix4::IDENTITY.to_cols_array();
        let filled_scene = [DrawableObject::Mesh {
            mesh_id: 0,
            model,
            mode: RenderMode::Filled,
        }];
        let wire_scene = [DrawableObject::Mesh {
            mesh_id: 0,
            model,
            mode: RenderMode::Wireframe,
        }];

        let filled = render_with_readback(&device, &queue, format, width, height, |q, e, v| {
            mesh.encode(
                q,
                e,
                v,
                FrameParams::IDENTITY,
                &filled_scene,
                Viewport { width, height },
            );
        });

        let wire = render_with_readback(&device, &queue, format, width, height, |q, e, v| {
            mesh.encode(
                q,
                e,
                v,
                FrameParams::IDENTITY,
                &wire_scene,
                Viewport { width, height },
            );
        });

        assert_ne!(filled, wire, "wireframe must differ from filled");

        let lit = |px: &[u8]| -> usize {
            (0..(width * height) as usize)
                .filter(|i| {
                    let b = i * 4;
                    px[b] > 0 || px[b + 1] > 0 || px[b + 2] > 0
                })
                .count()
        };
        let (filled_lit, wire_lit) = (lit(&filled), lit(&wire));
        assert!(wire_lit > 0, "wireframe must light its edges");
        assert!(
            wire_lit < filled_lit,
            "wireframe ({wire_lit}) must light fewer pixels than filled ({filled_lit})"
        );

        // The triangle's centroid is interior: filled there, background in
        // wireframe (no edge crosses the center of mass).
        let centroid = {
            let v = &Mesh::hello_triangle().vertices;
            let cx = (v[0].position[0] + v[1].position[0] + v[2].position[0]) / 3.0;
            let cy = (v[0].position[1] + v[1].position[1] + v[2].position[1]) / 3.0;
            // NDC (clip, y-up) -> pixel (y-down).
            let px = ((cx * 0.5 + 0.5) * width as f32).round() as u32;
            let py = ((1.0 - (cy * 0.5 + 0.5)) * height as f32).round() as u32;
            ((py.min(height - 1) * width + px.min(width - 1)) * 4) as usize
        };
        assert!(
            filled[centroid] > 0 || filled[centroid + 1] > 0 || filled[centroid + 2] > 0,
            "filled centroid must be lit"
        );
        assert_eq!(
            (wire[centroid], wire[centroid + 1], wire[centroid + 2]),
            (0, 0, 0),
            "wireframe centroid must be background"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    #[cfg(not(target_arch = "wasm32"))]
    fn mesh_renderer_aabb_overlay_draws_green_box() {
        let (device, queue) = pollster::block_on(test_device());
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let (width, height) = (64, 64);
        let mut mesh = MeshRenderer::new(&device, format, &Mesh::hello_triangle());
        let model = Matrix4::IDENTITY.to_cols_array();
        let plain_scene = [DrawableObject::Mesh {
            mesh_id: 0,
            model,
            mode: RenderMode::Filled,
        }];
        let box_scene = [
            DrawableObject::Mesh {
                mesh_id: 0,
                model,
                mode: RenderMode::Filled,
            },
            DrawableObject::AabbBox { mesh_id: 0, model },
        ];

        let plain = render_with_readback(&device, &queue, format, width, height, |q, e, v| {
            mesh.encode(
                q,
                e,
                v,
                FrameParams::IDENTITY,
                &plain_scene,
                Viewport { width, height },
            );
        });

        let with_box = render_with_readback(&device, &queue, format, width, height, |q, e, v| {
            mesh.encode(
                q,
                e,
                v,
                FrameParams::IDENTITY,
                &box_scene,
                Viewport { width, height },
            );
        });

        assert_ne!(plain, with_box, "AABB overlay must change the image");

        // The overlay must light pure-green pixels (R≈0, G>0, B≈0) that are not
        // present without it — the box drawn in AABB_COLOR = [0, 1, 0].
        let pure_green = |px: &[u8]| -> usize {
            (0..(width * height) as usize)
                .filter(|i| {
                    let b = i * 4;
                    px[b] == 0 && px[b + 1] > 0 && px[b + 2] == 0
                })
                .count()
        };
        assert_eq!(
            pure_green(&plain),
            0,
            "no green box expected without the overlay"
        );
        assert!(
            pure_green(&with_box) > 0,
            "AABB overlay must light green box pixels"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    #[cfg(not(target_arch = "wasm32"))]
    fn mesh_renderer_axes_overlay_draws_rgb_gizmo() {
        let (device, queue) = pollster::block_on(test_device());
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let (width, height) = (64, 64);
        let mut mesh = MeshRenderer::new(&device, format, &Mesh::hello_triangle());
        let model = Matrix4::IDENTITY.to_cols_array();
        let plain_scene = [DrawableObject::Mesh {
            mesh_id: 0,
            model,
            mode: RenderMode::Filled,
        }];
        let axes_scene = [
            DrawableObject::Mesh {
                mesh_id: 0,
                model,
                mode: RenderMode::Filled,
            },
            DrawableObject::CoordinateAxes {
                model: Matrix4::IDENTITY.to_cols_array(),
            },
        ];

        let plain = render_with_readback(&device, &queue, format, width, height, |q, e, v| {
            mesh.encode(
                q,
                e,
                v,
                FrameParams::IDENTITY,
                &plain_scene,
                Viewport { width, height },
            );
        });

        let with_axes = render_with_readback(&device, &queue, format, width, height, |q, e, v| {
            mesh.encode(
                q,
                e,
                v,
                FrameParams::IDENTITY,
                &axes_scene,
                Viewport { width, height },
            );
        });

        assert_ne!(plain, with_axes, "axes overlay must change the image");

        // Under the identity camera the +X axis draws a pure-red horizontal line
        // and the +Y axis a pure-green vertical line from the center; both must
        // add colored pixels beyond whatever the filled triangle already lit.
        let count = |px: &[u8], pred: fn(u8, u8, u8) -> bool| -> usize {
            (0..(width * height) as usize)
                .filter(|i| {
                    let b = i * 4;
                    pred(px[b], px[b + 1], px[b + 2])
                })
                .count()
        };
        let pure_red = |r: u8, g: u8, b: u8| r > 0 && g == 0 && b == 0;
        let pure_green = |r: u8, g: u8, b: u8| r == 0 && g > 0 && b == 0;

        assert!(
            count(&with_axes, pure_red) > count(&plain, pure_red),
            "X axis must add pure-red pixels"
        );
        assert!(
            count(&with_axes, pure_green) > count(&plain, pure_green),
            "Y axis must add pure-green pixels"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    #[cfg(not(target_arch = "wasm32"))]
    fn scene_composes_all_drawable_kinds_together() {
        // #41: every primitive is a `DrawableObject`, and the renderer walks a
        // single heterogeneous `Scene` with no per-type branching. A scene mixing
        // a filled mesh, a wireframe mesh, an AABB box, and the axes gizmo must
        // render all of them at once — the filled mesh alone lights fewer pixels
        // than the full composed scene, and the green box + RGB axes appear.
        let (device, queue) = pollster::block_on(test_device());
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let (width, height) = (64, 64);
        let mut mesh = MeshRenderer::new(&device, format, &Mesh::hello_triangle());
        let model = Matrix4::IDENTITY.to_cols_array();

        let filled_only = [DrawableObject::Mesh {
            mesh_id: 0,
            model,
            mode: RenderMode::Filled,
        }];
        let composed = [
            DrawableObject::Mesh {
                mesh_id: 0,
                model,
                mode: RenderMode::Filled,
            },
            DrawableObject::Mesh {
                mesh_id: 0,
                model,
                mode: RenderMode::Wireframe,
            },
            DrawableObject::AabbBox { mesh_id: 0, model },
            DrawableObject::CoordinateAxes {
                model: Matrix4::IDENTITY.to_cols_array(),
            },
        ];

        let filled_px = render_with_readback(&device, &queue, format, width, height, |q, e, v| {
            mesh.encode(
                q,
                e,
                v,
                FrameParams::IDENTITY,
                &filled_only,
                Viewport { width, height },
            );
        });
        let composed_px =
            render_with_readback(&device, &queue, format, width, height, |q, e, v| {
                mesh.encode(
                    q,
                    e,
                    v,
                    FrameParams::IDENTITY,
                    &composed,
                    Viewport { width, height },
                );
            });

        assert_ne!(
            filled_px, composed_px,
            "the composed scene must differ from the filled mesh alone"
        );

        // The AABB box (pure green) and the axes gizmo (pure red +X line) are
        // both present only in the composed scene.
        let count = |px: &[u8], pred: fn(u8, u8, u8) -> bool| -> usize {
            (0..(width * height) as usize)
                .filter(|i| {
                    let b = i * 4;
                    pred(px[b], px[b + 1], px[b + 2])
                })
                .count()
        };
        let pure_green = |r: u8, g: u8, b: u8| r == 0 && g > 0 && b == 0;
        let pure_red = |r: u8, g: u8, b: u8| r > 0 && g == 0 && b == 0;
        assert!(
            count(&composed_px, pure_green) > 0,
            "AABB box must light green pixels in the composed scene"
        );
        assert!(
            count(&composed_px, pure_red) > count(&filled_px, pure_red),
            "axes gizmo must add pure-red pixels in the composed scene"
        );
    }

    /// A unit quad centered at the origin in the z=0 plane, spanning
    /// `[-0.5, 0.5]²`. Used to render a *loaded* mesh (not the baked triangle).
    #[cfg(not(target_arch = "wasm32"))]
    const QUAD_OBJ: &str = "\
v -0.5 -0.5 0.0
v 0.5 -0.5 0.0
v 0.5 0.5 0.0
v -0.5 0.5 0.0
f 1 2 3 4
";

    // A unit cube centered at the origin (±0.5) — a mesh with real depth extent,
    // used by the dolly-turntable scenario test to make near/far framing matter.
    const CUBE_OBJ: &str = "\
v -0.5 -0.5 -0.5
v 0.5 -0.5 -0.5
v 0.5 0.5 -0.5
v -0.5 0.5 -0.5
v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v 0.5 0.5 0.5
v -0.5 0.5 0.5
f 1 2 3 4
f 5 6 7 8
f 1 5 8 4
f 2 6 7 3
f 4 3 7 8
f 1 2 6 5
";

    #[test]
    #[ignore = "requires a GPU adapter"]
    #[cfg(not(target_arch = "wasm32"))]
    fn mesh_renderer_renders_loaded_quad_filled_with_correct_coverage() {
        // #37/#41: a mesh loaded from OBJ (not the baked triangle) renders filled
        // via `draw_indexed` as a `DrawableObject::Mesh`. Under the identity camera
        // the unit quad spans NDC [-0.5, 0.5]², i.e. the central quarter of the
        // frame — so the center is lit, the corners are dark, and coverage ≈ 25%.
        let (device, queue) = pollster::block_on(test_device());
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let (width, height) = (64, 64);
        let quad = Mesh::from_obj(QUAD_OBJ).expect("quad OBJ parses");
        let mut mesh = MeshRenderer::new(&device, format, &quad);

        let scene = [DrawableObject::Mesh {
            mesh_id: 0,
            model: Matrix4::IDENTITY.to_cols_array(),
            mode: RenderMode::Filled,
        }];
        let px = render_with_readback(&device, &queue, format, width, height, |q, e, v| {
            mesh.encode(
                q,
                e,
                v,
                FrameParams::IDENTITY,
                &scene,
                Viewport { width, height },
            );
        });

        let (w, h) = (width as usize, height as usize);
        let covered = |x: usize, y: usize| -> bool {
            let b = (y * w + x) * 4;
            px[b] > 0 || px[b + 1] > 0 || px[b + 2] > 0
        };
        assert!(covered(w / 2, h / 2), "quad center must be covered");
        assert!(!covered(1, 1), "top-left corner must be outside the quad");
        assert!(
            !covered(w - 2, 1),
            "top-right corner must be outside the quad"
        );
        assert!(
            !covered(1, h - 2),
            "bottom-left corner must be outside the quad"
        );
        assert!(
            !covered(w - 2, h - 2),
            "bottom-right corner must be outside the quad"
        );

        let covered_count = (0..w * h)
            .filter(|i| {
                let b = i * 4;
                px[b] > 0 || px[b + 1] > 0 || px[b + 2] > 0
            })
            .count();
        let frac = covered_count as f32 / (w * h) as f32;
        assert!(
            (0.18..=0.32).contains(&frac),
            "quad coverage {frac} is not ≈ the central quarter (0.25)"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    #[cfg(not(target_arch = "wasm32"))]
    fn cg_and_cv_cameras_render_matching_output() {
        // #43/#49: a CG-authored camera (eye/target/up/fovy) and its CV-lowered
        // equivalent (pose = world-from-camera, K = intrinsics) describe the *same*
        // camera, so rendering the same `Scene` under each yields matching pixels.
        let (device, queue) = pollster::block_on(test_device());
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let (width, height) = (96, 96);
        let viewport = Viewport { width, height };
        let quad = Mesh::from_obj(QUAD_OBJ).expect("quad OBJ parses");
        let mut mesh = MeshRenderer::new(&device, format, &quad);

        let scene = [DrawableObject::Mesh {
            mesh_id: 0,
            model: Matrix4::IDENTITY.to_cols_array(),
            mode: RenderMode::Filled,
        }];

        // An off-axis camera so orientation actually matters.
        let eye_arr = [0.6f32, 0.4, 1.4];
        let target_arr = [0.0f32, 0.0, 0.0];
        let up_arr = [0.0f32, 1.0, 0.0];
        let fovy = crate::DEFAULT_FOV_Y;

        let cg = FrameParams {
            eye: Some(eye_arr),
            target: Some(target_arr),
            up: Some(up_arr),
            fovy: Some(fovy),
            ..FrameParams::IDENTITY
        };

        // Lower the same camera to CV form (K + pose) via the camera API.
        let cam = crate::Camera::look_at(
            Point3::new(eye_arr[0], eye_arr[1], eye_arr[2]),
            Point3::new(target_arr[0], target_arr[1], target_arr[2]),
            Vector3::new(up_arr[0], up_arr[1], up_arr[2]),
            fovy,
            viewport,
        );
        let cv = FrameParams {
            pose: Some(cam.to_pose().matrix().to_cols_array()),
            k: Some(cam.to_intrinsics()),
            ..FrameParams::IDENTITY
        };

        let cg_px = render_with_readback(&device, &queue, format, width, height, |q, e, v| {
            mesh.encode(q, e, v, cg, &scene, viewport);
        });
        let cv_px = render_with_readback(&device, &queue, format, width, height, |q, e, v| {
            mesh.encode(q, e, v, cv, &scene, viewport);
        });

        // Both must actually show the quad (non-trivial coverage).
        let lit = |px: &[u8]| {
            px.chunks_exact(4)
                .filter(|c| c[0] > 0 || c[1] > 0 || c[2] > 0)
                .count()
        };
        assert!(lit(&cg_px) > 200, "CG camera must render the quad");
        assert!(lit(&cv_px) > 200, "CV camera must render the quad");

        // ...and their outputs must match within a tiny tolerance (a few edge
        // pixels may differ by rounding in the K⇄projection round-trip).
        let differing = cg_px
            .chunks_exact(4)
            .zip(cv_px.chunks_exact(4))
            .filter(|(a, b)| {
                a.iter()
                    .zip(b.iter())
                    .any(|(x, y)| (i16::from(*x) - i16::from(*y)).abs() > 2)
            })
            .count();
        let frac = differing as f32 / (width * height) as f32;
        assert!(
            frac < 0.01,
            "CG and CV renders differ in {differing} px (fraction {frac})"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    #[cfg(not(target_arch = "wasm32"))]
    fn dolly_turntable_bird_eye_cg_cv_wireframe_stays_framed() {
        // #49 scenario end-to-end: a fixed 45° bird's-eye camera dollies in and
        // out while a mesh spins about +Y, rendered as a **wireframe**. At every
        // (dolly distance, spin angle) this asserts the three defining behaviors
        // of the slice:
        //   (a) the CG-authored camera (eye/target/up/fovy) and its CV-lowered
        //       equivalent (pose + K) render matching pixels;
        //   (b) near/far fit: the spinning mesh stays fully framed — visible
        //       wireframe, empty frame border (nothing clipped at any distance);
        //   (c) the dolly actually reframes: dollying in covers more pixels than
        //       dollying out.
        // (A cube stands in for the bunny for a fast, deterministic GPU test; the
        // same scenario is exercised on the real bunny by examples/bunny_dolly.py
        // + `render.sh --wireframe`.)
        let (device, queue) = pollster::block_on(test_device());
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let (width, height) = (128, 128);
        let viewport = Viewport { width, height };
        let cube = Mesh::from_obj(CUBE_OBJ).expect("cube OBJ parses");
        let mut mesh = MeshRenderer::new(&device, format, &cube);

        let fovy = crate::DEFAULT_FOV_Y; // 45°
                                         // Fixed bird's-eye view direction: 45° elevation, 35° azimuth (unit).
        let elev = 45f32.to_radians();
        let azim = 35f32.to_radians();
        let view_dir =
            Vec3::new(elev.cos() * azim.sin(), elev.sin(), elev.cos() * azim.cos()).normalize();
        let target = Point3::new(0.0, 0.0, 0.0);
        let up = Vector3::Y;

        // Dolly-in (near) → mid → dolly-out (far).
        let distances = [3.5f32, 4.75, 6.0];
        // Turntable spin angles about +Y.
        let angles = [0.0f32, std::f32::consts::FRAC_PI_2, 2.4];

        let lit = |px: &[u8]| -> usize {
            px.chunks_exact(4)
                .filter(|c| c[0] > 0 || c[1] > 0 || c[2] > 0)
                .count()
        };
        // Lit pixels in the outer 2-px ring — must stay 0 (mesh never clipped).
        let border_lit = |px: &[u8]| -> usize {
            let w = width as usize;
            let h = height as usize;
            let mut n = 0;
            for y in 0..h {
                for x in 0..w {
                    if x < 2 || x >= w - 2 || y < 2 || y >= h - 2 {
                        let b = (y * w + x) * 4;
                        if px[b] > 0 || px[b + 1] > 0 || px[b + 2] > 0 {
                            n += 1;
                        }
                    }
                }
            }
            n
        };

        let mut lit_at_zero_spin = Vec::new();
        for &dist in &distances {
            let eye_arr = [view_dir.x * dist, view_dir.y * dist, view_dir.z * dist];
            let eye = Point3::new(eye_arr[0], eye_arr[1], eye_arr[2]);
            // Lower the same camera to CV form (K + pose) once per distance.
            let cam = crate::Camera::look_at(eye, target, up, fovy, viewport);
            let pose = cam.to_pose().matrix().to_cols_array();
            let k = cam.to_intrinsics();

            for &theta in &angles {
                let scene = [DrawableObject::Mesh {
                    mesh_id: 0,
                    model: Mat4::from_rotation_y(theta).to_cols_array(),
                    mode: RenderMode::Wireframe,
                }];

                let cg = FrameParams {
                    eye: Some(eye_arr),
                    target: Some([0.0, 0.0, 0.0]),
                    up: Some([0.0, 1.0, 0.0]),
                    fovy: Some(fovy),
                    ..FrameParams::IDENTITY
                };
                let cv = FrameParams {
                    pose: Some(pose),
                    k: Some(k),
                    ..FrameParams::IDENTITY
                };

                let cg_px =
                    render_with_readback(&device, &queue, format, width, height, |q, e, v| {
                        mesh.encode(q, e, v, cg, &scene, viewport);
                    });
                let cv_px =
                    render_with_readback(&device, &queue, format, width, height, |q, e, v| {
                        mesh.encode(q, e, v, cv, &scene, viewport);
                    });

                let cg_lit = lit(&cg_px);
                // (b) near/far fit: visible wireframe, but framed — never fills
                // the frame and never touches the border (nothing clipped).
                assert!(
                    cg_lit > 20,
                    "dist {dist} theta {theta}: wireframe must be visible (near/far fit)"
                );
                assert!(
                    (cg_lit as f32) < 0.5 * (width * height) as f32,
                    "dist {dist} theta {theta}: mesh must stay framed, not overflow ({cg_lit} px)"
                );
                assert_eq!(
                    border_lit(&cg_px),
                    0,
                    "dist {dist} theta {theta}: mesh must not touch the frame border (stays framed)"
                );

                // (a) CG and CV forms render matching pixels (few edge pixels may
                // differ by rounding in the K⇄projection round-trip).
                let differing = cg_px
                    .chunks_exact(4)
                    .zip(cv_px.chunks_exact(4))
                    .filter(|(a, b)| {
                        a.iter()
                            .zip(b.iter())
                            .any(|(x, y)| (i16::from(*x) - i16::from(*y)).abs() > 2)
                    })
                    .count();
                let frac = differing as f32 / (width * height) as f32;
                assert!(
                    frac < 0.02,
                    "dist {dist} theta {theta}: CG vs CV differ in {differing} px ({frac})"
                );

                if theta == angles[0] {
                    lit_at_zero_spin.push((dist, cg_lit));
                }
            }
        }

        // (c) the dolly reframes the mesh: closer distance ⇒ larger footprint.
        for pair in lit_at_zero_spin.windows(2) {
            let (near_d, near_lit) = pair[0];
            let (far_d, far_lit) = pair[1];
            assert!(
                near_lit > far_lit,
                "dolly-in ({near_d}, {near_lit}px) must cover more than dolly-out ({far_d}, {far_lit}px)"
            );
        }
    }
}
