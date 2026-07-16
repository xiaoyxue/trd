//! Shared, platform-agnostic parametric triangle rendering.
//!
//! [`render_triangle`] draws the hello-triangle transformed by [`FrameParams`]
//! into the given texture view. Both the native batch renderer and the wasm
//! entry point build on [`create_triangle_pipeline`].

use glam::{Mat4, Vec3};

/// Default clip near/far planes used when deriving a projection from camera
/// intrinsics `K`. The hello-triangle is authored on the `z = 0` plane, so the
/// exact values only need to bracket it; they are renderer constants until the
/// camera slice (#18) makes them configurable.
const DEFAULT_NEAR: f32 = 0.1;
const DEFAULT_FAR: f32 = 1000.0;

/// Per-frame transform parameters for the triangle.
///
/// The base triangle vertices `p_i` are transformed by the full MVP chain
/// `clip = P · V · M · (p_i, 0, 1)` in the vertex shader, where:
/// - **M** (model) is [`FrameParams::model`] if present, else synthesized from
///   the 2D affine `center`/`size`/`theta` as
///   `translate(center) · rotate_z(theta) · scale(size)` (reproducing the
///   original `p' = center + R(theta) · (size ⊙ p_i)`).
/// - **V** (view) is `inverse(pose)` when [`FrameParams::pose`] (a
///   world-from-camera transform) is present, else identity.
/// - **P** (projection) is derived from camera intrinsics [`FrameParams::k`] and
///   the render target's viewport, else identity.
///
/// With no `model`/`k`/`pose` columns, `P = V = I` and `M` is the 2D affine, so
/// the output is byte-for-byte the protocol `0.0.1` result.
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
    /// Optional camera **intrinsics** `K`: a 3×3 pinhole matrix, column-major
    /// (9 floats). `Some` derives the projection; `None` uses identity.
    pub k: Option<[f32; 9]>,
    /// Optional camera **pose** (world-from-camera): a 4×4 matrix, column-major
    /// (16 floats). The view matrix is its inverse; `None` uses identity.
    pub pose: Option<[f32; 16]>,
}

/// The render target's pixel dimensions, needed to turn pixel-space camera
/// intrinsics `K` into a clip-space projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Viewport {
    pub width: u32,
    pub height: u32,
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
    };

    /// The effective 4×4 model matrix: the explicit [`FrameParams::model`] if
    /// present, else the 2D affine synthesized from `center`/`size`/`theta`.
    pub(crate) fn model_matrix(&self) -> Mat4 {
        match self.model {
            Some(cols) => Mat4::from_cols_array(&cols),
            None => model_from_2d_affine(self.center, self.size, self.theta),
        }
    }

    /// The view matrix `camera-from-world = inverse(pose)`, or identity.
    pub(crate) fn view_matrix(&self) -> Mat4 {
        match self.pose {
            Some(cols) => Mat4::from_cols_array(&cols).inverse(),
            None => Mat4::IDENTITY,
        }
    }

    /// The projection matrix derived from intrinsics `K` and the viewport, or
    /// identity when no intrinsics are supplied.
    pub(crate) fn projection_matrix(&self, viewport: Viewport) -> Mat4 {
        match self.k {
            Some(k) => projection_from_intrinsics(k, viewport),
            None => Mat4::IDENTITY,
        }
    }

    /// The full clip transform `P · V · M` for a given viewport.
    pub(crate) fn clip_transform(&self, viewport: Viewport) -> Mat4 {
        self.projection_matrix(viewport) * self.view_matrix() * self.model_matrix()
    }
}

/// Builds the 2D-affine model matrix `translate(center) · rotate_z(theta) ·
/// scale(size)` (z untouched), the `0.0.1` transform expressed as a `Mat4`.
pub(crate) fn model_from_2d_affine(center: [f32; 2], size: [f32; 2], theta: f32) -> Mat4 {
    Mat4::from_translation(Vec3::new(center[0], center[1], 0.0))
        * Mat4::from_rotation_z(theta)
        * Mat4::from_scale(Vec3::new(size[0], size[1], 1.0))
}

/// Builds a right-handed, wgpu-clip-space (`z ∈ [0, 1]`) perspective projection
/// from a pinhole intrinsics matrix `K` (column-major: `fx = k[0]`, `fy = k[4]`,
/// `cx = k[6]`, `cy = k[7]`) and the target viewport.
///
/// Conventions (to be validated visually / refined in the camera slice #18):
/// `K` shares NDC orientation (x right, y up, camera looking down `-z`); a
/// centered principal point (`cx = W/2`, `cy = H/2`) with square pixels reduces
/// to [`glam::Mat4::perspective_rh`]. `near`/`far` are [`DEFAULT_NEAR`]/
/// [`DEFAULT_FAR`].
pub(crate) fn projection_from_intrinsics(k: [f32; 9], viewport: Viewport) -> Mat4 {
    let fx = k[0];
    let fy = k[4];
    let cx = k[6];
    let cy = k[7];
    let w = viewport.width.max(1) as f32;
    let h = viewport.height.max(1) as f32;
    let (n, f) = (DEFAULT_NEAR, DEFAULT_FAR);

    // Column-major: each row below is one column of the matrix.
    Mat4::from_cols_array(&[
        2.0 * fx / w,
        0.0,
        0.0,
        0.0,
        0.0,
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
/// clip transform (`P · V · M`, 64 bytes).
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

/// Writes `params` (projected for `viewport`) into an existing uniform buffer.
///
/// Shared by the native `BatchRenderer` and the persistent [`TriangleRenderer`],
/// which reuse one uniform buffer across frames instead of rebuilding it.
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

#[cfg(test)]
mod tests {
    use super::*;

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
            Mat4::IDENTITY.to_cols_array()
        );
    }

    #[test]
    fn identity_params_produce_identity_model() {
        assert_eq!(FrameParams::IDENTITY.model_matrix(), Mat4::IDENTITY);
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
        assert_eq!(params.model_matrix(), Mat4::from_cols_array(&cols));
    }

    #[test]
    fn synthesized_model_reproduces_2d_affine_transform() {
        // The synthesized model must map base vertices exactly like the legacy
        // `p' = center + R(theta) * (size ⊙ p)` formula.
        let center = [0.1_f32, -0.2];
        let size = [0.5_f32, 0.75];
        let theta = 1.25_f32;
        let model = model_from_2d_affine(center, size, theta);

        for base in [[0.0_f32, 0.5], [-0.5, -0.5], [0.5, -0.5]] {
            let scaled = [base[0] * size[0], base[1] * size[1]];
            let (s, c) = theta.sin_cos();
            let expected = [
                center[0] + c * scaled[0] - s * scaled[1],
                center[1] + s * scaled[0] + c * scaled[1],
            ];
            let got = model.transform_point3(Vec3::new(base[0], base[1], 0.0));
            assert!(
                (got.x - expected[0]).abs() < 1e-6,
                "x: {got:?} {expected:?}"
            );
            assert!(
                (got.y - expected[1]).abs() < 1e-6,
                "y: {got:?} {expected:?}"
            );
            assert!(got.z.abs() < 1e-6, "z should stay 0: {got:?}");
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
        assert!(params.view_matrix().abs_diff_eq(pose.inverse(), 1e-5));
        // No pose => identity view.
        assert_eq!(FrameParams::IDENTITY.view_matrix(), Mat4::IDENTITY);
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
        assert!(got.abs_diff_eq(expected, 1e-4), "{got:?} vs {expected:?}");
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
        let clip = p * glam::Vec4::new(0.0, 0.0, -5.0, 1.0);
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
        assert!(params
            .clip_transform(viewport)
            .abs_diff_eq(params.model_matrix(), 1e-6));
    }

    #[test]
    fn math_transform_reproduces_2d_affine_model() {
        // The typed `math::Transform` API must be able to rebuild the legacy
        // `translate · rotate_z · scale` model matrix that drives the GPU
        // uniform, guarding the future render.rs migration onto `math`.
        use crate::{Rotation, Transform, Vector3};

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
        assert!(
            Mat4::from_cols_array(&t.to_cols_array()).abs_diff_eq(expected, 1e-6),
            "{:?} vs {expected:?}",
            t.to_cols_array()
        );
    }
}
