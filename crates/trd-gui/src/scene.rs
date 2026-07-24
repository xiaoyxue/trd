//! [`SceneState`] — the interactive scene the GUI authors and re-renders (#97).
//!
//! This is the shared, platform-agnostic model the interaction loop mutates:
//! an **orbit camera** (the CG `eye`/`target`/`fovy` form) plus a single
//! **object transform** (the per-draw model matrix), together with the render
//! mode and overlay flags. It carries no GPU or egui state — it only produces
//! the two values `trd-core` consumes each frame: a [`FrameParams`] camera and a
//! list of [`Draw`]s. The render backend ([`crate::render_backend`]) turns those
//! into pixels; the interaction controller ([`crate::interaction`]) mutates this
//! state from user gestures.
//!
//! Conventions follow `trd-core::math`: right-handed world, `+Y` up, radians.
//! The loaded mesh is centered and scaled to fit by
//! [`trd_core::Mesh::preview_transform`] beneath the draw model, so the scene is
//! authored around the world origin — the camera targets the origin by default.

use trd_core::{Draw, FrameParams, Point3, RenderMode, Rotation, Transform, Vector3};

/// The minimum orbit distance (never let the camera cross the target).
const MIN_DISTANCE: f32 = 0.2;
/// The maximum orbit distance (keep the framed object on screen).
const MAX_DISTANCE: f32 = 100.0;
/// Clamp the pitch just shy of the poles so `up = +Y` never degenerates.
const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

/// A camera that orbits a target point on a sphere: `yaw`/`pitch` place the eye,
/// `distance` sets the radius, `fovy` the vertical field of view. This is the
/// CG (`eye`/`target`/`fovy`) half of [`FrameParams`], the natural form for an
/// orbit interaction (the object stays put; the camera moves around it).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    /// Azimuth about `+Y`, radians. `0` looks along `-Z` toward the target.
    pub yaw: f32,
    /// Elevation above the `XZ` plane, radians, clamped to `±MAX_PITCH`.
    pub pitch: f32,
    /// Distance from `target` to the eye, clamped to `[MIN_DISTANCE, MAX_DISTANCE]`.
    pub distance: f32,
    /// The world-space point the camera looks at.
    pub target: [f32; 3],
    /// Vertical field of view, radians.
    pub fovy: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.3,
            // The mesh is preview-scaled to a max extent of
            // `DEFAULT_PREVIEW_TARGET` (2.0) about the origin; ~4 units frames it
            // comfortably at a 45° fov.
            distance: 4.0,
            target: [0.0, 0.0, 0.0],
            fovy: trd_core::DEFAULT_FOV_Y,
        }
    }
}

impl OrbitCamera {
    /// The world-space eye position derived from `yaw`/`pitch`/`distance` about
    /// `target`. `yaw = pitch = 0` sits on `+Z` looking toward `-Z`.
    pub fn eye(&self) -> [f32; 3] {
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        let dir = Vector3::new(cp * sy, sp, cp * cy);
        let target = Point3::new(self.target[0], self.target[1], self.target[2]);
        (target + dir * self.distance).to_array()
    }

    /// Rotates the eye about the target by `(dyaw, dpitch)` radians (pitch is
    /// clamped shy of the poles).
    pub fn orbit(&mut self, dyaw: f32, dpitch: f32) {
        self.yaw += dyaw;
        self.pitch = (self.pitch + dpitch).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// Dollies toward (`factor < 1`) or away from (`factor > 1`) the target,
    /// scaling `distance` and clamping it to the working range.
    pub fn dolly(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(MIN_DISTANCE, MAX_DISTANCE);
    }
}

/// A single object's placement: an intrinsic yaw/pitch rotation composed under a
/// world translation. Produces the per-draw **model** matrix `T · R` (rotate,
/// then translate) that `trd-core` applies beneath the mesh's preview base model.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ObjectTransform {
    /// Rotation about the object's `+Y` axis, radians.
    pub yaw: f32,
    /// Rotation about the object's `+X` axis, radians.
    pub pitch: f32,
    /// World-space translation applied after rotation.
    pub translation: [f32; 3],
}

impl ObjectTransform {
    /// The column-major model matrix `T · R`: rotate about `+Y` then `+X`, then
    /// translate. Built with the typed `trd-core` transforms so the affine
    /// composition rule (`a.then(b) == b · a`) holds.
    pub fn model_matrix(&self) -> [f32; 16] {
        let rotation = Rotation::from_rotation_y(self.yaw) * Rotation::from_rotation_x(self.pitch);
        let t = Vector3::new(
            self.translation[0],
            self.translation[1],
            self.translation[2],
        );
        Transform::from_rotation(rotation)
            .then(Transform::from_translation(t))
            .matrix()
            .to_cols_array()
    }

    /// Translates the object by a world-space delta.
    pub fn translate(&mut self, delta: [f32; 3]) {
        self.translation[0] += delta[0];
        self.translation[1] += delta[1];
        self.translation[2] += delta[2];
    }

    /// Rotates the object by `(dyaw, dpitch)` radians.
    pub fn rotate(&mut self, dyaw: f32, dpitch: f32) {
        self.yaw += dyaw;
        self.pitch += dpitch;
    }
}

/// The full interactive scene: the orbit camera, the object placement, the mesh
/// render mode, and the overlay toggles. Rebuilt into a [`FrameParams`] + `draws`
/// each frame the state changes, with **no per-primitive branching** — a single
/// mesh is the degenerate one-draw scene.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneState {
    /// The orbit camera framing the object.
    pub camera: OrbitCamera,
    /// The placement of the (single) drawn object.
    pub object: ObjectTransform,
    /// How the mesh is drawn (filled / wireframe).
    pub mode: RenderMode,
    /// Overlay each drawn mesh instance's axis-aligned bounding box (#42).
    pub show_aabb: bool,
    /// Overlay a world-origin coordinate-axes gizmo (#42).
    pub show_axes: bool,
    /// Overlay a coordinate-axes gizmo at the object's own (model) frame (#77).
    pub show_local_axes: bool,
}

impl Default for SceneState {
    fn default() -> Self {
        Self {
            camera: OrbitCamera::default(),
            object: ObjectTransform::default(),
            mode: RenderMode::Filled,
            show_aabb: false,
            show_axes: false,
            show_local_axes: false,
        }
    }
}

impl SceneState {
    /// The camera [`FrameParams`] for the given viewport `aspect` (width/height):
    /// the CG `eye`/`target`/`up`/`fovy`/`aspect` form derived from the orbit
    /// camera. The model matrix rides on the per-draw list (see [`Self::draws`]),
    /// not here, so this carries only the camera.
    pub fn frame_params(&self, aspect: f32) -> FrameParams {
        let eye = self.camera.eye();
        FrameParams {
            eye: Some(eye),
            target: Some(self.camera.target),
            up: Some([0.0, 1.0, 0.0]),
            fovy: Some(self.camera.fovy),
            aspect: Some(aspect),
            ..FrameParams::IDENTITY
        }
    }

    /// The per-frame draw list: a single instance of mesh `0` placed by the
    /// object's model matrix, inheriting the renderer's global render mode.
    pub fn draws(&self) -> Vec<Draw> {
        vec![Draw {
            mesh_id: 0,
            model: self.object.model_matrix(),
            mode: None,
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trd_core::Matrix4;

    fn approx(a: [f32; 3], b: [f32; 3]) {
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-4, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn eye_at_zero_yaw_pitch_is_on_positive_z() {
        let cam = OrbitCamera {
            yaw: 0.0,
            pitch: 0.0,
            distance: 4.0,
            target: [0.0, 0.0, 0.0],
            fovy: 0.8,
        };
        approx(cam.eye(), [0.0, 0.0, 4.0]);
    }

    #[test]
    fn eye_orbits_around_target() {
        let cam = OrbitCamera {
            yaw: std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
            distance: 3.0,
            target: [1.0, 0.0, 0.0],
            fovy: 0.8,
        };
        // 90° yaw about +Y from +Z lands on +X, offset by the target.
        approx(cam.eye(), [4.0, 0.0, 0.0]);
    }

    #[test]
    fn dolly_clamps_distance() {
        let mut cam = OrbitCamera::default();
        cam.dolly(0.0);
        assert!(cam.distance >= MIN_DISTANCE);
        cam.distance = MAX_DISTANCE;
        cam.dolly(10.0);
        assert!(cam.distance <= MAX_DISTANCE);
    }

    #[test]
    fn orbit_clamps_pitch() {
        let mut cam = OrbitCamera::default();
        cam.orbit(0.0, 100.0);
        assert!(cam.pitch <= MAX_PITCH);
        cam.orbit(0.0, -100.0);
        assert!(cam.pitch >= -MAX_PITCH);
    }

    #[test]
    fn identity_object_model_is_identity() {
        let obj = ObjectTransform::default();
        assert_eq!(obj.model_matrix(), Matrix4::IDENTITY.to_cols_array());
    }

    #[test]
    fn translation_lands_in_the_last_column() {
        let obj = ObjectTransform {
            yaw: 0.0,
            pitch: 0.0,
            translation: [1.0, 2.0, 3.0],
        };
        let m = obj.model_matrix();
        // Column-major: the translation is the 4th column (indices 12..15).
        approx([m[12], m[13], m[14]], [1.0, 2.0, 3.0]);
    }

    #[test]
    fn frame_params_carry_only_the_cg_camera() {
        let state = SceneState::default();
        let params = state.frame_params(1.5);
        assert!(params.eye.is_some());
        assert!(params.target.is_some());
        assert_eq!(params.fovy, Some(trd_core::DEFAULT_FOV_Y));
        assert_eq!(params.aspect, Some(1.5));
        // Camera only: no CV form, no model on the params.
        assert!(params.k.is_none() && params.pose.is_none());
        assert!(params.model.is_none());
    }

    #[test]
    fn draws_place_the_object_model() {
        let mut state = SceneState::default();
        state.object.translation = [0.5, 0.0, 0.0];
        let draws = state.draws();
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].mesh_id, 0);
        assert_eq!(draws[0].model, state.object.model_matrix());
    }
}
