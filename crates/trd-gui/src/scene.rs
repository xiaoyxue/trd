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

use trd_core::{Draw, FrameParams, PbrMaterial, Point3, RenderMode, Rotation, Transform, Vector3};

/// The minimum orbit distance (never let the camera cross the target).
const MIN_DISTANCE: f32 = 0.2;
/// The maximum orbit distance (keep the framed object on screen).
const MAX_DISTANCE: f32 = 100.0;
/// Clamp the pitch just shy of the poles so `up = +Y` never degenerates.
const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
/// The smallest per-axis object scale, so a scale gesture/widget can never
/// collapse the object to a degenerate (zero or negative) size.
pub const MIN_SCALE: f32 = 0.01;
/// The largest per-axis object scale, keeping the framed object on screen.
pub const MAX_SCALE: f32 = 100.0;
/// World-space spacing between adjacent objects in a multi-object scene. The
/// preview transform fits each mesh to a max extent of `DEFAULT_PREVIEW_TARGET`
/// (~2 world units), so `2.6` leaves a small gap between neighbours.
const OBJECT_SPACING: f32 = 2.6;

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

/// A single object's placement: an intrinsic yaw/pitch/roll rotation and a
/// per-axis scale composed under a world translation. Produces the per-draw
/// **model** matrix `T · R · S` (scale, then rotate, then translate) that
/// `trd-core` applies beneath the mesh's preview base model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectTransform {
    /// Rotation about the object's `+Y` axis, radians.
    pub yaw: f32,
    /// Rotation about the object's `+X` axis, radians.
    pub pitch: f32,
    /// Rotation about the object's `+Z` axis, radians.
    pub roll: f32,
    /// Per-axis scale (`x`, `y`, `z`), each clamped to `[MIN_SCALE, MAX_SCALE]`.
    pub scale: [f32; 3],
    /// World-space translation applied after rotation.
    pub translation: [f32; 3],
}

impl Default for ObjectTransform {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            scale: [1.0, 1.0, 1.0],
            translation: [0.0, 0.0, 0.0],
        }
    }
}

impl ObjectTransform {
    /// The column-major model matrix `T · R · S`: scale, then rotate about `+Y`,
    /// `+X`, `+Z`, then translate. Built with the typed `trd-core` transforms so
    /// the affine composition rule (`a.then(b) == b · a`) holds.
    pub fn model_matrix(&self) -> [f32; 16] {
        self.model_matrix_offset([0.0, 0.0, 0.0])
    }

    /// The model matrix `T(translation + offset) · R · S` — the object's own
    /// `T·R·S` shifted by a **world-space** `offset` (applied after rotation, like
    /// the translation). Used to lay out a multi-object scene: each object keeps
    /// its own transform while a per-object layout `offset` spreads them apart.
    pub fn model_matrix_offset(&self, offset: [f32; 3]) -> [f32; 16] {
        let rotation = Rotation::from_rotation_y(self.yaw)
            * Rotation::from_rotation_x(self.pitch)
            * Rotation::from_rotation_z(self.roll);
        let scale = Vector3::new(self.scale[0], self.scale[1], self.scale[2]);
        let translation = Vector3::new(
            self.translation[0] + offset[0],
            self.translation[1] + offset[1],
            self.translation[2] + offset[2],
        );
        Transform::from_scale_rotation_translation(scale, rotation, translation)
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

    /// Rolls the object about its `+Z` axis by `droll` radians.
    pub fn roll_by(&mut self, droll: f32) {
        self.roll += droll;
    }

    /// Multiplies every axis of the scale by `factor` (uniform scale), clamping
    /// each to `[MIN_SCALE, MAX_SCALE]`.
    pub fn scale_uniform(&mut self, factor: f32) {
        for s in &mut self.scale {
            *s = (*s * factor).clamp(MIN_SCALE, MAX_SCALE);
        }
    }
}

/// The full interactive scene: the orbit camera, the object placement, the mesh
/// render mode, and the overlay toggles. Rebuilt into a [`FrameParams`] + `draws`
/// each frame the state changes, with **no per-primitive branching** — a single
/// mesh is the degenerate one-draw scene.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneState {
    /// The orbit camera framing the object(s).
    pub camera: OrbitCamera,
    /// The placement of each drawn object (one per loaded mesh), laid out
    /// side-by-side by [`Self::draws`]. A single-object scene is the degenerate
    /// one-element case. Transforms edit the [`selected`](Self::selected) object.
    pub objects: Vec<ObjectTransform>,
    /// How the mesh is drawn (filled / wireframe / textured / PBR).
    pub mode: RenderMode,
    /// The Disney PBR material of **each** object (parallel to [`objects`](Self::objects)),
    /// applied when [`mode`](Self::mode) is [`RenderMode::Pbr`]. Interactive: the UI
    /// edits the **selected** object's material (metallic/roughness/etc.); the bound
    /// HDR env probe lives on the renderer, set once from `--env` / `?env=`.
    pub materials: Vec<PbrMaterial>,
    /// Overlay each drawn mesh instance's axis-aligned bounding box (#42).
    pub show_aabb: bool,
    /// Overlay a world-origin coordinate-axes gizmo (#42).
    pub show_axes: bool,
    /// Overlay a coordinate-axes gizmo at the object's own (model) frame (#77).
    pub show_local_axes: bool,
    /// Overlay a world-origin XZ **plane grid** (a floor at the world origin).
    pub show_world_grid: bool,
    /// Overlay an XZ **plane grid** at the object's own (model) frame — a grid
    /// that follows the object as it is translated / rotated / scaled.
    pub show_local_grid: bool,
    /// The currently **selected** object as a 0-based index into
    /// [`objects`](Self::objects) / [`Self::draws`] (#141), set by click-to-select
    /// picking; `None` when nothing is selected. The selected object shows its
    /// AABB and is the target of every transform gesture/widget — with no
    /// selection, transforms are disabled.
    pub selected: Option<u32>,
}

impl Default for SceneState {
    fn default() -> Self {
        Self {
            camera: OrbitCamera::default(),
            objects: vec![ObjectTransform::default()],
            mode: RenderMode::Filled,
            materials: vec![PbrMaterial::default()],
            show_aabb: false,
            show_axes: false,
            show_local_axes: false,
            show_world_grid: false,
            show_local_grid: false,
            selected: None,
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

    /// The per-frame draw list: one instance per object in
    /// [`objects`](Self::objects), drawn as mesh `i` (row index) placed by that
    /// object's model matrix **shifted by a per-object layout offset** so multiple
    /// objects spread side-by-side along world `X` (a single object stays at the
    /// origin). Inherits the renderer's global render mode.
    pub fn draws(&self) -> Vec<Draw> {
        let n = self.objects.len();
        self.objects
            .iter()
            .enumerate()
            .map(|(i, obj)| Draw {
                mesh_id: i as u32,
                model: obj.model_matrix_offset(layout_offset(i, n)),
                mode: None,
            })
            .collect()
    }

    /// A shared reference to the selected object's transform, or `None` when
    /// nothing is selected (or the index is out of range).
    pub fn selected_object(&self) -> Option<&ObjectTransform> {
        self.selected.and_then(|i| self.objects.get(i as usize))
    }

    /// A mutable reference to the selected object's transform, or `None` when
    /// nothing is selected — the seam that makes transforms **require a
    /// selection**: with no selection there is nothing to edit.
    pub fn selected_object_mut(&mut self) -> Option<&mut ObjectTransform> {
        self.selected.and_then(|i| self.objects.get_mut(i as usize))
    }

    /// A mutable reference to the selected object's PBR **material**, or `None`
    /// when nothing is selected — so the PBR panel edits the selected object's
    /// material and is disabled otherwise (#141).
    pub fn selected_material_mut(&mut self) -> Option<&mut PbrMaterial> {
        self.selected
            .and_then(|i| self.materials.get_mut(i as usize))
    }

    /// Whether the scene carries a **placement quad** — a given rectangle in E³
    /// (the #77 frame-plane, basis `(e1, e2, e3)`) that a transform could use as a
    /// *quad* coordinate frame. The interactive gui authors object(s) around the
    /// world origin with **no** quad, so this is always `false` for now and the
    /// UI must hide every quad-frame affordance (#140). A future slice that
    /// introduces a quad source into the scene flips this to gate the quad frame on.
    pub fn has_quad(&self) -> bool {
        false
    }
}

/// The world-space layout offset for object `i` of `n`: objects are spread along
/// `X`, centered about the origin, [`OBJECT_SPACING`] apart (so a single object
/// stays at the origin — `offset = 0`).
fn layout_offset(i: usize, n: usize) -> [f32; 3] {
    let centered = i as f32 - (n.saturating_sub(1) as f32) / 2.0;
    [centered * OBJECT_SPACING, 0.0, 0.0]
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
            translation: [1.0, 2.0, 3.0],
            ..ObjectTransform::default()
        };
        let m = obj.model_matrix();
        // Column-major: the translation is the 4th column (indices 12..15).
        approx([m[12], m[13], m[14]], [1.0, 2.0, 3.0]);
    }

    #[test]
    fn default_object_scale_is_unit() {
        let obj = ObjectTransform::default();
        assert_eq!(obj.scale, [1.0, 1.0, 1.0]);
        assert_eq!(obj.roll, 0.0);
        // Unit scale + zero rotation/translation ⇒ identity model.
        assert_eq!(obj.model_matrix(), Matrix4::IDENTITY.to_cols_array());
    }

    #[test]
    fn scale_writes_the_diagonal_of_the_model() {
        let obj = ObjectTransform {
            scale: [2.0, 3.0, 4.0],
            ..ObjectTransform::default()
        };
        let m = obj.model_matrix();
        // No rotation ⇒ T·R·S is diagonal in its upper-left 3×3 (column-major).
        approx([m[0], m[5], m[10]], [2.0, 3.0, 4.0]);
    }

    #[test]
    fn scale_uniform_clamps_to_the_working_range() {
        let mut obj = ObjectTransform::default();
        obj.scale_uniform(0.0);
        for s in obj.scale {
            assert!(s >= MIN_SCALE);
        }
        obj.scale = [MAX_SCALE, MAX_SCALE, MAX_SCALE];
        obj.scale_uniform(10.0);
        for s in obj.scale {
            assert!(s <= MAX_SCALE);
        }
    }

    #[test]
    fn roll_rotates_about_z() {
        let mut obj = ObjectTransform::default();
        obj.roll_by(std::f32::consts::FRAC_PI_2);
        // +90° roll about +Z maps object +X to +Y (column 0 of the model).
        let m = obj.model_matrix();
        approx([m[0], m[1], m[2]], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn gui_scene_has_no_quad() {
        assert!(!SceneState::default().has_quad());
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
        state.objects[0].translation = [0.5, 0.0, 0.0];
        let draws = state.draws();
        // A single object stays at the origin (layout offset 0).
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].mesh_id, 0);
        assert_eq!(draws[0].model, state.objects[0].model_matrix());
    }

    #[test]
    fn multi_object_draws_spread_along_x() {
        let state = SceneState {
            objects: vec![ObjectTransform::default(); 3],
            ..SceneState::default()
        };
        let draws = state.draws();
        assert_eq!(draws.len(), 3);
        // mesh_id follows the object index; world X (model col 3, index 12) is
        // centered and increasing left→right.
        let x = |d: &Draw| d.model[12];
        assert_eq!(
            (draws[0].mesh_id, draws[1].mesh_id, draws[2].mesh_id),
            (0, 1, 2)
        );
        assert!(x(&draws[0]) < 0.0 && x(&draws[2]) > 0.0);
        assert!((x(&draws[1])).abs() < 1e-6, "middle object centered at x≈0");
    }

    #[test]
    fn selected_object_mut_requires_a_selection() {
        let mut state = SceneState::default();
        assert!(state.selected_object_mut().is_none(), "no selection → None");
        state.selected = Some(0);
        assert!(state.selected_object_mut().is_some());
        // Out-of-range selection is also None (never panics).
        state.selected = Some(5);
        assert!(state.selected_object_mut().is_none());
    }
}
