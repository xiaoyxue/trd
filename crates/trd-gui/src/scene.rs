//! [`SceneState`] — the interactive scene the GUI authors and re-renders (#97).
//!
//! The shared, platform-agnostic model the interaction loop mutates: an orbit
//! camera plus a single object transform, the render mode and overlay flags. It
//! holds no GPU or egui state — it produces the two values `trd-core` consumes
//! each frame, a [`FrameParams`](trd_core::FrameParams) camera and a [`Draw`] list.
//!
//! Conventions follow `trd-core::math`: right-handed world, `+Y` up, radians; the
//! mesh is preview-scaled by [`trd_core::Mesh::preview_transform`] about the origin.

use trd_core::{
    Camera, DisneyMaterial, Draw, DrawSelection, ImageBasedLighting, Lighting, Matrix4,
    PbrDebugView, Point3, RenderMode, Rotation, ToneMapping, Transform, Vector3, Viewport,
};

/// Orbit limits: never cross the target, never reach the poles (`up = +Y` would
/// degenerate), keep the framed object on screen.
const MIN_DISTANCE: f32 = 0.2;
const MAX_DISTANCE: f32 = 100.0;
const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

/// Scale limits: a gesture can never collapse an object to a degenerate size.
pub const MIN_SCALE: f32 = 0.01;
pub const MAX_SCALE: f32 = 100.0;

/// World-space spacing between neighbours in a multi-object scene. Each mesh is
/// preview-fitted to ~2 world units, so this leaves a small gap.
const OBJECT_SPACING: f32 = 2.6;

/// The rig a runtime-loaded model is lit by: **image-based lighting only** — no
/// direct light and no ambient, so every photon comes from the HDR probe (#353).
///
/// This is exactly the video editor's `CatalogAsset::Dragon` rig, which is what
/// makes a GLB dropped into the viewer look the way the same GLB looks in the
/// editor. It relies on a probe being bound: with none, an IBL-only scene is
/// black.
pub fn ibl_only_lighting() -> Lighting {
    Lighting {
        ambient: 0.0,
        scale: 0.0,
        ..Lighting::default()
    }
}

/// A camera that orbits a target point on a sphere — the CG
/// (`eye`/`target`/`fovy`) half of [`FrameParams`](trd_core::FrameParams), which
/// is the natural form for an orbit interaction: the object stays put and the
/// camera moves around it. Angles in radians; `yaw = 0` looks along `-Z`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    pub yaw: f32,
    /// Elevation above the `XZ` plane, clamped to `±MAX_PITCH`.
    pub pitch: f32,
    /// Clamped to `[MIN_DISTANCE, MAX_DISTANCE]`.
    pub distance: f32,
    pub target: [f32; 3],
    pub fovy: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.3,
            // ~4 units frames the preview-scaled mesh at a 45° fov.
            distance: 4.0,
            target: [0.0, 0.0, 0.0],
            fovy: trd_core::DEFAULT_FOV_Y,
        }
    }
}

impl OrbitCamera {
    /// `yaw = pitch = 0` sits on `+Z` looking toward `-Z`.
    pub fn eye(&self) -> [f32; 3] {
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        let dir = Vector3::new(cp * sy, sp, cp * cy);
        let target = Point3::new(self.target[0], self.target[1], self.target[2]);
        (target + dir * self.distance).to_array()
    }

    pub fn orbit(&mut self, dyaw: f32, dpitch: f32) {
        self.yaw += dyaw;
        self.pitch = (self.pitch + dpitch).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// `factor < 1` dollies toward the target, `> 1` away.
    pub fn dolly(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(MIN_DISTANCE, MAX_DISTANCE);
    }
}

/// A single object's placement, producing the per-draw model matrix `T · R · S`
/// that `trd-core` applies beneath the mesh's preview base model. Rotation is
/// intrinsic yaw (`+Y`), pitch (`+X`), roll (`+Z`), in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectTransform {
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    /// Per-axis, each clamped to `[MIN_SCALE, MAX_SCALE]`.
    pub scale: [f32; 3],
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
    fn rotation(&self) -> Rotation {
        Rotation::from_rotation_y(self.yaw)
            * Rotation::from_rotation_x(self.pitch)
            * Rotation::from_rotation_z(self.roll)
    }

    /// Built with the typed `trd-core` transforms so the affine composition rule
    /// (`a.then(b) == b · a`) holds, and handed out typed (#235 R3).
    pub fn model_matrix(&self) -> Matrix4 {
        self.model_matrix_offset([0.0, 0.0, 0.0])
    }

    /// The model matrix `T(translation + offset) · R · S` — the object's own
    /// `T·R·S` shifted by a **world-space** `offset` (applied after rotation, like
    /// the translation). Used to lay out a multi-object scene: each object keeps
    /// its own transform while a per-object layout `offset` spreads them apart.
    pub fn model_matrix_offset(&self, offset: [f32; 3]) -> Matrix4 {
        let rotation = self.rotation();
        let scale = Vector3::new(self.scale[0], self.scale[1], self.scale[2]);
        let translation = Vector3::new(
            self.translation[0] + offset[0],
            self.translation[1] + offset[1],
            self.translation[2] + offset[2],
        );
        Transform::from_scale_rotation_translation(scale, rotation, translation).matrix()
    }

    /// Translates the object by a world-space delta.
    pub fn translate(&mut self, delta: [f32; 3]) {
        self.translation[0] += delta[0];
        self.translation[1] += delta[1];
        self.translation[2] += delta[2];
    }

    /// Translates along the object's rotated local axes.
    pub fn translate_local(&mut self, delta: [f32; 3]) {
        let rotated = self.rotation() * Vector3::new(delta[0], delta[1], delta[2]);
        self.translate(rotated.to_array());
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
/// render mode, and the overlay toggles. Rebuilt into a [`FrameParams`](trd_core::FrameParams) + `draws`
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
    /// How **each** object is drawn (parallel to [`objects`](Self::objects)):
    /// filled / wireframe / textured / PBR — per-object, so each object can use a
    /// different render mode. The UI edits the **selected** object's mode.
    pub modes: Vec<RenderMode>,
    /// The Disney PBR material of **each** object (parallel to [`objects`](Self::objects)),
    /// applied to objects whose [`mode`](Self::modes) is [`RenderMode::Shaded`].
    /// Interactive: the UI edits the **selected** object's material
    /// (metallic/roughness/etc.); the bound HDR env probe lives on the renderer,
    /// set once from `--env` / `?env=`.
    pub materials: Vec<DisneyMaterial>,
    /// Per-object image-based-lighting controls, parallel to `materials`.
    pub image_based_lighting: Vec<ImageBasedLighting>,
    /// Per-object tone mapping, parallel to `materials`.
    pub tone_mappings: Vec<ToneMapping>,
    /// Per-object PBR diagnostic output, parallel to `materials`.
    pub pbr_debug_views: Vec<PbrDebugView>,
    /// Scene light-rig controls shared by every PBR object.
    pub lighting: Lighting,
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
    /// Whether the loaded HDR probe is also displayed behind the scene.
    pub show_environment_background: bool,
    /// Mip-based HDR background blur (`0` sharp, `1` fully blurred).
    pub environment_background_blur: f32,
    /// The output transform applied to the HDR **sky** — its own value, not the
    /// selected object's and not mesh 0's (#235 S6).
    ///
    /// The sky used to borrow `tone_mappings.first()`, which answered *"which
    /// object's exposure does the background follow?"* with *"index 0"* — the
    /// same class of defect #182/P9 removed for the probe yaw. Tone mapping stays
    /// **per object** for the objects (it is a feature, edited in the PBR panel);
    /// the background simply owns the one that is the background's. Seeded from
    /// the front-end's initial tone mapping, so a freshly loaded scene looks
    /// exactly as it did.
    pub environment_background_tone_mapping: ToneMapping,
    /// The current front end supplied an HDR probe that can be displayed.
    pub environment_available: bool,
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
            modes: vec![RenderMode::Filled],
            materials: vec![DisneyMaterial::default()],
            image_based_lighting: vec![ImageBasedLighting::default()],
            tone_mappings: vec![ToneMapping::default()],
            pbr_debug_views: vec![PbrDebugView::default()],
            lighting: Lighting::default(),
            show_aabb: false,
            show_axes: false,
            show_local_axes: false,
            show_world_grid: false,
            show_local_grid: false,
            show_environment_background: false,
            environment_background_blur: 0.65,
            environment_background_tone_mapping: ToneMapping::default(),
            environment_available: false,
            selected: None,
        }
    }
}

/// What a front-end wants the scene to start out looking like.
///
/// Both delivery surfaces used to assemble [`SceneState`]'s parallel per-object
/// vectors by hand — the native CLI in `trd-gui-app`'s `Cli::scene_state`, the
/// browser in `start()` — which meant duplicating the rule that
/// `objects` / `modes` / `materials` / `image_based_lighting` / `tone_mappings` /
/// `pbr_debug_views` must all stay the **same length**, since `draws()` and the
/// UI index into them positionally. This carries only the values that actually
/// differ between front-ends; [`SceneState::seeded`] enforces the invariant.
#[derive(Debug, Clone)]
pub struct SceneSeed {
    /// One material per object. Its length **is** the object count: glTF assets
    /// bring their own imported material, everything else repeats a default.
    pub materials: Vec<DisneyMaterial>,
    /// The render mode every object starts in (each is editable afterwards).
    pub mode: RenderMode,
    /// Initial image-based-lighting controls, applied to every object.
    pub image_based_lighting: ImageBasedLighting,
    /// Initial output transform, applied to every object.
    pub tone_mapping: ToneMapping,
    /// The shared scene light rig.
    pub lighting: Lighting,
    /// Whether an HDR probe was supplied; also decides whether the environment
    /// is drawn as the background.
    pub environment_available: bool,
}

impl SceneState {
    /// Appends an object for a mesh just added to the renderer, keeping every
    /// per-object vector the same length, and returns its index — which **is**
    /// the renderer's new mesh id, since [`draws`](Self::draws) maps row `i` to
    /// `mesh_id: i`.
    ///
    /// The new object is selected, so the transform and PBR panels act on what
    /// was just loaded rather than on whatever was selected before.
    pub fn add_object(
        &mut self,
        material: DisneyMaterial,
        mode: RenderMode,
        tone_mapping: ToneMapping,
    ) -> u32 {
        self.objects.push(ObjectTransform::default());
        self.modes.push(mode);
        self.materials.push(material);
        self.image_based_lighting
            .push(ImageBasedLighting::default());
        self.tone_mappings.push(tone_mapping);
        self.pbr_debug_views.push(PbrDebugView::default());
        let index = (self.objects.len() - 1) as u32;
        self.selected = Some(index);
        index
    }

    /// The initial scene for [`SceneSeed`], with every per-object vector exactly
    /// `seed.materials.len()` long.
    pub fn seeded(seed: SceneSeed) -> Self {
        let n = seed.materials.len();
        Self {
            objects: vec![ObjectTransform::default(); n],
            modes: vec![seed.mode; n],
            materials: seed.materials,
            image_based_lighting: vec![seed.image_based_lighting; n],
            tone_mappings: vec![seed.tone_mapping; n],
            pbr_debug_views: vec![PbrDebugView::default(); n],
            lighting: seed.lighting,
            environment_available: seed.environment_available,
            show_environment_background: seed.environment_available,
            // Same starting point as the objects, then edited independently.
            environment_background_tone_mapping: seed.tone_mapping,
            ..Self::default()
        }
    }
}

impl SceneState {
    /// The [`Camera`] for a `viewport`-sized target, derived from the orbit
    /// camera's eye/target/fovy.
    ///
    /// The GUI owns a real camera, so it builds one directly rather than
    /// encoding it into the wire-shaped `FrameParams` the renderer used to take
    /// (#203). The model matrix rides on the per-draw list (see [`Self::draws`]),
    /// not here.
    pub fn camera(&self, viewport: Viewport) -> Camera {
        let eye = self.camera.eye();
        Camera::look_at(
            Point3::new(eye[0], eye[1], eye[2]),
            Point3::new(
                self.camera.target[0],
                self.camera.target[1],
                self.camera.target[2],
            ),
            Vector3::Y,
            self.camera.fovy,
            viewport,
        )
    }

    /// The per-frame draw list: one instance per object in
    /// [`objects`](Self::objects), drawn as mesh `i` (row index) placed by that
    /// object's model matrix **shifted by a per-object layout offset** so multiple
    /// objects spread side-by-side along world `X` (a single object stays at the
    /// origin). Each draw carries its **own** render mode (`Some(modes[i])`), so
    /// objects can mix filled / wireframe / textured / PBR.
    pub fn draws(&self) -> Vec<Draw> {
        let n = self.objects.len();
        self.objects
            .iter()
            .enumerate()
            .map(|(i, obj)| Draw {
                mesh_id: i as u32,
                model: obj.model_matrix_offset(layout_offset(i, n)),
                selection: DrawSelection::Mesh(Some(self.mode_of(i))),
            })
            .collect()
    }

    /// The render mode of object `i` (defaults to [`RenderMode::Filled`] if the
    /// index is out of range — should not happen, `modes` parallels `objects`).
    pub fn mode_of(&self, i: usize) -> RenderMode {
        self.modes.get(i).copied().unwrap_or(RenderMode::Filled)
    }

    /// A mutable reference to the selected object's render **mode**, or `None`
    /// when nothing is selected — so the Render-mode panel edits the selected
    /// object's mode and is disabled otherwise (#141).
    pub fn selected_mode_mut(&mut self) -> Option<&mut RenderMode> {
        self.selected.and_then(|i| self.modes.get_mut(i as usize))
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
    pub fn selected_pbr_mut(
        &mut self,
    ) -> Option<(
        &mut DisneyMaterial,
        &mut ImageBasedLighting,
        &mut ToneMapping,
        &mut PbrDebugView,
    )> {
        let i = self.selected? as usize;
        Some((
            self.materials.get_mut(i)?,
            self.image_based_lighting.get_mut(i)?,
            self.tone_mappings.get_mut(i)?,
            self.pbr_debug_views.get_mut(i)?,
        ))
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
        assert_eq!(obj.model_matrix(), Matrix4::IDENTITY);
    }

    #[test]
    fn translation_lands_in_the_last_column() {
        let obj = ObjectTransform {
            translation: [1.0, 2.0, 3.0],
            ..ObjectTransform::default()
        };
        let m = obj.model_matrix().to_cols_array();
        // Column-major: the translation is the 4th column (indices 12..15).
        approx([m[12], m[13], m[14]], [1.0, 2.0, 3.0]);
    }

    #[test]
    fn default_object_scale_is_unit() {
        let obj = ObjectTransform::default();
        assert_eq!(obj.scale, [1.0, 1.0, 1.0]);
        assert_eq!(obj.roll, 0.0);
        // Unit scale + zero rotation/translation ⇒ identity model.
        assert_eq!(obj.model_matrix(), Matrix4::IDENTITY);
    }

    #[test]
    fn scale_writes_the_diagonal_of_the_model() {
        let obj = ObjectTransform {
            scale: [2.0, 3.0, 4.0],
            ..ObjectTransform::default()
        };
        let m = obj.model_matrix().to_cols_array();
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
        let m = obj.model_matrix().to_cols_array();
        approx([m[0], m[1], m[2]], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn gui_scene_has_no_quad() {
        assert!(!SceneState::default().has_quad());
    }

    /// The GUI builds a `Camera` straight from its orbit state (#203) — no wire
    /// round-trip — and that camera carries the viewport it was built for, looks
    /// at the orbit target, and applies no model (which rides on the draw list).
    #[test]
    fn camera_is_built_from_the_orbit_state() {
        let state = SceneState::default();
        let viewport = Viewport {
            width: 300,
            height: 200,
        };
        let camera = state.camera(viewport);

        assert_eq!(camera.viewport(), viewport);
        // The eye sits where the orbit camera says, and the view looks at the
        // orbit target — so the pose's translation is the eye.
        let eye = state.camera.eye();
        let pose = camera.to_pose().matrix().to_cols_array();
        for (axis, expected) in eye.iter().enumerate() {
            assert!(
                (pose[12 + axis] - expected).abs() < 1e-5,
                "eye axis {axis}: {} vs {expected}",
                pose[12 + axis]
            );
        }
        assert_eq!(camera.position(), eye);
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
        let x = |d: &Draw| d.model.to_cols_array()[12];
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

    /// The whole point of [`SceneState::seeded`]: every per-object vector must
    /// come out the same length as `materials`, because `draws()` and the UI
    /// index into them positionally.
    #[test]
    fn seeded_keeps_every_per_object_vector_the_same_length() {
        for n in [1usize, 3] {
            let state = SceneState::seeded(SceneSeed {
                materials: vec![DisneyMaterial::default(); n],
                mode: RenderMode::Shaded,
                image_based_lighting: ImageBasedLighting::default(),
                tone_mapping: ToneMapping::default(),
                lighting: Lighting::default(),
                environment_available: true,
            });
            assert_eq!(state.objects.len(), n);
            assert_eq!(state.modes.len(), n);
            assert_eq!(state.materials.len(), n);
            assert_eq!(state.image_based_lighting.len(), n);
            assert_eq!(state.tone_mappings.len(), n);
            assert_eq!(state.pbr_debug_views.len(), n);
            assert_eq!(state.draws().len(), n);
            // An env probe seeds both flags together.
            assert!(state.environment_available);
            assert!(state.show_environment_background);
        }
    }

    /// The same invariant, on the runtime-add path: a model loaded into a live
    /// scene must extend **every** per-object vector, or `draws()` and the PBR
    /// panel start reading a different object's row than the one selected (#353).
    #[test]
    fn add_object_keeps_every_per_object_vector_the_same_length() {
        let mut state = SceneState::seeded(SceneSeed {
            materials: vec![DisneyMaterial::default(); 2],
            mode: RenderMode::Filled,
            image_based_lighting: ImageBasedLighting::default(),
            tone_mapping: ToneMapping::default(),
            lighting: Lighting::default(),
            environment_available: false,
        });

        let index = state.add_object(
            DisneyMaterial::default(),
            RenderMode::Shaded,
            ToneMapping::default(),
        );

        assert_eq!(index, 2, "the new object is appended, keeping earlier rows");
        for len in [
            state.objects.len(),
            state.modes.len(),
            state.materials.len(),
            state.image_based_lighting.len(),
            state.tone_mappings.len(),
            state.pbr_debug_views.len(),
            state.draws().len(),
        ] {
            assert_eq!(len, 3, "every per-object vector grew by exactly one");
        }
        // The draw list is what binds an object row to a renderer mesh id.
        assert_eq!(state.draws()[index as usize].mesh_id, index);
        assert_eq!(
            state.selected,
            Some(index),
            "the freshly loaded object is what the panels edit"
        );
    }

    /// Adding an object must not re-render the objects already on screen with a
    /// different mode or material — only lay them out around the newcomer.
    #[test]
    fn add_object_leaves_the_existing_objects_alone() {
        let mut state = SceneState::seeded(SceneSeed {
            materials: vec![DisneyMaterial {
                base_color: [1.0, 0.0, 0.0],
                ..DisneyMaterial::default()
            }],
            mode: RenderMode::Wireframe,
            image_based_lighting: ImageBasedLighting::default(),
            tone_mapping: ToneMapping::default(),
            lighting: Lighting::default(),
            environment_available: false,
        });

        state.add_object(
            DisneyMaterial::default(),
            RenderMode::Shaded,
            ToneMapping::default(),
        );

        assert_eq!(state.modes[0], RenderMode::Wireframe);
        assert_eq!(state.materials[0].base_color, [1.0, 0.0, 0.0]);
        assert_eq!(state.modes[1], RenderMode::Shaded);
    }

    /// The rig a loaded model is lit by is the video editor's Dragon rig: the
    /// probe alone, with direct light and ambient both off.
    #[test]
    fn the_runtime_load_rig_is_image_based_lighting_only() {
        let rig = ibl_only_lighting();
        assert_eq!(rig.scale, 0.0, "no direct light");
        assert_eq!(rig.ambient, 0.0, "and no ambient");
    }
}
