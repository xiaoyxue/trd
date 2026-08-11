//! Headless GPU batch render harness (#134).
//!
//! [`BatchRenderer`] is trd-core's persistent offscreen render context: it owns
//! the wgpu `device`/`queue`, a [`SceneRenderer`], and the shared
//! [`OffscreenTarget`] readback plumbing, plus the CLI overlay toggles
//! (`show_aabb`/`show_axes`/`show_local_axes`/`show_local_grid`) and the mesh-pass
//! MSAA sample count, rendering one [`FrameParams`] to tightly-packed row-major
//! RGBA bytes per call. Relocated out of `stream.rs` (which keeps only the
//! decode->render->encode orchestration) so it sits beside its siblings
//! [`OffscreenTarget`] and [`SceneRenderer`] under `render/` (#134, #82).

use std::sync::Arc;

use super::{
    Draw, DrawableObject, FrameFit, FrameParams, GpuContext, GridPlane, Mesh, OffscreenTarget,
    PickTarget, RenderMode, SceneRenderer, OFFSCREEN_FORMAT,
};
use crate::math::Matrix4;
use crate::stream::{check_dimensions, StreamError};

/// A persistent GPU context that renders one [`FrameParams`] to tightly-packed
/// row-major RGBA bytes (`width*height*4`) per call.
pub struct BatchRenderer {
    gpu: Arc<GpuContext>,
    renderer: SceneRenderer,
    /// The shared offscreen render target + readback buffer (#103, Part B).
    target: OffscreenTarget,
    /// Render mode (filled/wireframe) applied to every mesh drawable this
    /// renderer builds into its per-frame [`Scene`](crate::Scene).
    mode: RenderMode,
    /// Whether to add a [`DrawableObject::AabbBox`] gizmo per drawn instance.
    show_aabb: bool,
    /// Whether to add a single origin [`DrawableObject::CoordinateAxes`] gizmo.
    show_axes: bool,
    /// Whether to add a [`DrawableObject::CoordinateAxes`] at *each* drawn
    /// instance's own `model` — the object's local coordinate frame.
    show_local_axes: bool,
    /// If `Some(plane)`, add a [`DrawableObject::PlaneGrid`] on that coordinate
    /// plane at *each* drawn instance's own `model` — a grid lattice in the
    /// object's local frame (e.g. an `xy` grid tiling a placement quad).
    show_local_grid: Option<GridPlane>,
    /// If `Some(id)`, narrow the [`show_local_grid`](Self::show_local_grid)
    /// overlay to draws of that `mesh_id` only (the placement quad), so a
    /// wireframe *content* mesh doesn't also pick up a floor grid (#114).
    show_local_grid_mesh: Option<u32>,
    /// If `Some(plane)`, add one world-origin [`DrawableObject::PlaneGrid`] on
    /// that plane (identity model — a world floor), ungated by render mode.
    show_world_grid: Option<GridPlane>,
    /// If `Some(plane)`, add a [`DrawableObject::PlaneGrid`] on that plane at
    /// *each* drawn instance's own `model`, ungated by render mode (unlike
    /// [`show_local_grid`](Self::show_local_grid), which is wireframe-scoped).
    show_object_grid: Option<GridPlane>,
    /// If `Some(index)`, highlight that draw's [`DrawableObject::AabbBox`] — the
    /// **selected** object (#141) — regardless of the global `show_aabb` toggle.
    selected_aabb: Option<u32>,
    /// The object-id picking target (#141), created lazily on the first
    /// [`pick`](Self::pick) call and resized to track the render size. `None`
    /// until a front-end actually picks, so the headless CLI never allocates it.
    pick_target: Option<PickTarget>,
}

impl BatchRenderer {
    /// Builds the GPU context (instance/adapter/device/pipeline/target/readback)
    /// once for a fixed `width` x `height`, rendering the `meshes` of the stream's
    /// leading mesh table, applying each mesh's [`Mesh::preview_transform`]
    /// (center + uniform scale-to-fit) beneath its per-frame model so an
    /// arbitrary-unit asset renders centered and at a reasonable size. Per-frame
    /// draw lists place instances of these meshes by index. The mesh pass renders
    /// at 4× MSAA; use [`with_meshes_sample_count`](Self::with_meshes_sample_count)
    /// to override (e.g. `1` = no MSAA).
    pub fn with_meshes(width: u32, height: u32, meshes: &[Mesh]) -> Result<Self, StreamError> {
        Self::with_meshes_sample_count(width, height, meshes, crate::render::MSAA_SAMPLE_COUNT)
    }

    /// Like [`with_meshes`](Self::with_meshes) but with an explicit mesh-pass MSAA
    /// `sample_count` (`4` = anti-aliased, `1` = single-sampled / no MSAA).
    pub fn with_meshes_sample_count(
        width: u32,
        height: u32,
        meshes: &[Mesh],
        sample_count: u32,
    ) -> Result<Self, StreamError> {
        let base_models: Vec<Matrix4> = meshes
            .iter()
            .map(|mesh| {
                mesh.preview_transform(crate::DEFAULT_PREVIEW_TARGET)
                    .matrix()
            })
            .collect();
        pollster::block_on(Self::new_async(
            width,
            height,
            meshes,
            &base_models,
            sample_count,
        ))
    }

    async fn new_async(
        width: u32,
        height: u32,
        meshes: &[Mesh],
        base_models: &[Matrix4],
        sample_count: u32,
    ) -> Result<Self, StreamError> {
        // Guard against zero / overflow before allocating (device limits below).
        check_dimensions(width, height)?;

        let instance = crate::create_instance();
        // The headless CLI keeps its historical conservative limits + memory hint
        // (Downlevel 2048 cap, MemoryUsage) so the golden render stays
        // byte-identical; power preference is HighPerformance (the default).
        let gpu = crate::GpuContext::request(
            &instance,
            &crate::GpuRequest {
                limits: crate::LimitsPreset::Downlevel,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                ..Default::default()
            },
        )
        .await
        .map_err(|e| StreamError::Render(e.to_string()))?;
        let format = OFFSCREEN_FORMAT;
        let renderer = SceneRenderer::with_sample_count(
            gpu.clone(),
            format,
            meshes,
            base_models,
            sample_count,
        );

        // The shared offscreen harness owns the render target + readback buffer
        // and re-validates the size against the adapter's max dimension.
        let target = OffscreenTarget::new(&gpu.device, width, height)?;

        Ok(Self {
            gpu,
            renderer,
            target,
            mode: RenderMode::Filled,
            show_aabb: false,
            show_axes: false,
            show_local_axes: false,
            show_local_grid: None,
            show_local_grid_mesh: None,
            show_world_grid: None,
            show_object_grid: None,
            selected_aabb: None,
            pick_target: None,
        })
    }

    /// The number of loaded meshes; valid [`Draw::mesh_id`]s are `0..mesh_count`.
    pub fn mesh_count(&self) -> usize {
        self.renderer.mesh_count()
    }

    /// Sets the [`RenderMode`] (filled or wireframe) applied to later `render`s.
    pub fn set_mode(&mut self, mode: RenderMode) {
        self.mode = mode;
    }

    /// Binds `texture` as the source sampled by [`RenderMode::Textured`] meshes
    /// (`0.0.4`). Delegates to [`SceneRenderer::set_texture`]; the image is
    /// (re)uploaded on the next `render`.
    pub fn set_texture(&mut self, texture: &dyn crate::texture::Texture) {
        self.renderer.set_texture(texture);
    }

    /// Binds `texture` as the albedo of mesh `mesh_id` — a **per-object** diffuse
    /// for multi-object scenes (#141). Delegates to
    /// [`SceneRenderer::set_mesh_texture`]; out-of-range ids are ignored.
    pub fn set_mesh_texture(&mut self, mesh_id: usize, texture: &dyn crate::texture::Texture) {
        self.renderer.set_mesh_texture(mesh_id, texture);
    }

    /// Sets the [`DisneyMaterial`](crate::DisneyMaterial) applied to every PBR mesh.
    pub fn set_disney_material(&mut self, material: crate::DisneyMaterial) {
        self.renderer.set_disney_material(material);
    }

    /// Sets one mesh's Disney material.
    pub fn set_mesh_disney_material(&mut self, mesh_id: usize, material: crate::DisneyMaterial) {
        self.renderer.set_mesh_disney_material(mesh_id, material);
    }

    /// Sets scene lighting controls shared by every PBR object.
    pub fn set_lighting(&mut self, lighting: crate::Lighting) {
        self.renderer.set_lighting(lighting);
    }

    /// Sets image-based-lighting controls for every PBR object.
    pub fn set_image_based_lighting(&mut self, ibl: crate::ImageBasedLighting) {
        self.renderer.set_image_based_lighting(ibl);
    }

    /// Sets one mesh's image-based-lighting controls.
    pub fn set_mesh_image_based_lighting(
        &mut self,
        mesh_id: usize,
        ibl: crate::ImageBasedLighting,
    ) {
        self.renderer.set_mesh_image_based_lighting(mesh_id, ibl);
    }

    /// Sets the output transform of every PBR object.
    pub fn set_tone_mapping(&mut self, tone_mapping: crate::ToneMapping) {
        self.renderer.set_tone_mapping(tone_mapping);
    }

    /// Sets one mesh's output transform.
    pub fn set_mesh_tone_mapping(&mut self, mesh_id: usize, tone_mapping: crate::ToneMapping) {
        self.renderer.set_mesh_tone_mapping(mesh_id, tone_mapping);
    }

    /// Selects a diagnostic PBR output for one mesh.
    pub fn set_mesh_pbr_debug_view(&mut self, mesh_id: usize, debug_view: crate::PbrDebugView) {
        self.renderer.set_mesh_pbr_debug_view(mesh_id, debug_view);
    }

    /// Binds `env` as the equirectangular HDR environment map reflected by
    /// [`RenderMode::Pbr`] meshes. Delegates to [`SceneRenderer::set_env_map`]; the
    /// probe is (re)uploaded on the next `render`.
    pub fn set_env_map(&mut self, env: crate::EnvMapData) {
        self.renderer.set_env_map(env);
    }

    /// Enables/disables the per-instance AABB overlay box: when on, each drawn
    /// instance also contributes a [`DrawableObject::AabbBox`] to the scene.
    pub fn set_show_aabb(&mut self, show: bool) {
        self.show_aabb = show;
    }

    /// Enables/disables the origin coordinate-axes overlay gizmo: when on, the
    /// scene gains a single [`DrawableObject::CoordinateAxes`] at the world
    /// origin.
    pub fn set_show_axes(&mut self, show: bool) {
        self.show_axes = show;
    }

    /// Enables/disables the per-instance *local* coordinate-axes overlay: when
    /// on, each drawn instance also gains a [`DrawableObject::CoordinateAxes`]
    /// placed by its own `model`, visualizing that object's local frame (e.g.
    /// #77's `(e1,e2,e3)` quad placement).
    pub fn set_show_local_axes(&mut self, show: bool) {
        self.show_local_axes = show;
    }

    /// Selects the per-instance *local* coordinate-plane grid overlay: when
    /// `Some(plane)`, each drawn instance also gains a
    /// [`DrawableObject::PlaneGrid`] on that plane placed by its own `model`,
    /// laying a grid lattice across the object's local frame (e.g. an `xy` grid
    /// tiling a placement quad's floor). `None` disables it.
    pub fn set_show_local_grid(&mut self, plane: Option<GridPlane>) {
        self.show_local_grid = plane;
    }

    /// Narrows the [`set_show_local_grid`](Self::set_show_local_grid) overlay to
    /// draws of a single `mesh_id` (the placement quad). `Some(id)` lays the grid
    /// only under that mesh — so a *content* mesh drawn wireframe (e.g. a
    /// wireframe-reveal intro) doesn't also pick up a floor grid; `None` keeps the
    /// grid on every wireframe draw (#114).
    pub fn set_show_local_grid_mesh(&mut self, mesh: Option<u32>) {
        self.show_local_grid_mesh = mesh;
    }

    /// Overlays a single **world-origin** [`DrawableObject::PlaneGrid`] on the
    /// given plane (identity model — a world floor), ungated by render mode.
    /// `None` disables it. Analogous to [`set_show_axes`](Self::set_show_axes).
    pub fn set_show_world_grid(&mut self, plane: Option<GridPlane>) {
        self.show_world_grid = plane;
    }

    /// Overlays a [`DrawableObject::PlaneGrid`] on the given plane at *each* drawn
    /// object's own `model` frame, ungated by render mode (unlike
    /// [`set_show_local_grid`](Self::set_show_local_grid), which is scoped to
    /// wireframe placement quads). `None` disables it. Analogous to
    /// [`set_show_local_axes`](Self::set_show_local_axes).
    pub fn set_show_object_grid(&mut self, plane: Option<GridPlane>) {
        self.show_object_grid = plane;
    }

    /// Highlights the **selected** object's [`DrawableObject::AabbBox`] (#141):
    /// `Some(index)` boxes the draw at that 0-based index (regardless of the
    /// global [`set_show_aabb`](Self::set_show_aabb) toggle); `None` clears it.
    pub fn set_selected_aabb(&mut self, index: Option<u32>) {
        self.selected_aabb = index;
    }

    /// Uploads `image` as the **background frame texture** (#63) sampled by a
    /// [`DrawableObject::FramePlane`]. The GPU texture is reused across frames
    /// (grown only on a resolution change). Call before a
    /// [`render_frame`](Self::render_frame) with a `Some(fit)` to composite the
    /// image beneath the mesh scene.
    pub fn update_frame_texture(&mut self, image: &crate::texture::ImageData) {
        self.renderer
            .update_frame_texture_rgba(&image.rgba, image.width, image.height);
    }

    /// Builds the per-frame [`Scene`](crate::Scene) from a wire `draws` list and
    /// this renderer's mode/overlay flags (delegates to [`build_scene`]). A
    /// `Some(fit)` prepends a background [`DrawableObject::FramePlane`] (#63).
    fn build_scene(&self, draws: &[Draw], frame: Option<FrameFit>) -> Vec<DrawableObject> {
        let mut scene = crate::render::build_scene(
            draws,
            self.mode,
            self.show_aabb,
            self.show_axes,
            self.show_local_axes,
            self.show_local_grid,
            self.show_local_grid_mesh,
            frame,
        );
        // World / object plane-grid overlays (#140) — ungated by render mode, so a
        // filled/PBR object still gets a floor grid. `encode` buckets by primitive
        // type, so appending here draws them in the grid pass regardless of order.
        scene.extend(crate::render::plane_grid_overlays(
            draws,
            self.show_world_grid,
            self.show_object_grid,
        ));
        // Selection highlight (#141): the selected object's AABB, drawn even when
        // the global show-all-AABBs toggle is off.
        scene.extend(crate::render::selection_aabb_overlay(
            draws,
            self.selected_aabb,
        ));
        scene
    }

    /// Renders `params` with the given per-frame instance `draws`, compositing a
    /// background [`DrawableObject::FramePlane`] (#63) beneath the scene when
    /// `frame` is `Some(fit)` and a frame texture has been uploaded via
    /// [`update_frame_texture`](Self::update_frame_texture). Returns
    /// tightly-packed row-major RGBA bytes (`width*height*4`).
    pub fn render_frame(
        &mut self,
        params: FrameParams,
        draws: &[Draw],
        frame: Option<FrameFit>,
    ) -> Result<Vec<u8>, StreamError> {
        let scene = self.build_scene(draws, frame);
        Ok(pollster::block_on(self.target.render(
            &self.gpu,
            &mut self.renderer,
            params,
            &scene,
        ))?)
    }

    /// **Object-id picking** (#141): renders `draws` through the flat id-color
    /// pass at the current render size and returns the **0-based index into
    /// `draws`** of the object under pixel `(x, y)`, or `None` for the background
    /// (or an out-of-bounds coordinate). The pass is single-sampled and
    /// depth-tested, so the nearest object wins and ids are never blended — the
    /// "color index" method, no ray-marching. The lazily-created pick target
    /// tracks the display size ([`resize`](Self::resize) keeps it in sync).
    pub fn pick(&mut self, params: FrameParams, draws: &[Draw], x: u32, y: u32) -> Option<u32> {
        let (w, h) = (self.target.width(), self.target.height());
        match self.pick_target.as_mut() {
            Some(target) => target.resize(&self.gpu.device, w, h),
            None => self.pick_target = Some(PickTarget::new(&self.gpu.device, w, h)),
        }
        let target = self.pick_target.as_ref()?;
        pollster::block_on(target.pick(&self.gpu, &mut self.renderer, params, draws, x, y))
    }
}
