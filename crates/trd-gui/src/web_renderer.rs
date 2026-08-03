//! [`WebRenderer`] — the browser render backend (#97, Slice 4).
//!
//! The web twin of the native [`InProcRenderer`](crate::render_backend::InProcRenderer):
//! it builds a `trd-core` [`MeshRenderer`] once (its own wgpu 30 device, no
//! surface), then renders the interactive [`SceneState`] to an **offscreen**
//! texture and reads it back to RGBA — the pixels the eframe app uploads as an
//! egui texture (Strategy A: only CPU RGBA crosses, so egui's WebGL backend stays
//! independent of `trd-core`'s wgpu). Both share `trd-core`'s
//! [`OffscreenTarget`] readback harness with `trd-wasm`'s `OffscreenRenderer`.
//!
//! Rendering is **async** on wasm (GPU readback can't block the browser event
//! loop), so — unlike the native `SceneRenderer` trait — `render` is an
//! `async fn`; the app schedules it with `wasm_bindgen_futures::spawn_local`.

use trd_core::{
    build_scene, decode_params_stream, encode_params_stream, plane_grid_overlays,
    read_image_stream, DrawableObject, EnvMapData, FrameParams, GridPlane, ImageTexture, Mesh,
    MeshRenderer, OffscreenTarget, OutputSession, PickTarget, OFFSCREEN_FORMAT,
};

use crate::error::GuiError;
use crate::render_backend::ImageRgba;
use crate::scene::SceneState;

/// The world / local XZ plane-grid overlay drawables **plus** the selection AABB
/// for `state` (browser twin of the native `apply_grid_overlays` +
/// `set_selected_aabb`): appended to the scene so a filled/PBR object still gets
/// its floor / local grid and the selected object's bounding box (#140/#141).
fn scene_overlays(draws: &[trd_core::Draw], state: &SceneState) -> Vec<DrawableObject> {
    let xz = |on: bool| on.then_some(GridPlane::Xz);
    let mut overlays =
        plane_grid_overlays(draws, xz(state.show_world_grid), xz(state.show_local_grid));
    overlays.extend(trd_core::selection_aabb_overlay(draws, state.selected));
    overlays
}

/// How the browser renderer produces each frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WebBackend {
    /// Render the scene state directly on the persistent `MeshRenderer` (lowest
    /// latency; the default) — the web twin of the native `InProcRenderer`.
    #[default]
    Inproc,
    /// Round-trip the frame through the Arrow wire format: encode the scene to a
    /// `[mesh][texture?][params]` stream, decode it back through the **wasm**
    /// decoder (`InputSession`), render on the persistent device, then encode the
    /// image to an Arrow stream and decode it back (`OutputSession` /
    /// `read_image_stream`) — the web twin of the native `ArrowRoundTripRenderer`
    /// and the seam an external producer would drive. Reuses the device, so only
    /// the per-frame encode/decode is extra.
    Arrow,
}

/// A browser offscreen renderer over a `trd-core` [`MeshRenderer`].
pub struct WebRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: MeshRenderer,
    /// The shared offscreen render target + readback buffer (#103, Part B).
    target: OffscreenTarget,
    /// The object-id picking target (#141), created lazily on first pick.
    pick_target: Option<PickTarget>,
    width: u32,
    height: u32,
    backend: WebBackend,
}

impl WebRenderer {
    /// Builds the offscreen renderer (async: wgpu device creation) from the
    /// static meshes, an optional bound `texture` (sampled by
    /// [`RenderMode::Textured`](trd_core::RenderMode::Textured)), and an optional
    /// `env` HDR probe (reflected by [`RenderMode::Pbr`](trd_core::RenderMode::Pbr)
    /// metallic surfaces; the interactive material rides on the scene state).
    pub async fn new(
        meshes: &[Mesh],
        textures: &[Option<ImageTexture>],
        env: Option<EnvMapData>,
        width: u32,
        height: u32,
        backend: WebBackend,
    ) -> Result<Self, GuiError> {
        let instance = trd_core::create_instance();
        let trd_core::GpuContext { device, queue, .. } = trd_core::GpuContext::request(
            &instance,
            &trd_core::GpuRequest {
                label: "trd-gui wasm device",
                ..Default::default()
            },
        )
        .await
        .map_err(|e| GuiError::WasmRender(format!("GPU init failed: {e}")))?;

        let max_dim = device.limits().max_texture_dimension_2d;
        if width == 0 || height == 0 || width > max_dim || height > max_dim {
            return Err(GuiError::WasmRender(format!(
                "invalid render size {width}x{height} (max {max_dim})"
            )));
        }

        let mut renderer = MeshRenderer::auto_fit(&device, OFFSCREEN_FORMAT, meshes);
        // Skin each object with its **own** albedo (#141): texture `i` → mesh `i`.
        for (i, texture) in textures.iter().enumerate() {
            if let Some(texture) = texture {
                renderer.set_mesh_texture(i, texture);
            }
        }
        if let Some(env) = env {
            renderer.set_env_map(env);
        }

        // The shared offscreen harness owns the render target + readback buffer.
        let target = OffscreenTarget::new(&device, width, height)
            .map_err(|e| GuiError::WasmRender(e.to_string()))?;

        // The Arrow backend round-trips only the per-frame params through the
        // wire (the mesh is uploaded once into `renderer`, like native), so no
        // mesh/texture stream needs caching here.
        Ok(Self {
            device,
            queue,
            renderer,
            target,
            pick_target: None,
            width,
            height,
            backend,
        })
    }

    /// The fixed render dimensions.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Resolves the object under render-target pixel `(x, y)` via the id-color
    /// picking pass (#141), returning its 0-based index into `state.draws()`, or
    /// `None` for the background. `async` (the read-back can't block the browser
    /// event loop). Lazily creates the pick target sized to the render surface.
    pub async fn pick(&mut self, state: &SceneState, x: u32, y: u32) -> Option<u32> {
        let aspect = self.width as f32 / self.height.max(1) as f32;
        if self.pick_target.is_none() {
            self.pick_target = Some(PickTarget::new(&self.device, self.width, self.height));
        }
        let target = self.pick_target.as_ref()?;
        target
            .pick(
                &self.device,
                &self.queue,
                &mut self.renderer,
                state.frame_params(aspect),
                &state.draws(),
                x,
                y,
            )
            .await
    }

    /// Renders the current scene state to an RGBA image (async GPU readback),
    /// via the direct path or the Arrow round-trip per the configured backend.
    pub async fn render(&mut self, state: &SceneState) -> Result<ImageRgba, GuiError> {
        let rgba = match self.backend {
            WebBackend::Inproc => self.render_direct(state).await?,
            WebBackend::Arrow => self.render_arrow(state).await?,
        };
        Ok(ImageRgba {
            width: self.width,
            height: self.height,
            rgba,
        })
    }

    /// Direct path: build the scene from `state` and render it on the persistent
    /// device (the web twin of `InProcRenderer`).
    async fn render_direct(&mut self, state: &SceneState) -> Result<Vec<u8>, GuiError> {
        let aspect = self.width as f32 / self.height.max(1) as f32;
        let params = state.frame_params(aspect);
        for (i, m) in state.materials.iter().enumerate() {
            self.renderer.set_mesh_pbr_material(i, *m);
        }
        let mut scene = build_scene(
            &state.draws(),
            state.mode,
            state.show_aabb,
            state.show_axes,
            state.show_local_axes,
            None,
            None,
            None,
        );
        scene.extend(scene_overlays(&state.draws(), state));
        self.render_scene(params, &scene).await
    }

    /// Arrow round-trip path: serialize **only** the per-frame params (the
    /// computed matrix + draws) to the wire, decode it back through the wasm
    /// decoder ([`decode_params_stream`]), render on the persistent device (the
    /// mesh was uploaded once), then serialize the image and decode it back — the
    /// full params + image wire round-trip, matching the native backend (the mesh
    /// does **not** cross the wire per frame).
    async fn render_arrow(&mut self, state: &SceneState) -> Result<Vec<u8>, GuiError> {
        let aspect = self.width as f32 / self.height.max(1) as f32;

        // 1. Serialize the params and decode them back through the wire decoder.
        let params_bytes =
            encode_params_stream(&[state.frame_params(aspect)], Some(&[state.draws()]))?;
        let frame = decode_params_stream(&params_bytes)?
            .into_iter()
            .next()
            .ok_or(GuiError::WasmRender("decoder produced no frame".to_owned()))?;

        // 2. Build the scene from the decoded frame and render on the device.
        //    The round-trip always encodes the state's draws, so a decoded empty
        //    or absent list falls back to the live state's draws.
        let draws = match &frame.draws {
            Some(d) if !d.is_empty() => d.clone(),
            _ => state.draws(),
        };
        let scene = build_scene(
            &draws,
            state.mode,
            state.show_aabb,
            state.show_axes,
            state.show_local_axes,
            None,
            None,
            None,
        );
        let mut scene = scene;
        scene.extend(scene_overlays(&draws, state));
        for (i, m) in state.materials.iter().enumerate() {
            self.renderer.set_mesh_pbr_material(i, *m);
        }
        let rgba = self.render_scene(frame.params, &scene).await?;

        // 3. Serialize the image to an Arrow stream and decode it back — the
        //    output half of the round-trip an external consumer would read.
        let mut out = OutputSession::new(self.width, self.height)?;
        let mut bytes = out.drain_new()?;
        out.write_rgba_batch(&[rgba])?;
        bytes.extend(out.drain_new()?);
        out.finish()?;
        bytes.extend(out.drain_new()?);
        read_image_stream(std::io::Cursor::new(bytes), self.width, self.height)?
            .pop()
            .ok_or(GuiError::WasmRender(
                "image decoder produced no frame".to_owned(),
            ))
    }

    /// Encodes `scene` under `params` to the offscreen target and reads it back
    /// to tightly-packed RGBA (the async GPU-readback step shared by both paths).
    async fn render_scene(
        &mut self,
        params: FrameParams,
        scene: &[DrawableObject],
    ) -> Result<Vec<u8>, GuiError> {
        self.target
            .render(&self.device, &self.queue, &mut self.renderer, params, scene)
            .await
            .map_err(|e| GuiError::WasmRender(e.to_string()))
    }
}
