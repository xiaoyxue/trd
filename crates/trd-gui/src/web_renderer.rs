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
    build_scene, decode_params_stream, encode_params_stream, read_image_stream, DrawableObject,
    FrameParams, Mesh, MeshRenderer, OffscreenTarget, OutputSession, Texture,
    DEFAULT_PREVIEW_TARGET, OFFSCREEN_FORMAT,
};

use crate::error::GuiError;
use crate::render_backend::ImageRgba;
use crate::scene::SceneState;

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
    width: u32,
    height: u32,
    backend: WebBackend,
}

impl WebRenderer {
    /// Builds the offscreen renderer (async: wgpu device creation) from the
    /// static meshes and an optional bound `texture` (sampled by
    /// [`RenderMode::Textured`](trd_core::RenderMode::Textured)).
    pub async fn new(
        meshes: &[Mesh],
        texture: Option<&dyn Texture>,
        width: u32,
        height: u32,
        backend: WebBackend,
    ) -> Result<Self, GuiError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .map_err(|e| GuiError::WasmRender(format!("request_adapter failed: {e}")))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("trd-gui wasm device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GuiError::WasmRender(format!("request_device failed: {e}")))?;

        let max_dim = device.limits().max_texture_dimension_2d;
        if width == 0 || height == 0 || width > max_dim || height > max_dim {
            return Err(GuiError::WasmRender(format!(
                "invalid render size {width}x{height} (max {max_dim})"
            )));
        }

        let base_models: Vec<trd_core::Matrix4> = meshes
            .iter()
            .map(|m| m.preview_transform(DEFAULT_PREVIEW_TARGET).matrix())
            .collect();
        let mut renderer = MeshRenderer::new(&device, OFFSCREEN_FORMAT, meshes, &base_models);
        if let Some(texture) = texture {
            renderer.set_texture(texture);
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
            width,
            height,
            backend,
        })
    }

    /// The fixed render dimensions.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
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
        let scene = build_scene(
            &state.draws(),
            state.mode,
            state.show_aabb,
            state.show_axes,
            state.show_local_axes,
            None,
        );
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
        let draws = if frame.draws.is_empty() {
            state.draws()
        } else {
            frame.draws.clone()
        };
        let scene = build_scene(
            &draws,
            state.mode,
            state.show_aabb,
            state.show_axes,
            state.show_local_axes,
            None,
        );
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
