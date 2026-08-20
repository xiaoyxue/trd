use wasm_bindgen::prelude::*;

use trd_core::{
    DecodedFrame, DisneyMaterial, EnvMapData, FrameBatch, FrameFit, FrameParams,
    ImageBasedLighting, InlineFrameCache, InputSession, Lighting, OutputStream, RenderMode,
    RenderOptions, Renderer, Scene, ToneMapping, Tonemap,
};

use crate::PbrState;

fn error_message(context: &str, error: impl std::fmt::Display) -> String {
    format!("{context}: {error}")
}

#[derive(Debug, Clone)]
enum RendererState {
    Open,
    Finished,
    Failed(String),
}

impl RendererState {
    fn ensure_open(&self) -> Result<(), String> {
        match self {
            Self::Open => Ok(()),
            Self::Finished => Err("OffscreenRenderer is already finished".to_string()),
            Self::Failed(message) => Err(format!("OffscreenRenderer is failed: {message}")),
        }
    }
}

#[wasm_bindgen]
pub struct OffscreenRenderer {
    /// The shared GPU context, held as one value rather than cloned apart into
    /// separate `device` + `queue` fields (#180). Created eagerly by
    /// [`create`](Self::create) so a JS caller learns immediately whether the
    /// browser can render at all; the harness below waits for the stream's meshes.
    gpu: std::sync::Arc<trd_core::GpuContext>,
    /// The shared render harness (`trd-core`'s offscreen render + read-back),
    /// built lazily on the first rendered frame from the stream's leading mesh
    /// table — a streaming front-end owns the device long before it owns the
    /// meshes, so the device above is eager and this is not (#180).
    renderer: Option<Renderer>,
    /// The texture target the harness renders into and reads back from, built
    /// alongside `renderer` (#203): the harness owns no target of its own, so
    /// this front-end holds the one `Renderer::with_gpu` returns. Concretely a
    /// [`TextureTarget`](trd_core::TextureTarget), not the
    /// [`RenderTarget`](trd_core::RenderTarget) enum, because this renderer
    /// always reads its pixels back.
    target: Option<trd_core::TextureTarget>,
    /// Draw mode + every overlay toggle, in the **one** type every front-end uses
    /// to describe a frame's appearance; [`Scene::from_draws`] turns it into the
    /// scene. The renderer keeps no overlay state of its own (#180).
    options: RenderOptions,
    /// Composite the uploaded background frame texture beneath the scene as the
    /// scene's [`Background::frame`](trd_core::Background::frame) plane (#63); a
    /// no-op until a background is uploaded via
    /// [`update_frame_texture_rgba`](Self::update_frame_texture_rgba).
    composite_frame: bool,
    /// The typed Disney PBR configuration for [`RenderMode::Shaded`] draws, set via
    /// [`set_pbr_material`](Self::set_pbr_material) before the first frame and
    /// applied when the renderer is built. `None` ⇒ the renderer's default.
    pbr: Option<PbrState>,
    /// The decoded equirectangular HDR environment probe reflected by
    /// [`RenderMode::Shaded`] draws, set via [`set_env_map_hdr`](Self::set_env_map_hdr).
    /// `None` ⇒ no probe reflection.
    env_map: Option<EnvMapData>,
    /// Blur of the HDR **background sky** requested via
    /// [`set_env_background`](Self::set_env_background); `None` ⇒ no sky. The
    /// exposure and operator come from the staged PBR tone mapping, so the two
    /// are re-derived together into `options.env_background` whenever either
    /// changes (#235 R2).
    env_background_blur: Option<f32>,
    input: InputSession,
    /// Frames decoded by [`load_ipc`](Self::load_ipc) but not yet rendered,
    /// replayed on demand by [`render_index`](Self::render_index) (the generic
    /// viewer's paced playback).
    frames: Vec<DecodedFrame>,
    /// Last inline frames-table resource uploaded to the frame-plane texture.
    inline_frames: InlineFrameCache,
    /// An external/manual upload waiting to be consumed by the next render.
    external_frame_ready: bool,
    output: OutputStream<trd_core::SharedBuffer>,
    width: u32,
    height: u32,
    state: RendererState,
}

#[wasm_bindgen]
impl OffscreenRenderer {
    #[wasm_bindgen(js_name = create)]
    pub async fn create(width: u32, height: u32) -> Result<Self, JsValue> {
        console_error_panic_hook::set_once();

        // The browser has no `Write` target: JS wants the finished IPC bytes as
        // a `Uint8Array`, so this is the buffered form and keeps `drain_new`.
        let output = OutputStream::buffered(width, height, None).map_err(|error| {
            crate::js_error(error_message("invalid OffscreenRenderer dimensions", error))
        })?;

        let instance = trd_core::create_instance();
        let gpu = trd_core::GpuContext::request(
            &instance,
            &trd_core::GpuRequest {
                label: "trd OffscreenRenderer device",
                ..Default::default()
            },
        )
        .await
        .map_err(|error| crate::js_error(error_message("GPU init failed", error)))?;

        Ok(Self {
            gpu,
            renderer: None,
            target: None,
            options: RenderOptions::default(),
            width,
            height,
            composite_frame: false,
            pbr: None,
            env_map: None,
            env_background_blur: None,
            input: InputSession::new(),
            frames: Vec::new(),
            inline_frames: InlineFrameCache::default(),
            external_frame_ready: false,
            output,
            state: RendererState::Open,
        })
    }

    #[wasm_bindgen(js_name = pushIpc)]
    pub async fn push_ipc(&mut self, chunk: Vec<u8>) -> Result<Vec<u8>, JsValue> {
        if let Err(message) = self.state.ensure_open() {
            return Err(crate::js_error(message));
        }

        match self.push_open(chunk).await {
            Ok(bytes) => Ok(bytes),
            Err(message) => self.fail(message),
        }
    }

    pub fn finish(&mut self) -> Result<Vec<u8>, JsValue> {
        if let Err(message) = self.state.ensure_open() {
            return Err(crate::js_error(message));
        }

        let result = (|| {
            self.input
                .finish()
                .map_err(|error| error_message("input IPC finish failed", error))?;
            self.output
                .finish()
                .map_err(|error| error_message("output IPC finish failed", error))?;
            self.state = RendererState::Finished;
            self.output
                .drain_new()
                .map_err(|error| error_message("output IPC drain failed", error))
        })();

        match result {
            Ok(bytes) => Ok(bytes),
            Err(message) => self.fail(message),
        }
    }

    /// Selects filled (`false`) or wireframe (`true`) rendering for later frames.
    #[wasm_bindgen(js_name = setWireframe)]
    pub fn set_wireframe(&mut self, enabled: bool) {
        self.options.mode = if enabled {
            RenderMode::Wireframe
        } else {
            RenderMode::Filled
        };
    }

    /// Selects textured (`true`) rendering — sampling the stream's bound texture
    /// table at each vertex UV — or per-vertex color (`false`) for later frames.
    /// Textured meshes without a stream texture sample the default 1×1 white.
    #[wasm_bindgen(js_name = setTextured)]
    pub fn set_textured(&mut self, enabled: bool) {
        self.options.mode = if enabled {
            RenderMode::Textured
        } else {
            RenderMode::Filled
        };
    }

    /// Selects the Disney principled-BRDF (`true`) or per-vertex color (`false`)
    /// mesh path for later frames — the browser twin of the native `--pbr` flag.
    /// PBR shades the bound albedo (or the default 1×1 white) with the material
    /// set by [`set_pbr_material`](Self::set_pbr_material) and the environment
    /// probe from [`set_env_map_hdr`](Self::set_env_map_hdr).
    #[wasm_bindgen(js_name = setPbr)]
    pub fn set_pbr(&mut self, enabled: bool) {
        self.options.mode = if enabled {
            RenderMode::Shaded
        } else {
            RenderMode::Filled
        };
    }

    /// Sets the typed Disney PBR configuration applied to every
    /// [`RenderMode::Shaded`] draw — the browser twin of trd-cli's
    /// `--metallic/--roughness/--specular/--clearcoat/--env-intensity/--exposure/
    /// --ambient/--tonemap` flags. `tonemap` is `"aces"` (filmic) or anything
    /// else for Reinhard. Non-forwarded Disney parameters keep their defaults.
    #[wasm_bindgen(js_name = setPbrMaterial)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_pbr_material(
        &mut self,
        metallic: f32,
        roughness: f32,
        specular: f32,
        clearcoat: f32,
        env_intensity: f32,
        exposure: f32,
        ambient: f32,
        tonemap: &str,
    ) {
        let material = DisneyMaterial {
            metallic,
            roughness,
            specular,
            clearcoat,
            ..DisneyMaterial::default()
        };
        let lighting = Lighting {
            ambient,
            ..Lighting::default()
        };
        let ibl = ImageBasedLighting {
            intensity: env_intensity,
        };
        let tone_mapping = ToneMapping {
            exposure,
            operator: match tonemap.to_ascii_lowercase().as_str() {
                "aces" => Tonemap::Aces,
                _ => Tonemap::Reinhard,
            },
        };
        let pbr = PbrState::new(material, lighting, ibl, tone_mapping);
        if let Some(renderer) = self.renderer.as_mut() {
            pbr.apply(renderer);
        }
        self.pbr = Some(pbr);
        // The sky follows the same output transform as the objects in front of
        // it, so re-derive it whenever the tone mapping changes (#235 R2).
        self.refresh_env_background();
    }

    /// Draws the bound HDR environment probe as the frame's **background sky**
    /// behind every primitive — the browser twin of trd-cli's `--env-background`.
    /// `blur` is `0.0` (sharp) … `1.0` (fully blurred); the exposure and
    /// tone-map operator follow [`set_pbr_material`](Self::set_pbr_material), so
    /// the sky and the objects cannot be tone-mapped differently. Needs a probe
    /// from [`set_env_map_hdr`](Self::set_env_map_hdr) — with none bound the sky
    /// is black.
    #[wasm_bindgen(js_name = setEnvBackground)]
    pub fn set_env_background(&mut self, enabled: bool, blur: f32) {
        self.env_background_blur = enabled.then_some(blur);
        self.refresh_env_background();
    }

    /// Re-derives the scene's sky from the requested blur + the staged PBR tone
    /// mapping. The **one** place the two are combined in this front-end.
    fn refresh_env_background(&mut self) {
        self.options.env_background =
            crate::env_background(self.env_background_blur, self.pbr.as_ref());
    }

    /// Decodes an equirectangular Radiance `.hdr` buffer and binds it as the
    /// environment probe reflected by [`RenderMode::Shaded`] draws — the browser
    /// twin of trd-cli's `--env HDR` (decoded here, downscaled to 2048px).
    #[wasm_bindgen(js_name = setEnvMapHdr)]
    pub fn set_env_map_hdr(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        let env = crate::decode_env_hdr(bytes).map_err(crate::js_error)?;
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_env_map(env.clone());
        }
        self.env_map = Some(env);
        Ok(())
    }

    /// Toggles the per-instance AABB overlay box for later frames.
    #[wasm_bindgen(js_name = setShowAabb)]
    pub fn set_show_aabb(&mut self, enabled: bool) {
        self.options.show_aabb = enabled;
    }

    /// Toggles the origin coordinate-axes overlay gizmo for later frames.
    #[wasm_bindgen(js_name = setShowAxes)]
    pub fn set_show_axes(&mut self, enabled: bool) {
        self.options.show_axes = enabled;
    }

    /// Toggles the per-draw **local** coordinate-axes gizmo for later frames — one
    /// [`CoordinateAxes`](trd_core::Primitive::CoordinateAxes) gizmo at each object's own `model`. The
    /// browser twin of the native `--axes-local` flag.
    #[wasm_bindgen(js_name = setShowLocalAxes)]
    pub fn set_show_local_axes(&mut self, enabled: bool) {
        self.options.show_local_axes = enabled;
    }

    /// Toggles compositing the uploaded background frame beneath the scene as the
    /// scene's [`Background::frame`](trd_core::Background::frame) plane (#63).
    /// Enable it, then upload one background
    /// per frame via [`update_frame_texture_rgba`](Self::update_frame_texture_rgba)
    /// before that frame's [`render_index`](Self::render_index).
    #[wasm_bindgen(js_name = setCompositeFrame)]
    pub fn set_composite_frame(&mut self, enabled: bool) {
        self.composite_frame = enabled;
    }

    /// Uploads one RGBA background image as the reused frame-plane texture (#63),
    /// composited beneath the scene when [`set_composite_frame`](Self::set_composite_frame)
    /// is enabled. `rgba` must be exactly `width * height * 4` bytes (tightly
    /// packed) and neither dimension may be zero.
    #[wasm_bindgen(js_name = updateFrameTextureRgba)]
    pub fn update_frame_texture_rgba(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(), JsValue> {
        self.state.ensure_open().map_err(crate::js_error)?;
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|wh| wh.checked_mul(4));
        if width == 0 || height == 0 || expected != Some(rgba.len()) {
            return Err(crate::js_error(format!(
                "frame texture rgba must be width*height*4 bytes (got {} for {width}x{height})",
                rgba.len()
            )));
        }
        // Building the renderer needs the leading mesh table already decoded
        // (the protocol is mesh-first; the mesh renderer requires ≥1 mesh).
        if !self.input.has_meshes() {
            return Err(crate::js_error(
                "input is missing the required leading mesh table (protocol is mesh-first)",
            ));
        }
        self.ensure_renderer()
            .map_err(crate::js_error)?
            .update_frame_texture_rgba(rgba, width, height);
        self.inline_frames.invalidate();
        self.external_frame_ready = true;
        Ok(())
    }

    /// Decodes frames from an Arrow IPC chunk and **buffers** them without
    /// rendering, returning the running total. Unlike [`push_ipc`](Self::push_ipc)
    /// (which renders and emits an output stream), this only stages frames for
    /// paced replay by [`render_index`](Self::render_index). Push the whole
    /// `[mesh?][texture?][params]` stream once, then render by index.
    #[wasm_bindgen(js_name = loadIpc)]
    pub fn load_ipc(&mut self, chunk: Vec<u8>) -> Result<u32, JsValue> {
        if let Err(message) = self.state.ensure_open() {
            return Err(crate::js_error(message));
        }
        let result = (|| {
            let batches = self
                .input
                .push(&chunk)
                .map_err(|error| error_message("input IPC decode failed", error))?;
            for batch in batches {
                self.frames.extend(batch);
            }
            u32::try_from(self.frames.len())
                .map_err(|_| "buffered frame count does not fit u32".to_string())
        })();
        match result {
            Ok(count) => Ok(count),
            Err(message) => self.fail(message),
        }
    }

    /// The number of frames buffered by [`load_ipc`](Self::load_ipc).
    #[wasm_bindgen(js_name = frameCount)]
    pub fn frame_count(&self) -> u32 {
        u32::try_from(self.frames.len()).unwrap_or(u32::MAX)
    }

    /// The buffered frame's optional external background reference
    /// (`frame_path`/`frame_url`) the JS shell resolves + uploads before
    /// [`render_index`](Self::render_index). `None` when out of range or absent.
    #[wasm_bindgen(js_name = frameRef)]
    pub fn frame_ref(&self, index: u32) -> Option<String> {
        self.frames
            .get(index as usize)
            .and_then(|frame| frame.frame_ref.clone())
    }

    /// Renders one buffered frame (by index) to the offscreen texture and returns
    /// its tightly-packed RGBA (`width * height * 4` bytes) for the JS shell to
    /// display — the generic viewer's per-tick call.
    #[wasm_bindgen(js_name = renderIndex)]
    pub async fn render_index(&mut self, index: u32) -> Result<Vec<u8>, JsValue> {
        if let Err(message) = self.state.ensure_open() {
            return Err(crate::js_error(message));
        }
        let frame = match self.frames.get(index as usize).cloned() {
            Some(frame) => frame,
            None => {
                return self.fail(format!(
                    "frame index {index} out of range ({} buffered)",
                    self.frames.len()
                ));
            }
        };
        let (params, scene) = match self.scene_for(&frame) {
            Ok(pair) => pair,
            Err(message) => return self.fail(message),
        };
        match self.render_frame(params, &scene).await {
            Ok(rgba) => Ok(rgba),
            Err(message) => self.fail(message),
        }
    }
}

impl OffscreenRenderer {
    fn fail<T>(&mut self, message: String) -> Result<T, JsValue> {
        self.state = RendererState::Failed(message.clone());
        Err(crate::js_error(message))
    }

    /// Lazily builds the mesh renderer on first use: a multi-mesh renderer over
    /// the stream's (required) leading mesh table (each mesh under its
    /// `preview_transform` base model). The protocol is mesh-first, so the session
    /// always carries meshes by the time frames are produced.
    fn ensure_renderer(&mut self) -> Result<&mut Renderer, String> {
        if self.renderer.is_none() {
            let meshes = self.input.meshes();
            let (renderer, target) =
                Renderer::with_gpu(self.gpu.clone(), self.width, self.height, meshes)
                    .map_err(|error| error_message("OffscreenRenderer target", error))?;
            self.renderer = Some(renderer);
            self.target = Some(target);

            // Bind the stream's texture (0.0.4) as the sampled albedo so
            // RenderMode::Textured meshes show it; absent ⇒ the default 1×1 white.
            if let Some(texture) = self.input.texture() {
                self.renderer
                    .as_mut()
                    .expect("renderer just built")
                    .set_texture(texture);
            }

            // Apply the Disney PBR material + HDR environment probe staged by the
            // JS shell before the first frame (RenderMode::Shaded draws only).
            if let Some(pbr) = &self.pbr {
                pbr.apply(self.renderer.as_mut().expect("renderer just built"));
            }
            if let Some(env) = self.env_map.clone() {
                self.renderer
                    .as_mut()
                    .expect("renderer just built")
                    .set_env_map(env);
            }
        }
        Ok(self.renderer.as_mut().expect("renderer just built"))
    }

    /// Resolves a decoded frame into its params + scene: defaults the draw list
    /// to one instance of mesh `0` for a legacy single-object frame, validates
    /// the mesh ids, then builds the scene with the current flags + optional
    /// background compositing. Shared by [`push_open`](Self::push_open) (the
    /// output-stream path) and [`render_index`](Self::render_index) (paced replay).
    fn scene_for(&mut self, frame: &DecodedFrame) -> Result<(FrameParams, Scene), String> {
        // The protocol is mesh-first: without a leading mesh table there is
        // nothing to draw (and the mesh renderer requires ≥1 mesh).
        if !self.input.has_meshes() {
            return Err(
                "input is missing the required leading mesh table (protocol is mesh-first)"
                    .to_owned(),
            );
        }
        let params = frame.params;
        let has_inline_frame = self.upload_inline_frame(frame.frame_id)?;
        let has_external_frame = std::mem::take(&mut self.external_frame_ready);
        let mesh_count = self.ensure_renderer()?.mesh_count();
        // Draw-list resolution + mesh-id validation are the protocol's rules, not
        // this harness's, so they come from the shared assembly every front-end
        // uses — same scene, same error text, as the CLI.
        let scene = Scene::try_from_frame(
            frame,
            mesh_count,
            &self.options,
            (has_inline_frame || (self.composite_frame && has_external_frame))
                .then_some(FrameFit::Stretch),
        )
        .map_err(|error| error.to_string())?
        // The staged light rig belongs to the frame, not to the harness (#182).
        .with_lighting(
            self.pbr
                .as_ref()
                .map(PbrState::lighting)
                .unwrap_or_default(),
        );
        Ok((params, scene))
    }

    fn upload_inline_frame(&mut self, frame_id: Option<u32>) -> Result<bool, String> {
        let resolved = self
            .inline_frames
            .resolve(frame_id, self.input.frames())
            .map_err(|error| error.to_string())?;
        let Some((image, changed)) = resolved else {
            return Ok(false);
        };
        self.external_frame_ready = false;
        if changed {
            self.ensure_renderer()?.update_frame_texture_rgba(
                &image.rgba,
                image.width,
                image.height,
            );
        }
        Ok(true)
    }

    async fn push_open(&mut self, chunk: Vec<u8>) -> Result<Vec<u8>, String> {
        let frame_batches: Vec<FrameBatch> = self
            .input
            .push(&chunk)
            .map_err(|error| error_message("input IPC decode failed", error))?;

        for frame_batch in frame_batches {
            let mut images = Vec::with_capacity(frame_batch.len());

            for frame in frame_batch {
                let (params, scene) = self.scene_for(&frame)?;
                images.push(self.render_frame(params, &scene).await?);
            }

            self.output
                .write_rgba_batch(&images)
                .map_err(|error| error_message("output IPC write failed", error))?;
        }

        self.output
            .drain_new()
            .map_err(|error| error_message("output IPC drain failed", error))
    }

    async fn render_frame(
        &mut self,
        params: FrameParams,
        scene: &Scene,
    ) -> Result<Vec<u8>, String> {
        let target = self
            .target
            .as_ref()
            .expect("target built before render_frame");
        self.renderer
            .as_mut()
            .expect("renderer built before render_frame")
            .render_params(params, scene, target)
            .await
            .map_err(|error| error_message("offscreen render", error))
    }
}
