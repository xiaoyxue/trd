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
    /// Shared GPU context; created eagerly so the caller learns immediately if the browser can render.
    gpu: std::sync::Arc<trd_core::GpuContext>,
    /// Built lazily on the first rendered frame once the leading mesh table is decoded.
    renderer: Option<Renderer>,
    /// Texture target built alongside `renderer` (#203); `TextureTarget` since pixels are always read back.
    target: Option<trd_core::TextureTarget>,
    /// Draw mode and overlay toggles; [`Scene::from_draws`] turns it into the scene.
    options: RenderOptions,
    /// Enable frame-plane compositing (#63); no-op until a background is uploaded.
    composite_frame: bool,
    /// Disney PBR config for shaded draws; staged before the renderer is built. `None` ⇒ default.
    pbr: Option<PbrState>,
    /// HDR environment probe for shaded draws; `None` ⇒ no reflection.
    env_map: Option<EnvMapData>,
    /// Sky blur (0.0–1.0) or `None` for no sky; re-derived with tone mapping on change (#235).
    env_background_blur: Option<f32>,
    input: InputSession,
    /// Frames decoded by [`load_ipc`] for paced replay by [`render_index`].
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

        // Buffered output: JS reads finished IPC bytes as `Uint8Array` via `drain_new`.
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

    /// Enables textured rendering (vertex UV sampling) or per-vertex color. Absent texture ⇒ 1×1 white.
    #[wasm_bindgen(js_name = setTextured)]
    pub fn set_textured(&mut self, enabled: bool) {
        self.options.mode = if enabled {
            RenderMode::Textured
        } else {
            RenderMode::Filled
        };
    }

    /// Enables Disney PBR shading (`true`) or per-vertex color (`false`) for later frames.
    #[wasm_bindgen(js_name = setPbr)]
    pub fn set_pbr(&mut self, enabled: bool) {
        self.options.mode = if enabled {
            RenderMode::Shaded
        } else {
            RenderMode::Filled
        };
    }

    #[wasm_bindgen(js_name = meshResourceCount)]
    pub fn mesh_resource_count(&self) -> u32 {
        u32::try_from(self.input.mesh_resource_count()).unwrap_or(u32::MAX)
    }

    #[wasm_bindgen(js_name = gltfPath)]
    pub fn gltf_path(&self, index: u32) -> Option<String> {
        self.input
            .unresolved_mesh_references()
            .into_iter()
            .find(|(row, _)| *row == index)
            .and_then(|(_, reference)| reference.path)
    }

    #[wasm_bindgen(js_name = gltfUrl)]
    pub fn gltf_url(&self, index: u32) -> Option<String> {
        self.input
            .unresolved_mesh_references()
            .into_iter()
            .find(|(row, _)| *row == index)
            .and_then(|(_, reference)| reference.url)
    }

    #[wasm_bindgen(js_name = resolveGltf)]
    pub fn resolve_gltf(&mut self, index: u32, bytes: &[u8]) -> Result<(), JsValue> {
        self.input
            .resolve_gltf(index, bytes)
            .map_err(|error| crate::js_error(format!("glTF resolution failed: {error}")))
    }

    /// Sets the Disney PBR material for shaded draws. `tonemap`: `"aces"` or anything
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

    /// Draws the HDR probe as background sky. `blur`: 0.0 (sharp) … 1.0 (fully blurred).
    /// Tone map follows `set_pbr_material`; without a bound probe the sky is black.
    #[wasm_bindgen(js_name = setEnvBackground)]
    pub fn set_env_background(&mut self, enabled: bool, blur: f32) {
        self.env_background_blur = enabled.then_some(blur);
        self.refresh_env_background();
    }

    /// Re-derives the sky background from the staged blur + PBR tone mapping.
    fn refresh_env_background(&mut self) {
        self.options.env_background =
            crate::env_background(self.env_background_blur, self.pbr.as_ref());
        self.apply_stream_tonemap();
    }

    fn apply_stream_tonemap(&mut self) {
        let Some(operator) = self.input.tonemap_override() else {
            return;
        };
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_tonemap_operator(trd_core::MeshTarget::All, operator);
        }
        if let Some(background) = self.options.env_background.as_mut() {
            background.tonemap = operator;
        }
    }

    /// Decodes a Radiance `.hdr` buffer and binds it as the environment probe (downscaled to 2048px).
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

    /// Toggles per-draw local coordinate-axes gizmo for later frames.
    #[wasm_bindgen(js_name = setShowLocalAxes)]
    pub fn set_show_local_axes(&mut self, enabled: bool) {
        self.options.show_local_axes = enabled;
    }

    /// Enables frame-plane compositing (#63); upload a background per frame before each render.
    #[wasm_bindgen(js_name = setCompositeFrame)]
    pub fn set_composite_frame(&mut self, enabled: bool) {
        self.composite_frame = enabled;
    }

    /// Uploads an RGBA background for frame-plane compositing (#63).
    /// `rgba` must be `width * height * 4` bytes; neither dimension may be zero.
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
        // Protocol is mesh-first: needs the leading mesh table before a renderer can be built.
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

    /// Decodes frames from a chunk and buffers them (no render). Push the whole stream
    /// once, then replay by index via [`render_index`].
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

    /// External background reference for the buffered frame at `index`; `None` if absent.
    #[wasm_bindgen(js_name = frameRef)]
    pub fn frame_ref(&self, index: u32) -> Option<String> {
        self.frames
            .get(index as usize)
            .and_then(|frame| frame.frame_ref.clone())
    }

    /// Renders a buffered frame by index; returns `width * height * 4` RGBA bytes.
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

    /// Builds the renderer on first use from the leading mesh table; applies staged texture, PBR, and env map.
    fn ensure_renderer(&mut self) -> Result<&mut Renderer, String> {
        if self.renderer.is_none() {
            let meshes = self.input.meshes();
            let (renderer, target) =
                Renderer::with_gpu(self.gpu.clone(), self.width, self.height, meshes)
                    .map_err(|error| error_message("OffscreenRenderer target", error))?;
            self.renderer = Some(renderer);
            self.target = Some(target);

            for (index, asset) in self.input.mesh_assets().iter().enumerate() {
                let mesh_id = asset.mesh_id_or(index as u32) as usize;
                let renderer = self.renderer.as_mut().expect("renderer just built");
                renderer.set_disney_material(
                    trd_core::MeshTarget::One(mesh_id),
                    asset.material.clone(),
                );
                if let Some(texture) = asset.base_color_texture.as_ref() {
                    renderer.set_mesh_texture(mesh_id, texture);
                }
                if let Some(texture) = asset.metallic_roughness_texture.as_ref() {
                    renderer.set_mesh_metallic_roughness_texture(mesh_id, texture);
                }
                if let Some(texture) = asset.normal_texture.as_ref() {
                    renderer.set_mesh_normal_texture(mesh_id, texture);
                }
            }

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
        self.apply_stream_tonemap();
        Ok(self.renderer.as_mut().expect("renderer just built"))
    }

    /// Resolves a decoded frame into `(FrameParams, Scene)`, shared by `push_open` and `render_index`.
    fn scene_for(&mut self, frame: &DecodedFrame) -> Result<(FrameParams, Scene), String> {
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
