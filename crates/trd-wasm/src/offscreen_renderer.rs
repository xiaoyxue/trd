use wasm_bindgen::prelude::*;

use trd_core::{
    DisneyMaterial, EnvMapData, FrameFit, ImageBasedLighting, Lighting, OutputStream, RenderMode,
    RenderOptions, Renderer, ToneMapping, Tonemap,
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
    /// Built when a complete params/GLB document is loaded.
    renderer: Option<Renderer>,
    /// Texture target built alongside `renderer` (#203); `TextureTarget` since pixels are always read back.
    target: Option<trd_core::TextureTarget>,
    /// Draw mode and overlay toggles for document scene assembly.
    options: RenderOptions,
    /// Enable frame-plane compositing (#63); no-op until a background is uploaded.
    composite_frame: bool,
    /// Disney PBR config for shaded draws; staged before the renderer is built. `None` ⇒ default.
    pbr: Option<PbrState>,
    /// HDR environment probe for shaded draws; `None` ⇒ no reflection.
    env_map: Option<EnvMapData>,
    /// Sky blur (0.0–1.0) or `None` for no sky; re-derived with tone mapping on change (#235).
    env_background_blur: Option<f32>,
    document: Option<std::rc::Rc<std::cell::RefCell<trd_core::SceneDocument>>>,
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
            document: None,
            external_frame_ready: false,
            output,
            state: RendererState::Open,
        })
    }

    /// Closes rendering and returns the remaining image IPC bytes, including EOS.
    /// Only [`render_ipc`](Self::render_ipc) appends images to this output.
    pub fn finish(&mut self) -> Result<Vec<u8>, JsValue> {
        if let Err(message) = self.state.ensure_open() {
            return Err(crate::js_error(message));
        }

        let result = (|| {
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
        if let Some(document) = &self.document {
            return u32::try_from(document.borrow().meshes().len()).unwrap_or(u32::MAX);
        }
        0
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
        let Some(operator) = self.document.as_ref().and_then(|document| {
            document
                .borrow()
                .tonemap_override()
                .expect("loaded document metadata is immutable and validated")
        }) else {
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
        self.options.overlays.aabb = enabled;
    }

    /// Toggles the origin coordinate-axes overlay gizmo for later frames.
    #[wasm_bindgen(js_name = setShowAxes)]
    pub fn set_show_axes(&mut self, enabled: bool) {
        self.options.overlays.axes = enabled;
    }

    /// Toggles per-draw local coordinate-axes gizmo for later frames.
    #[wasm_bindgen(js_name = setShowLocalAxes)]
    pub fn set_show_local_axes(&mut self, enabled: bool) {
        self.options.overlays.local_axes = enabled;
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
        self.ensure_renderer()
            .map_err(crate::js_error)?
            .update_frame_texture_rgba(rgba, width, height);
        self.external_frame_ready = true;
        Ok(())
    }

    /// Loads a complete current params/GLB document, not incremental legacy IPC.
    #[wasm_bindgen(js_name = loadIpc)]
    pub fn load_ipc(&mut self, chunk: Vec<u8>) -> Result<u32, JsValue> {
        if let Err(message) = self.state.ensure_open() {
            return Err(crate::js_error(message));
        }
        let document = crate::ArrowSceneDocument::from_arrow(&chunk)?;
        self.load_scene_document(&document)
    }

    /// The number of frames buffered by [`load_ipc`](Self::load_ipc).
    #[wasm_bindgen(js_name = frameCount)]
    pub fn frame_count(&self) -> u32 {
        if let Some(document) = &self.document {
            return u32::try_from(document.borrow().row_count()).unwrap_or(u32::MAX);
        }
        0
    }

    #[wasm_bindgen(js_name = loadSceneDocument)]
    pub fn load_scene_document(
        &mut self,
        document: &crate::ArrowSceneDocument,
    ) -> Result<u32, JsValue> {
        self.state.ensure_open().map_err(crate::js_error)?;
        let source = document.inner.borrow();
        let count =
            u32::try_from(source.row_count()).map_err(|_| crate::js_error("too many rows"))?;
        source.frames().map_err(crate::js_error)?;
        let renderer = crate::scene_document::document_renderer(
            &source,
            self.gpu.clone(),
            trd_core::TEXTURE_TARGET_FORMAT,
            self.pbr.as_ref(),
            self.env_map.as_ref(),
        )?;
        let target = renderer
            .create_texture_target(self.width, self.height)
            .map_err(crate::js_error)?;
        self.renderer = Some(renderer);
        self.target = Some(target);
        self.document = Some(std::rc::Rc::clone(&document.inner));
        self.external_frame_ready = false;
        drop(source);
        self.refresh_env_background();
        Ok(count)
    }

    /// External background reference for the buffered frame at `index`; `None` if absent.
    #[wasm_bindgen(js_name = frameRef)]
    pub fn frame_ref(&self, index: u32) -> Result<Option<String>, JsValue> {
        if let Some(document) = &self.document {
            return document
                .borrow()
                .frame_ref(index as usize)
                .map_err(crate::js_error);
        }
        Err(crate::js_error("load a params/GLB document first"))
    }

    /// Renders a buffered frame by index; returns `width * height * 4` RGBA bytes.
    #[wasm_bindgen(js_name = renderIndex)]
    pub async fn render_index(&mut self, index: u32) -> Result<Vec<u8>, JsValue> {
        if let Err(message) = self.state.ensure_open() {
            return Err(crate::js_error(message));
        }
        if let Some(document) = self.document.clone() {
            let (camera, scene) = {
                let source = document.borrow();
                let frame = source.frame(index as usize).map_err(crate::js_error)?;
                let fit = (self.composite_frame && self.external_frame_ready)
                    .then_some(FrameFit::Stretch);
                trd_placement::document_scene(
                    &source,
                    &frame,
                    trd_core::Viewport {
                        width: self.width,
                        height: self.height,
                    },
                    &self.options,
                    fit,
                )
                .map_err(crate::js_error)?
            };
            self.external_frame_ready = false;
            let target = self
                .target
                .as_ref()
                .expect("document target was created on load");
            let renderer = self
                .renderer
                .as_mut()
                .expect("document renderer was created on load");
            let scene = scene.with_lighting(
                self.pbr
                    .as_ref()
                    .map(PbrState::lighting)
                    .unwrap_or_default(),
            );
            renderer.draw_layers(&[trd_core::SceneLayer::new(camera, &scene)], target);
            return renderer.read_pixels(target).await.map_err(crate::js_error);
        }
        Err(crate::js_error("load a params/GLB document first"))
    }

    /// Renders one document row into the image IPC output, in call order.
    /// Concatenate returned chunks and [`finish`](Self::finish)'s final chunk.
    /// [`render_index`](Self::render_index) remains RGBA-only for playback.
    #[wasm_bindgen(js_name = renderIpc)]
    pub async fn render_ipc(&mut self, index: u32) -> Result<Vec<u8>, JsValue> {
        let rgba = self.render_index(index).await?;
        let result = (|| {
            self.output
                .write_rgba_batch(&[rgba])
                .map_err(|error| error_message("output IPC write failed", error))?;
            self.output
                .drain_new()
                .map_err(|error| error_message("output IPC drain failed", error))
        })();
        match result {
            Ok(bytes) => Ok(bytes),
            Err(message) => self.fail(message),
        }
    }
}

impl OffscreenRenderer {
    fn fail<T>(&mut self, message: String) -> Result<T, JsValue> {
        self.state = RendererState::Failed(message.clone());
        Err(crate::js_error(message))
    }

    fn ensure_renderer(&mut self) -> Result<&mut Renderer, String> {
        self.renderer
            .as_mut()
            .ok_or_else(|| "load a params/GLB document first".to_owned())
    }
}
