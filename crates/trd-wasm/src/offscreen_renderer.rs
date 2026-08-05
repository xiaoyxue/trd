use wasm_bindgen::prelude::*;

use trd_core::{
    build_scene, DecodedFrame, Draw, DrawableObject, EnvMapData, FrameBatch, FrameFit, FrameParams,
    InputSession, MeshRenderer, OffscreenTarget, OutputSession, PbrMaterial, RenderMode, Tonemap,
    OFFSCREEN_FORMAT,
};

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
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Built lazily on the first rendered frame from the stream's leading mesh
    /// table, or the built-in hello-triangle for a legacy params-only stream.
    renderer: Option<MeshRenderer>,
    mode: RenderMode,
    show_aabb: bool,
    show_axes: bool,
    /// Per-draw *local* coordinate-axes gizmos (each object's own model frame) —
    /// the browser twin of the native `--axes-local` flag.
    show_local_axes: bool,
    /// Composite the uploaded background frame texture beneath the scene as a
    /// [`DrawableObject::FramePlane`] (#63); a no-op until a background is
    /// uploaded via [`update_frame_texture_rgba`](Self::update_frame_texture_rgba).
    composite_frame: bool,
    /// The global Disney [`PbrMaterial`] for [`RenderMode::Pbr`] draws, set via
    /// [`set_pbr_material`](Self::set_pbr_material) before the first frame and
    /// applied when the renderer is built. `None` ⇒ the renderer's default.
    pbr_material: Option<PbrMaterial>,
    /// The decoded equirectangular HDR environment probe reflected by
    /// [`RenderMode::Pbr`] draws, set via [`set_env_map_hdr`](Self::set_env_map_hdr).
    /// `None` ⇒ no probe reflection.
    env_map: Option<EnvMapData>,
    /// The shared offscreen render target + readback buffer (#103, Part B).
    target: OffscreenTarget,
    input: InputSession,
    /// Frames decoded by [`load_ipc`](Self::load_ipc) but not yet rendered,
    /// replayed on demand by [`render_index`](Self::render_index) (the generic
    /// viewer's paced playback).
    frames: Vec<DecodedFrame>,
    /// Last inline frames-table resource uploaded to the frame-plane texture.
    last_inline_frame_id: Option<u32>,
    output: OutputSession,
    state: RendererState,
}

#[wasm_bindgen]
impl OffscreenRenderer {
    #[wasm_bindgen(js_name = create)]
    pub async fn create(width: u32, height: u32) -> Result<Self, JsValue> {
        console_error_panic_hook::set_once();

        let output = OutputSession::new(width, height).map_err(|error| {
            crate::js_error(error_message("invalid OffscreenRenderer dimensions", error))
        })?;

        let instance = trd_core::create_instance();
        let trd_core::GpuContext { device, queue, .. } = trd_core::GpuContext::request(
            &instance,
            &trd_core::GpuRequest {
                label: "trd OffscreenRenderer device",
                ..Default::default()
            },
        )
        .await
        .map_err(|error| crate::js_error(error_message("GPU init failed", error)))?;

        // The shared offscreen harness owns the render target + readback buffer
        // and re-validates the size against the adapter's max dimension.
        let target = OffscreenTarget::new(&device, width, height)
            .map_err(|error| crate::js_error(error_message("OffscreenRenderer target", error)))?;

        Ok(Self {
            device,
            queue,
            renderer: None,
            mode: RenderMode::Filled,
            show_aabb: false,
            show_axes: false,
            show_local_axes: false,
            composite_frame: false,
            pbr_material: None,
            env_map: None,
            target,
            input: InputSession::new(),
            frames: Vec::new(),
            last_inline_frame_id: None,
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
        self.mode = if enabled {
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
        self.mode = if enabled {
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
        self.mode = if enabled {
            RenderMode::Pbr
        } else {
            RenderMode::Filled
        };
    }

    /// Sets the global Disney [`PbrMaterial`] applied to every
    /// [`RenderMode::Pbr`] draw — the browser twin of trd-cli's
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
        let material = PbrMaterial {
            metallic,
            roughness,
            specular,
            clearcoat,
            env_intensity,
            exposure,
            ambient,
            tonemap: match tonemap.to_ascii_lowercase().as_str() {
                "aces" => Tonemap::Aces,
                _ => Tonemap::Reinhard,
            },
            ..PbrMaterial::default()
        };
        self.pbr_material = Some(material);
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_pbr_material(material);
        }
    }

    /// Decodes an equirectangular Radiance `.hdr` buffer and binds it as the
    /// environment probe reflected by [`RenderMode::Pbr`] draws — the browser
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
        self.show_aabb = enabled;
    }

    /// Toggles the origin coordinate-axes overlay gizmo for later frames.
    #[wasm_bindgen(js_name = setShowAxes)]
    pub fn set_show_axes(&mut self, enabled: bool) {
        self.show_axes = enabled;
    }

    /// Toggles the per-draw **local** coordinate-axes gizmo for later frames — one
    /// [`DrawableObject::CoordinateAxes`] at each object's own `model`. The
    /// browser twin of the native `--axes-local` flag.
    #[wasm_bindgen(js_name = setShowLocalAxes)]
    pub fn set_show_local_axes(&mut self, enabled: bool) {
        self.show_local_axes = enabled;
    }

    /// Toggles compositing the uploaded background frame beneath the scene as a
    /// [`DrawableObject::FramePlane`] (#63). Enable it, then upload one background
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
        self.ensure_renderer();
        let queue = &self.queue;
        self.renderer
            .as_mut()
            .expect("renderer built above")
            .update_frame_texture_rgba(queue, rgba, width, height);
        self.last_inline_frame_id = None;
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
    fn ensure_renderer(&mut self) -> &mut MeshRenderer {
        if self.renderer.is_none() {
            let meshes = self.input.meshes();
            let renderer = MeshRenderer::auto_fit(&self.device, OFFSCREEN_FORMAT, meshes);
            self.renderer = Some(renderer);

            // Bind the stream's texture (0.0.4) as the sampled albedo so
            // RenderMode::Textured meshes show it; absent ⇒ the default 1×1 white.
            if let Some(texture) = self.input.texture() {
                self.renderer
                    .as_mut()
                    .expect("renderer just built")
                    .set_texture(texture);
            }

            // Apply the Disney PBR material + HDR environment probe staged by the
            // JS shell before the first frame (RenderMode::Pbr draws only).
            if let Some(material) = self.pbr_material {
                self.renderer
                    .as_mut()
                    .expect("renderer just built")
                    .set_pbr_material(material);
            }
            if let Some(env) = self.env_map.clone() {
                self.renderer
                    .as_mut()
                    .expect("renderer just built")
                    .set_env_map(env);
            }
        }
        self.renderer.as_mut().expect("renderer just built")
    }

    /// Resolves a decoded frame into its params + scene: defaults the draw list
    /// to one instance of mesh `0` for a legacy single-object frame, validates
    /// the mesh ids, then builds the scene with the current flags + optional
    /// background compositing. Shared by [`push_open`](Self::push_open) (the
    /// output-stream path) and [`render_index`](Self::render_index) (paced replay).
    fn scene_for(
        &mut self,
        frame: &DecodedFrame,
    ) -> Result<(FrameParams, Vec<DrawableObject>), String> {
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
        // Explicit wire draw list ⇒ drawn verbatim (an empty list ⇒ background
        // only); an absent draw list ⇒ one instance of mesh 0 placed by the
        // frame's own model (legacy single-object behavior).
        let draws: Vec<Draw> = frame.resolved_draws();

        let mesh_count = self.ensure_renderer().mesh_count();
        for draw in &draws {
            if draw.mesh_id as usize >= mesh_count {
                return Err(format!(
                    "draw references mesh {} but only {mesh_count} mesh(es) are loaded",
                    draw.mesh_id
                ));
            }
        }
        let scene = build_scene(
            &draws,
            self.mode,
            self.show_aabb,
            self.show_axes,
            self.show_local_axes,
            None,
            None,
            (has_inline_frame || self.composite_frame).then_some(FrameFit::Stretch),
        );
        Ok((params, scene))
    }

    fn upload_inline_frame(&mut self, frame_id: Option<u32>) -> Result<bool, String> {
        let Some(frame_id) = frame_id else {
            self.last_inline_frame_id = None;
            return Ok(false);
        };
        if self.last_inline_frame_id == Some(frame_id) {
            return Ok(true);
        }
        let image = self
            .input
            .frames()
            .get(frame_id as usize)
            .ok_or_else(|| format!("frame_id {frame_id} is out of range"))?
            .decode()
            .map_err(|error| format!("decode frame_id {frame_id}: {error}"))?;
        self.ensure_renderer();
        let queue = &self.queue;
        self.renderer
            .as_mut()
            .expect("renderer built above")
            .update_frame_texture_rgba(queue, &image.rgba, image.width, image.height);
        self.last_inline_frame_id = Some(frame_id);
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
        scene: &[DrawableObject],
    ) -> Result<Vec<u8>, String> {
        let renderer = self
            .renderer
            .as_mut()
            .expect("renderer built before render_frame");
        self.target
            .render(&self.device, &self.queue, renderer, params, scene)
            .await
            .map_err(|error| error_message("offscreen render", error))
    }
}
