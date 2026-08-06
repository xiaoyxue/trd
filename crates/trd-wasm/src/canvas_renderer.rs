use trd_core::{
    build_scene, DecodedFrame, DisneyMaterial, Draw, EnvMapData, FrameFit, ImageBasedLighting,
    Lighting, MeshRenderer, OnscreenTarget, RenderMode, ToneMapping, Tonemap,
};
use wasm_bindgen::prelude::*;

use crate::{js_error, PbrState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CanvasState {
    Open,
    Finished,
    Failed,
}

struct AcquiredFrame {
    texture: wgpu::SurfaceTexture,
    reconfigure_after_present: bool,
}

#[wasm_bindgen]
pub struct CanvasRenderer {
    instance: wgpu::Instance,
    canvas: web_sys::HtmlCanvasElement,
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The shared on-screen render harness (surface + config + sRGB view, #103).
    target: OnscreenTarget,
    /// Built lazily on the first rendered frame: a multi-mesh renderer over the
    /// stream's leading mesh table, or the built-in hello-triangle for a legacy
    /// params-only stream. `None` until the first frame arrives (the mesh table,
    /// if any, has been decoded by then).
    renderer: Option<MeshRenderer>,
    mode: RenderMode,
    show_aabb: bool,
    show_axes: bool,
    /// Per-draw *local* coordinate-axes gizmos (each object's own model frame,
    /// e.g. #77's reconstructed quad basis). The browser twin of the native
    /// `--axes-local` flag; toggled via [`set_show_local_axes`](Self::set_show_local_axes).
    show_local_axes: bool,
    /// Composite the uploaded background frame texture beneath the scene as a
    /// [`DrawableObject::FramePlane`] (#63). When `true`, later frames pass
    /// `Some(FrameFit::Stretch)` to [`build_scene`]; a [`FramePlane`] is a no-op
    /// until a background is uploaded via
    /// [`update_frame_texture_rgba`](Self::update_frame_texture_rgba).
    composite_frame: bool,
    /// The typed Disney PBR configuration for [`RenderMode::Pbr`] draws, set via
    /// [`set_pbr_material`](Self::set_pbr_material) before the first frame and
    /// applied when the renderer is built (the browser twin of trd-cli's
    /// `--metallic/--roughness/…` flags). `None` ⇒ the renderer's default.
    pbr: Option<PbrState>,
    /// The decoded equirectangular HDR environment probe reflected by
    /// [`RenderMode::Pbr`] draws, set via [`set_env_map_hdr`](Self::set_env_map_hdr)
    /// and applied when the renderer is built. `None` ⇒ no probe reflection.
    env_map: Option<EnvMapData>,
    input: trd_core::InputSession,
    /// Frames decoded by [`load_ipc`](Self::load_ipc) but not yet rendered,
    /// replayed on demand by [`render_index`](Self::render_index). The generic
    /// renderer loads the whole stream once, then paces playback by index (so the
    /// JS shell can upload each frame's background *before* rendering it).
    frames: Vec<DecodedFrame>,
    /// Last inline frames-table resource uploaded to the frame-plane texture.
    last_inline_frame_id: Option<u32>,
    /// An external/manual upload waiting to be consumed by the next render.
    external_frame_ready: bool,
    state: CanvasState,
}

#[wasm_bindgen]
impl CanvasRenderer {
    #[wasm_bindgen(js_name = create)]
    pub async fn create(canvas: web_sys::HtmlCanvasElement) -> Result<Self, JsValue> {
        console_error_panic_hook::set_once();

        let width = canvas.width();
        let height = canvas.height();
        if width == 0 || height == 0 {
            return Err(js_error("canvas width and height must be non-zero"));
        }

        let instance = trd_core::create_instance();
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|error| js_error(format!("create_surface failed: {error}")))?;
        let trd_core::GpuContext {
            adapter,
            device,
            queue,
        } = trd_core::GpuContext::request(
            &instance,
            &trd_core::GpuRequest {
                label: "trd canvas device",
                compatible_surface: Some(&surface),
                ..Default::default()
            },
        )
        .await
        .map_err(|error| js_error(format!("GPU init failed: {error}")))?;
        let config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| js_error("surface is unsupported by the selected adapter"))?;
        // The browser's preferred canvas format is non-sRGB (e.g. `Bgra8Unorm`),
        // so a pipeline targeting it writes *linear* fragment values with no
        // linear→sRGB encode — making colors look darker/muddier than the headless
        // CLI, whose target is `Rgba8UnormSrgb` (hardware-encoded on store). The
        // shared `OnscreenTarget` renders through an **sRGB view** of the surface
        // (registering it in `view_formats` + configuring), so the browser matches
        // the CLI byte-for-byte; build the mesh renderer with its `view_format()`.
        let target = OnscreenTarget::new(&device, surface, config);

        Ok(Self {
            renderer: None,
            mode: RenderMode::Filled,
            show_aabb: false,
            show_axes: false,
            show_local_axes: false,
            composite_frame: false,
            pbr: None,
            env_map: None,
            instance,
            canvas,
            device,
            queue,
            target,
            input: trd_core::InputSession::new(),
            frames: Vec::new(),
            last_inline_frame_id: None,
            external_frame_ready: false,
            state: CanvasState::Open,
        })
    }

    #[wasm_bindgen(js_name = pushIpc)]
    pub fn push_ipc(&mut self, chunk: &[u8]) -> Result<u32, JsValue> {
        self.require_open()?;

        let result = (|| {
            let batches = measure("trd.ipc.decode", || {
                self.input
                    .push(chunk)
                    .map_err(|error| js_error(format!("Arrow IPC input failed: {error}")))
            })?;
            let rendered = batches.iter().try_fold(0_u32, |total, batch| {
                let rows = u32::try_from(batch.len())
                    .map_err(|_| js_error("decoded batch row count does not fit u32"))?;
                total
                    .checked_add(rows)
                    .ok_or_else(|| js_error("rendered row count would overflow u32"))
            })?;

            for batch in &batches {
                for frame in batch {
                    self.render_frame(frame)?;
                }
            }

            Ok(rendered)
        })();

        if result.is_err() {
            self.state = CanvasState::Failed;
        }
        result
    }

    /// Decodes frames from an Arrow IPC chunk and **buffers** them without
    /// rendering (the generic viewer's load phase), returning the running total
    /// of buffered frames. The stream's leading mesh/texture tables are consumed
    /// as usual; only the params frames are buffered, to be replayed by
    /// [`render_index`](Self::render_index). Push the whole `[mesh?][texture?][params]`
    /// stream, then pace playback by index.
    #[wasm_bindgen(js_name = loadIpc)]
    pub fn load_ipc(&mut self, chunk: &[u8]) -> Result<u32, JsValue> {
        self.require_open()?;
        let result = (|| {
            let batches = self
                .input
                .push(chunk)
                .map_err(|error| js_error(format!("Arrow IPC input failed: {error}")))?;
            for batch in batches {
                self.frames.extend(batch);
            }
            u32::try_from(self.frames.len())
                .map_err(|_| js_error("buffered frame count does not fit u32"))
        })();

        if result.is_err() {
            self.state = CanvasState::Failed;
        }
        result
    }

    /// The number of frames buffered by [`load_ipc`](Self::load_ipc).
    #[wasm_bindgen(js_name = frameCount)]
    pub fn frame_count(&self) -> u32 {
        u32::try_from(self.frames.len()).unwrap_or(u32::MAX)
    }

    /// The buffered frame's optional external background reference
    /// (`frame_path`/`frame_url`), which the JS shell resolves to RGBA and uploads
    /// via [`update_frame_texture_rgba`](Self::update_frame_texture_rgba) before
    /// [`render_index`](Self::render_index). `None` when out of range or the frame
    /// has no background.
    #[wasm_bindgen(js_name = frameRef)]
    pub fn frame_ref(&self, index: u32) -> Option<String> {
        self.frames
            .get(index as usize)
            .and_then(|frame| frame.frame_ref.clone())
    }

    /// Renders one buffered frame (by index) to the surface using the current
    /// flags and any uploaded background — the generic viewer's per-tick call.
    #[wasm_bindgen(js_name = renderIndex)]
    pub fn render_index(&mut self, index: u32) -> Result<(), JsValue> {
        self.require_open()?;
        let frame = match self.frames.get(index as usize).cloned() {
            Some(frame) => frame,
            None => {
                return Err(js_error(format!(
                    "frame index {index} out of range ({} buffered)",
                    self.frames.len()
                )));
            }
        };
        let result = self.render_frame(&frame);
        if result.is_err() {
            self.state = CanvasState::Failed;
        }
        result
    }

    pub fn finish(&mut self) -> Result<(), JsValue> {
        self.require_open()?;
        match self
            .input
            .finish()
            .map_err(|error| js_error(format!("Arrow IPC finish failed: {error}")))
        {
            Ok(()) => {
                self.state = CanvasState::Finished;
                Ok(())
            }
            Err(error) => {
                self.state = CanvasState::Failed;
                Err(error)
            }
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

    /// Sets the typed Disney PBR configuration applied to every
    /// [`RenderMode::Pbr`] draw — the browser twin of trd-cli's
    /// `--metallic/--roughness/--specular/--clearcoat/--env-intensity/--exposure/
    /// --ambient/--tonemap` flags. `tonemap` is `"aces"` (filmic) or anything
    /// else for Reinhard. Non-forwarded Disney parameters keep their defaults.
    /// Takes effect on the next rendered frame (applied immediately if the
    /// renderer is already built, else when it is).
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
            ..ImageBasedLighting::default()
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
    }

    /// Decodes an equirectangular Radiance `.hdr` buffer and binds it as the
    /// environment probe reflected by [`RenderMode::Pbr`] draws — the browser
    /// twin of trd-cli's `--env HDR` (decoded here, downscaled to 2048px). Takes
    /// effect on the next rendered frame.
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
    /// [`DrawableObject::CoordinateAxes`] at each object's own `model` (its
    /// reconstructed local frame, e.g. #77's quad basis). The browser twin of the
    /// native `--axes-local` flag.
    #[wasm_bindgen(js_name = setShowLocalAxes)]
    pub fn set_show_local_axes(&mut self, enabled: bool) {
        self.show_local_axes = enabled;
    }

    /// Toggles compositing the uploaded background frame beneath the scene as a
    /// [`DrawableObject::FramePlane`] (#63). The browser twin of the native
    /// `--frames-base` compositing: enable it, then push one background per frame
    /// via [`update_frame_texture_rgba`](Self::update_frame_texture_rgba) *before*
    /// that frame's [`push_ipc`](Self::push_ipc). Has no visible effect until a
    /// background has been uploaded.
    #[wasm_bindgen(js_name = setCompositeFrame)]
    pub fn set_composite_frame(&mut self, enabled: bool) {
        self.composite_frame = enabled;
    }

    /// Uploads one RGBA background image as the reused frame-plane texture (#63),
    /// composited beneath the scene when [`set_composite_frame`](Self::set_composite_frame)
    /// is enabled. Call it with the frame matching the *next* params frame (the
    /// per-frame video still behind the AR composite). The texture is reused
    /// across frames and reallocated only on a dimension change, so streaming a
    /// fixed-resolution video allocates once.
    ///
    /// `rgba` must be exactly `width * height * 4` bytes (tightly packed, no row
    /// padding) and neither dimension may be zero.
    #[wasm_bindgen(js_name = updateFrameTextureRgba)]
    pub fn update_frame_texture_rgba(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(), JsValue> {
        self.require_open()?;
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|wh| wh.checked_mul(4));
        if width == 0 || height == 0 || expected != Some(rgba.len()) {
            return Err(js_error(format!(
                "frame texture rgba must be width*height*4 bytes \
                 (got {} for {width}x{height})",
                rgba.len()
            )));
        }
        // Building the renderer needs the leading mesh table already decoded
        // (the protocol is mesh-first; the mesh renderer requires ≥1 mesh).
        if !self.input.has_meshes() {
            return Err(js_error(
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
        self.external_frame_ready = true;
        Ok(())
    }
}

impl CanvasRenderer {
    fn require_open(&self) -> Result<(), JsValue> {
        match self.state {
            CanvasState::Open => Ok(()),
            CanvasState::Finished => Err(js_error("CanvasRenderer is finished")),
            CanvasState::Failed => Err(js_error("CanvasRenderer is failed")),
        }
    }

    /// Renders a single decoded frame to the surface: resolves its draw list
    /// (defaulting to one instance of mesh `0` for a legacy single-object frame),
    /// validates the mesh ids, builds the scene with the current flags + optional
    /// background compositing, then encodes/submits/presents. Shared by
    /// [`push_ipc`](Self::push_ipc) (immediate) and
    /// [`render_index`](Self::render_index) (buffered replay).
    fn render_frame(&mut self, frame: &DecodedFrame) -> Result<(), JsValue> {
        // The protocol is mesh-first: without a leading mesh table there is
        // nothing to draw (and the mesh renderer requires ≥1 mesh).
        if !self.input.has_meshes() {
            return Err(js_error(
                "input is missing the required leading mesh table (protocol is mesh-first)",
            ));
        }
        let params = frame.params;
        let has_inline_frame = self.upload_inline_frame(frame.frame_id)?;
        let has_external_frame = std::mem::take(&mut self.external_frame_ready);
        // Explicit wire draw list ⇒ drawn verbatim (an empty list ⇒ background
        // only); an absent draw list ⇒ one instance of mesh 0 placed by the
        // frame's own model (legacy single-object behavior).
        let draws: Vec<Draw> = frame.resolved_draws();
        let mesh_count = self.ensure_renderer().mesh_count();
        for draw in &draws {
            if draw.mesh_id as usize >= mesh_count {
                return Err(js_error(format!(
                    "draw references mesh {} but only {mesh_count} mesh(es) are loaded",
                    draw.mesh_id
                )));
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
            (has_inline_frame || (self.composite_frame && has_external_frame))
                .then_some(FrameFit::Stretch),
        );

        measure("trd.canvas.render-submit", || {
            let acquired = self.acquire_frame()?;
            let renderer = self.renderer.as_mut().expect("renderer built above");
            self.target.present(
                &self.device,
                &self.queue,
                renderer,
                acquired.texture,
                params,
                &scene,
            );
            if acquired.reconfigure_after_present {
                self.target.reconfigure(&self.device);
            }
            Ok(())
        })
    }

    fn upload_inline_frame(&mut self, frame_id: Option<u32>) -> Result<bool, JsValue> {
        let Some(frame_id) = frame_id else {
            self.last_inline_frame_id = None;
            return Ok(false);
        };
        self.external_frame_ready = false;
        if self.last_inline_frame_id == Some(frame_id) {
            return Ok(true);
        }
        let image = self
            .input
            .frames()
            .get(frame_id as usize)
            .ok_or_else(|| js_error(format!("frame_id {frame_id} is out of range")))?
            .decode()
            .map_err(|error| js_error(format!("decode frame_id {frame_id}: {error}")))?;
        self.ensure_renderer();
        let queue = &self.queue;
        self.renderer
            .as_mut()
            .expect("renderer built above")
            .update_frame_texture_rgba(queue, &image.rgba, image.width, image.height);
        self.last_inline_frame_id = Some(frame_id);
        Ok(true)
    }

    /// Lazily builds the mesh renderer on first use. The protocol is mesh-first,
    /// so the session always carries a leading mesh table by the time frames are
    /// produced; builds a multi-mesh renderer with each mesh's
    /// [`preview_transform`](trd_core::Mesh::preview_transform) base model,
    /// targeting the surface's sRGB view format.
    fn ensure_renderer(&mut self) -> &mut MeshRenderer {
        if self.renderer.is_none() {
            let meshes = self.input.meshes();
            let renderer = MeshRenderer::auto_fit(&self.device, self.target.view_format(), meshes);
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
        self.renderer.as_mut().expect("renderer just built")
    }

    fn acquire_frame(&mut self) -> Result<AcquiredFrame, JsValue> {
        match self.target.acquire() {
            wgpu::CurrentSurfaceTexture::Success(texture) => Ok(AcquiredFrame {
                texture,
                reconfigure_after_present: false,
            }),
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => Ok(AcquiredFrame {
                texture,
                reconfigure_after_present: true,
            }),
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.target.reconfigure(&self.device);
                self.acquire_after_recovery("reconfiguration")
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                let surface = self
                    .instance
                    .create_surface(wgpu::SurfaceTarget::Canvas(self.canvas.clone()))
                    .map_err(|error| js_error(format!("surface recreation failed: {error}")))?;
                self.target.replace_surface(&self.device, surface);
                self.acquire_after_recovery("recreation")
            }
            wgpu::CurrentSurfaceTexture::Timeout => Err(js_error("surface acquisition timed out")),
            wgpu::CurrentSurfaceTexture::Occluded => Err(js_error("surface is occluded")),
            wgpu::CurrentSurfaceTexture::Validation => Err(js_error("surface validation failed")),
        }
    }

    fn acquire_after_recovery(&self, recovery: &str) -> Result<AcquiredFrame, JsValue> {
        match self.target.acquire() {
            wgpu::CurrentSurfaceTexture::Success(texture) => Ok(AcquiredFrame {
                texture,
                reconfigure_after_present: false,
            }),
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => Ok(AcquiredFrame {
                texture,
                reconfigure_after_present: true,
            }),
            wgpu::CurrentSurfaceTexture::Timeout => Err(js_error(format!(
                "surface acquisition timed out after {recovery}"
            ))),
            wgpu::CurrentSurfaceTexture::Occluded => {
                Err(js_error(format!("surface is occluded after {recovery}")))
            }
            wgpu::CurrentSurfaceTexture::Outdated => Err(js_error(format!(
                "surface remains outdated after {recovery}"
            ))),
            wgpu::CurrentSurfaceTexture::Lost => {
                Err(js_error(format!("surface remains lost after {recovery}")))
            }
            wgpu::CurrentSurfaceTexture::Validation => Err(js_error(format!(
                "surface validation failed after {recovery}"
            ))),
        }
    }
}

fn measure<T>(name: &str, work: impl FnOnce() -> Result<T, JsValue>) -> Result<T, JsValue> {
    let performance = web_sys::window()
        .and_then(|window| window.performance())
        .ok_or_else(|| js_error("Performance API is unavailable"))?;
    let start = format!("{name}:start");
    let end = format!("{name}:end");

    performance.mark(&start)?;
    // Record the measure regardless of whether `work` succeeded, so the `:start`
    // mark is never leaked on an error path. The measure itself is only emitted
    // on success; on error we still clear both scratch marks before returning.
    let outcome = work().and_then(|value| {
        performance.mark(&end)?;
        performance.measure_with_start_mark_and_end_mark(name, &start, &end)?;
        Ok(value)
    });
    performance.clear_marks_with_mark_name(&start);
    performance.clear_marks_with_mark_name(&end);
    outcome
}
