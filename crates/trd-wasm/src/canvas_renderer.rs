use trd_core::{
    DisneyMaterial, EnvMapData, FrameFit, ImageBasedLighting, Lighting, RenderError, RenderMode,
    RenderOptions, RenderTarget, Renderer, Scene, SurfaceRepair, SurfaceTarget, ToneMapping,
    Tonemap,
};
use wasm_bindgen::prelude::*;

use crate::{js_error, PbrState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CanvasState {
    Open,
    Finished,
}

#[wasm_bindgen]
pub struct CanvasRenderer {
    instance: wgpu::Instance,
    canvas: web_sys::HtmlCanvasElement,
    gpu: std::sync::Arc<trd_core::GpuContext>,
    /// The canvas surface, owned by the shell so it can be resized or recovered
    /// before the render harness exists.
    target: RenderTarget,
    /// Built when a complete params/GLB document is loaded.
    renderer: Option<Renderer>,
    /// Draw mode + overlay toggles; [`Scene::from_draws`] turns it into a scene.
    options: RenderOptions,
    /// Draw the uploaded background texture beneath the scene as the scene's
    /// frame plane. A no-op until a background is uploaded.
    composite_frame: bool,
    /// Staged until the renderer is built. `None` ⇒ the renderer's defaults.
    pbr: Option<PbrState>,
    /// Equirectangular HDR probe reflected by [`RenderMode::Shaded`] draws.
    env_map: Option<EnvMapData>,
    /// Sky blur; `None` ⇒ no sky. Re-derived into `options.env_background`
    /// together with the tone mapping, since the sky shares its exposure.
    env_background_blur: Option<f32>,
    document: Option<std::rc::Rc<std::cell::RefCell<trd_core::SceneDocument>>>,
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
        let gpu = trd_core::GpuContext::request(
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
            .get_default_config(&gpu.adapter, width, height)
            .ok_or_else(|| js_error("surface is unsupported by the selected adapter"))?;
        // The browser's preferred canvas format is non-sRGB, so a pipeline
        // targeting it directly would write linear values and look darker than
        // the CLI. Rendering through an sRGB view of the surface matches it.
        let target = RenderTarget::surface(SurfaceTarget::new(&gpu.device, surface, config));

        Ok(Self {
            renderer: None,
            options: RenderOptions::default(),
            composite_frame: false,
            pbr: None,
            env_map: None,
            env_background_blur: None,
            instance,
            canvas,
            gpu,
            target,
            document: None,
            external_frame_ready: false,
            state: CanvasState::Open,
        })
    }

    /// Loads a complete current params/GLB document, never legacy incremental IPC.
    #[wasm_bindgen(js_name = loadIpc)]
    pub fn load_ipc(&mut self, chunk: &[u8]) -> Result<u32, JsValue> {
        self.require_open()?;
        let document = crate::ArrowSceneDocument::from_arrow(chunk)?;
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
        self.require_open()?;
        let source = document.inner.borrow();
        let count = u32::try_from(source.row_count()).map_err(|_| js_error("too many rows"))?;
        source.frames().map_err(js_error)?;
        let renderer = crate::scene_document::document_renderer(
            &source,
            self.gpu.clone(),
            self.target.view_format(),
            self.pbr.as_ref(),
            self.env_map.as_ref(),
        )?;
        self.renderer = Some(renderer);
        self.document = Some(std::rc::Rc::clone(&document.inner));
        self.external_frame_ready = false;
        drop(source);
        self.refresh_env_background();
        Ok(count)
    }

    #[wasm_bindgen(js_name = meshResourceCount)]
    pub fn mesh_resource_count(&self) -> u32 {
        if let Some(document) = &self.document {
            return u32::try_from(document.borrow().meshes().len()).unwrap_or(u32::MAX);
        }
        0
    }

    /// The frame's external background reference, which the JS shell resolves to
    /// RGBA and uploads before rendering. A document row without a background
    /// returns `None`; an invalid document row returns an error.
    #[wasm_bindgen(js_name = frameRef)]
    pub fn frame_ref(&self, index: u32) -> Result<Option<String>, JsValue> {
        if let Some(document) = &self.document {
            return document
                .borrow()
                .frame_ref(index as usize)
                .map_err(js_error);
        }
        Err(js_error("load a params/GLB document first"))
    }

    /// Renders one buffered frame (by index) to the surface using the current
    /// flags and any uploaded background — the generic viewer's per-tick call.
    #[wasm_bindgen(js_name = renderIndex)]
    pub fn render_index(&mut self, index: u32) -> Result<(), JsValue> {
        self.require_open()?;
        if let Some(document) = self.document.clone() {
            let (camera, scene) = {
                let source = document.borrow();
                let frame = source.frame(index as usize).map_err(js_error)?;
                let fit = (self.composite_frame && self.external_frame_ready)
                    .then_some(FrameFit::Stretch);
                trd_placement::document_scene(
                    &source,
                    &frame,
                    self.target.viewport(),
                    &self.options,
                    fit,
                )
                .map_err(js_error)?
            };
            self.external_frame_ready = false;
            let scene = scene.with_lighting(
                self.pbr
                    .as_ref()
                    .map(PbrState::lighting)
                    .unwrap_or_default(),
            );
            return measure("trd.canvas.render-submit", || {
                self.present_camera(camera, &scene)
            });
        }
        Err(js_error("load a params/GLB document first"))
    }

    pub fn finish(&mut self) -> Result<(), JsValue> {
        self.require_open()?;
        self.state = CanvasState::Finished;
        Ok(())
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

    /// Textured (`true`) samples each GLB material's albedo at each vertex UV,
    /// falling back to a 1×1 white; `false` uses per-vertex color.
    #[wasm_bindgen(js_name = setTextured)]
    pub fn set_textured(&mut self, enabled: bool) {
        self.options.mode = if enabled {
            RenderMode::Textured
        } else {
            RenderMode::Filled
        };
    }

    /// Selects the Disney principled-BRDF (`true`) or per-vertex color (`false`)
    /// mesh path — the browser twin of the native `--pbr` flag.
    #[wasm_bindgen(js_name = setPbr)]
    pub fn set_pbr(&mut self, enabled: bool) {
        self.options.mode = if enabled {
            RenderMode::Shaded
        } else {
            RenderMode::Filled
        };
    }

    /// The Disney material for every [`RenderMode::Shaded`] draw. `tonemap` is
    /// `"aces"` or anything else for Reinhard; unlisted Disney parameters keep
    /// their defaults.
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

    /// Draws the bound HDR probe as the background sky. `blur` runs `0.0` (sharp)
    /// to `1.0`. Exposure and tone-map follow the PBR material, so sky and
    /// objects cannot be tone-mapped differently; with no probe the sky is black.
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

    /// Decodes an equirectangular Radiance `.hdr` buffer (downscaled to 2048px)
    /// and binds it as the environment probe.
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

    /// Toggles the per-draw **local** coordinate-axes gizmo for later frames — one
    /// [`CoordinateAxes`](trd_core::Primitive::CoordinateAxes) gizmo at each object's own `model` (its
    /// reconstructed local frame, e.g. #77's quad basis). The browser twin of the
    /// native `--axes-local` flag.
    #[wasm_bindgen(js_name = setShowLocalAxes)]
    pub fn set_show_local_axes(&mut self, enabled: bool) {
        self.options.overlays.local_axes = enabled;
    }

    /// Toggles compositing the uploaded background frame beneath the scene as the
    /// scene's [`Background::frame`](trd_core::Background::frame) plane (#63).
    /// The browser twin of the native
    /// `--frames-base` compositing: enable it, then push one background per frame
    /// via [`update_frame_texture_rgba`](Self::update_frame_texture_rgba) *before*
    /// that frame's [`render_index`](Self::render_index). Has no visible effect until a
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
        self.ensure_renderer()?
            .update_frame_texture_rgba(rgba, width, height);
        self.external_frame_ready = true;
        Ok(())
    }
}

impl CanvasRenderer {
    fn require_open(&self) -> Result<(), JsValue> {
        match self.state {
            CanvasState::Open => Ok(()),
            CanvasState::Finished => Err(js_error("CanvasRenderer is finished")),
        }
    }

    /// Presents one frame, recovering from a stale or lost surface **in-call**.
    ///
    /// The browser cannot defer to "the next redraw" the way the native window
    /// does — a `requestAnimationFrame` driver expects this call to have drawn —
    /// so a recoverable outcome is repaired here and the frame retried exactly
    /// once. That policy is the front-end's; the harness only reports what
    /// happened (#180).
    fn present_camera(&mut self, params: trd_core::Camera, scene: &Scene) -> Result<(), JsValue> {
        match self.present_once(params, scene) {
            // Presented. A repair (the surface no longer matches the canvas) is
            // applied now so the *next* frame is clean; this one is on screen.
            Ok(repair) => {
                if repair.is_some() {
                    self.reconfigure();
                }
                Ok(())
            }
            // Nothing was drawn. Repair, then retry exactly once — the browser
            // cannot defer to "the next redraw" the way the native window does.
            Err(RenderError::Surface(error)) => match error.repair() {
                Some(SurfaceRepair::Reconfigure) => {
                    self.reconfigure();
                    self.retry(params, scene, "reconfiguration")
                }
                Some(SurfaceRepair::Recreate) => {
                    let surface = self
                        .instance
                        .create_surface(wgpu::SurfaceTarget::Canvas(self.canvas.clone()))
                        .map_err(|error| js_error(format!("surface recreation failed: {error}")))?;
                    let device = self.gpu.device.clone();
                    if let Some(target) = self.target.as_surface_mut() {
                        Renderer::replace_surface(&device, target, surface);
                    }
                    self.retry(params, scene, "recreation")
                }
                // Transient: skipping the frame is the whole remedy, but the
                // caller drove this call expecting pixels, so report it.
                None => Err(js_error(error.to_string())),
            },
            // Not a surface problem (a malformed camera): nothing to repair.
            Err(error) => Err(js_error(error.to_string())),
        }
    }

    fn retry(
        &mut self,
        params: trd_core::Camera,
        scene: &Scene,
        recovery: &str,
    ) -> Result<(), JsValue> {
        match self.present_once(params, scene) {
            Ok(repair) => {
                if repair.is_some() {
                    self.reconfigure();
                }
                Ok(())
            }
            Err(error) => Err(js_error(format!(
                "surface still unusable after {recovery}: {error}"
            ))),
        }
    }

    /// Draws + presents one frame. **Synchronous**: presenting never awaits, and
    /// no `async fn` may cross the `wasm_bindgen` boundary.
    fn present_once(
        &mut self,
        camera: trd_core::Camera,
        scene: &Scene,
    ) -> Result<Option<SurfaceRepair>, RenderError> {
        let renderer = self
            .renderer
            .as_mut()
            .expect("renderer built before present");
        renderer.render(camera, scene, &mut self.target)
    }

    fn reconfigure(&mut self) {
        let device = self.gpu.device.clone();
        if let Some(target) = self.target.as_surface_mut() {
            Renderer::reconfigure_surface(&device, target);
        }
    }

    fn ensure_renderer(&mut self) -> Result<&mut Renderer, JsValue> {
        self.renderer
            .as_mut()
            .ok_or_else(|| js_error("load a params/GLB document first"))
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
