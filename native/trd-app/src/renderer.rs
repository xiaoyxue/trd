//! The live window surface, plus the per-frame render path driving the shared
//! [`trd_core::Renderer`] harness.
//!
//! This shell is **not** a renderer: `trd-core`'s harness takes its render target
//! as a per-call argument rather than owning one (#203), so the only thing left
//! here is what is genuinely window-specific — creating the surface from a
//! `winit` window, owning + resizing it, and the platform's surface-recovery
//! policy (#180).

use std::sync::Arc;

use trd_core::{
    DisneyMaterial, EnvMapData, FrameFit, ImageBasedLighting, ImageData, ImageTexture, Lighting,
    Mesh, RenderError, RenderOptions, RenderTarget, Renderer, Scene, SurfaceTarget, ToneMapping,
};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::error::AppError;
use crate::stream::FrameData;

/// GPU resources tied to a live window surface.
pub(crate) struct WindowRenderer {
    pub(crate) window: Arc<Window>,
    /// Retained so a **lost** surface can be rebuilt for the same window: wgpu
    /// asks for recreation, not reconfiguration, and a new surface needs the
    /// instance that made the first one (#203).
    instance: wgpu::Instance,
    /// The shared GPU context (adapter + device + queue), held as one value
    /// instead of cloned apart into separate fields (#180).
    gpu: Arc<trd_core::GpuContext>,
    /// The window surface, owned directly by the shell (#203): the harness holds
    /// no target of its own, so this front-end owns the surface and decides how to
    /// repair it. Held as a [`RenderTarget`] because presenting is reachable only
    /// through the renderer's one `render` dispatcher — the surface itself carries
    /// no render behaviour.
    target: RenderTarget,
    /// The shared render harness, built lazily once the stream's mesh table has
    /// arrived from the reader thread.
    pub(crate) renderer: Option<Renderer>,
    /// The frame's light rig. Held here rather than pushed onto the harness:
    /// lighting is a `Scene` property now (#182), so it is attached to the scene
    /// this window builds each frame.
    lighting: Lighting,
    /// CPU image backing the currently uploaded frame-plane texture. Inline
    /// frame reuse preserves the same Arc, so repeated IDs skip GPU writes.
    uploaded_frame_image: Option<Arc<ImageData>>,
}

impl WindowRenderer {
    pub(crate) async fn new(window: Arc<Window>, vsync: bool) -> Result<Self, AppError> {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        // `new_without_display_handle_from_env` honours WGPU_BACKEND (e.g. `gl` on
        // WSL2), matching the headless CLI. An `Arc<Window>` supplies both the
        // window and display handles at surface creation, so the surface outlives
        // borrows and is `'static`.
        let instance = trd_core::create_instance();
        let surface = instance.create_surface(window.clone())?;

        // Route through the shared device/adapter helper: `HighPerformance` (the
        // default) — fixing the prior `PowerPreference::default()` that could bind
        // a weak iGPU/display GPU on a multi-GPU box, against the AGENTS.md rule —
        // plus the adapter's real limits (so a large / high-DPI surface fits;
        // downlevel_defaults caps textures at 2048) and the mandated adapter log
        // line. Only the surface creation above stays shell-specific.
        let gpu = trd_core::GpuContext::request(
            &instance,
            &trd_core::GpuRequest {
                label: "trd app device",
                compatible_surface: Some(&surface),
                ..Default::default()
            },
        )
        .await?;

        let mut config = surface
            .get_default_config(&gpu.adapter, width, height)
            .ok_or(AppError::SurfaceUnsupported)?;
        // `--fps` sets the real playback/present rate, so by default we do NOT
        // lock presentation to the monitor's refresh (vsync). Pick a non-vsync
        // present mode when available (Mailbox is tear-free; Immediate may tear)
        // so the app can present above/below the refresh rate; `--vsync` forces
        // Fifo. Fifo is always supported, so it is the final fallback.
        let supported = surface.get_capabilities(&gpu.adapter).present_modes;
        config.present_mode = if vsync {
            wgpu::PresentMode::Fifo
        } else if supported.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else if supported.contains(&wgpu::PresentMode::Immediate) {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::Fifo
        };
        log::info!("present mode: {:?} (vsync={vsync})", config.present_mode);
        let target = RenderTarget::surface(SurfaceTarget::new(&gpu.device, surface, config));

        Ok(Self {
            window,
            instance,
            gpu,
            target,
            renderer: None,
            lighting: Lighting::default(),
            uploaded_frame_image: None,
        })
    }

    pub(crate) fn resize(&mut self, size: PhysicalSize<u32>) {
        // An associated `Renderer` function, not a method: a window is resized
        // long before the stream's mesh table has arrived, so there may be no
        // harness yet — but the surface still has to be reconfigured (#203).
        if let Some(surface) = self.target.as_surface_mut() {
            Renderer::resize_surface(&self.gpu.device, surface, size.width, size.height);
        }
    }

    /// Uploads the stream's meshes and builds the scene renderer (each mesh
    /// centered + scaled to fit via its preview base model). Idempotent per
    /// stream: called once when the mesh table first arrives.
    ///
    /// Reports an unusable mesh set (e.g. a stream whose mesh table decoded to
    /// nothing) rather than aborting the window on it (#235 R8) — the mesh set
    /// comes from the wire, so it is input, not a bug.
    pub(crate) fn set_meshes(&mut self, meshes: &[Mesh]) -> Result<(), RenderError> {
        self.renderer = Some(Renderer::auto_fit(
            self.gpu.clone(),
            self.target.view_format(),
            meshes,
        )?);
        self.uploaded_frame_image = None;
        Ok(())
    }

    /// Binds `texture` as the albedo sampled by [`RenderMode::Textured`] meshes
    /// (`0.0.4`). No-op until the renderer is built; re-uploaded lazily on the
    /// next `render`.
    pub(crate) fn set_texture(&mut self, texture: &ImageTexture) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_texture(texture);
        }
    }

    /// Sets the Disney PBR material applied to [`RenderMode::Shaded`] meshes. No-op
    /// until the renderer is built.
    pub(crate) fn set_disney_material(&mut self, material: DisneyMaterial) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_disney_material(material);
        }
    }

    pub(crate) fn set_lighting(&mut self, lighting: Lighting) {
        self.lighting = lighting;
    }

    pub(crate) fn set_image_based_lighting(&mut self, ibl: ImageBasedLighting) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_image_based_lighting(ibl);
        }
    }

    pub(crate) fn set_tone_mapping(&mut self, tone_mapping: ToneMapping) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_tone_mapping(tone_mapping);
        }
    }

    /// Binds the HDR environment probe reflected by PBR meshes. No-op until the
    /// renderer is built; re-uploaded lazily on the next `render`.
    pub(crate) fn set_env_map(&mut self, env: EnvMapData) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_env_map(env);
        }
    }

    /// Renders one frame's [`Scene`](trd_core::Scene) to the window surface.
    /// No-op until the renderer is built and a frame is available.
    pub(crate) fn render(&mut self, frame: Option<&FrameData>, options: &RenderOptions) {
        let (Some(renderer), Some(frame)) = (self.renderer.as_mut(), frame) else {
            return;
        };

        // Author the frame's Scene from its draw list + the render mode/overlay
        // flags, then hand it to the shared harness — the same Scene, built by the
        // same `Scene::from_draws`, that the headless CLI and the wasm front-ends
        // render. A per-frame background image (#63) is uploaded first, then
        // composited beneath the scene.
        let frame_fit = match frame.frame_image.as_ref() {
            Some(image) => {
                let already_uploaded = self
                    .uploaded_frame_image
                    .as_ref()
                    .is_some_and(|current| Arc::ptr_eq(current, image));
                if !already_uploaded {
                    renderer.update_frame_texture_rgba(&image.rgba, image.width, image.height);
                    self.uploaded_frame_image = Some(image.clone());
                }
                Some(FrameFit::Stretch)
            }
            None => {
                self.uploaded_frame_image = None;
                None
            }
        };
        let scene =
            Scene::from_draws(&frame.draws, options, frame_fit).with_lighting(self.lighting);

        let camera = match frame.params.to_camera(self.target.viewport()) {
            Ok(camera) => camera,
            Err(error) => {
                log::warn!("skipping frame with a malformed camera: {error}");
                return;
            }
        };
        // The recovery policy is the window's, not the harness's (#180): repair
        // the surface and defer to the next redraw. A frame that *was* presented
        // needs no redraw, only the repair.
        let (repair, redraw) = match renderer.render(camera, &scene, &mut self.target) {
            Ok(repair) => (repair, false),
            Err(trd_core::RenderError::Surface(error)) => (error.repair(), true),
            Err(error) => {
                log::warn!("skipping frame: {error}");
                (None, false)
            }
        };
        if let Some(repair) = repair {
            self.repair_surface(repair);
        }
        if redraw {
            self.window.request_redraw();
        }
    }

    /// Applies the repair a failed or suboptimal present asked for.
    ///
    /// `Recreate` really does build a new surface: wgpu reports a *lost* surface
    /// as unusable, and merely reconfiguring it leaves the window blank (#203).
    fn repair_surface(&mut self, repair: trd_core::SurfaceRepair) {
        let device = self.gpu.device.clone();
        let recreated = match repair {
            trd_core::SurfaceRepair::Reconfigure => None,
            trd_core::SurfaceRepair::Recreate => {
                match self.instance.create_surface(self.window.clone()) {
                    Ok(surface) => Some(surface),
                    Err(error) => {
                        log::error!("could not recreate the lost surface: {error}");
                        return;
                    }
                }
            }
        };
        let Some(target) = self.target.as_surface_mut() else {
            return;
        };
        match recreated {
            None => Renderer::reconfigure_surface(&device, target),
            Some(surface) => Renderer::replace_surface(&device, target, surface),
        }
    }
}
