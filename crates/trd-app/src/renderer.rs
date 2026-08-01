//! GPU resources tied to a live window surface, plus the per-frame render path
//! driving the shared [`trd_core::MeshRenderer`].

use std::sync::Arc;

use trd_core::{
    build_scene, EnvMapData, FrameFit, GridPlane, ImageTexture, Mesh, MeshRenderer, OnscreenTarget,
    PbrMaterial, RenderMode,
};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::error::AppError;
use crate::stream::FrameData;

/// GPU resources tied to a live window surface.
pub(crate) struct WindowRenderer {
    pub(crate) window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The shared on-screen render harness (surface + config + sRGB view, #103).
    target: OnscreenTarget,
    /// The scene renderer, built lazily once the stream's mesh table (or the
    /// legacy built-in fallback) has arrived from the reader thread.
    pub(crate) renderer: Option<MeshRenderer>,
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
        let trd_core::GpuContext {
            adapter,
            device,
            queue,
        } = trd_core::GpuContext::request(
            &instance,
            &trd_core::GpuRequest {
                label: "trd app device",
                compatible_surface: Some(&surface),
                ..Default::default()
            },
        )
        .await?;

        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or(AppError::SurfaceUnsupported)?;
        // `--fps` sets the real playback/present rate, so by default we do NOT
        // lock presentation to the monitor's refresh (vsync). Pick a non-vsync
        // present mode when available (Mailbox is tear-free; Immediate may tear)
        // so the app can present above/below the refresh rate; `--vsync` forces
        // Fifo. Fifo is always supported, so it is the final fallback.
        let supported = surface.get_capabilities(&adapter).present_modes;
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
        let target = OnscreenTarget::new(&device, surface, config);

        Ok(Self {
            window,
            device,
            queue,
            target,
            renderer: None,
        })
    }

    pub(crate) fn resize(&mut self, size: PhysicalSize<u32>) {
        self.target.resize(&self.device, size.width, size.height);
    }

    /// Uploads the stream's meshes and builds the scene renderer (each mesh
    /// centered + scaled to fit via its preview base model). Idempotent per
    /// stream: called once when the mesh table first arrives.
    pub(crate) fn set_meshes(&mut self, meshes: &[Mesh]) {
        self.renderer = Some(MeshRenderer::auto_fit(
            &self.device,
            self.target.view_format(),
            meshes,
        ));
    }

    /// Binds `texture` as the albedo sampled by [`RenderMode::Textured`] meshes
    /// (`0.0.4`). No-op until the renderer is built; re-uploaded lazily on the
    /// next `render`.
    pub(crate) fn set_texture(&mut self, texture: &ImageTexture) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_texture(texture);
        }
    }

    /// Sets the Disney PBR material applied to [`RenderMode::Pbr`] meshes. No-op
    /// until the renderer is built.
    pub(crate) fn set_pbr_material(&mut self, material: PbrMaterial) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_pbr_material(material);
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render(
        &mut self,
        frame: Option<&FrameData>,
        mode: RenderMode,
        show_aabb: bool,
        show_axes: bool,
        show_local_axes: bool,
        show_local_grid: Option<GridPlane>,
        show_local_grid_mesh: Option<u32>,
    ) {
        let (Some(renderer), Some(frame)) = (self.renderer.as_mut(), frame) else {
            return;
        };

        let texture = match self.target.acquire() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            // The surface config is stale (e.g. after a resize/minimise or a lost
            // surface); reconfigure and try again on the next redraw.
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.target.reconfigure(&self.device);
                self.window.request_redraw();
                return;
            }
            // Transient (timeout/occluded/other): skip this frame.
            _ => return,
        };

        // Author the frame's Scene from its draw list + the render mode/overlay
        // flags, then hand it to the shared MeshRenderer — the same Scene the
        // headless CLI and wasm front-ends build. A per-frame background image
        // (#63) is uploaded first, then composited beneath the scene.
        let frame_fit = frame.frame_image.as_ref().map(|img| {
            renderer.update_frame_texture_rgba(&self.queue, &img.rgba, img.width, img.height);
            FrameFit::Stretch
        });
        let scene = build_scene(
            &frame.draws,
            mode,
            show_aabb,
            show_axes,
            show_local_axes,
            show_local_grid,
            show_local_grid_mesh,
            frame_fit,
        );
        self.target.present(
            &self.device,
            &self.queue,
            renderer,
            texture,
            frame.params,
            &scene,
        );
    }
}
