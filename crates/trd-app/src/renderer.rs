//! GPU resources tied to a live window surface, plus the per-frame render path
//! driving the shared [`trd_core::MeshRenderer`].

use std::sync::Arc;

use trd_core::{
    build_scene, EnvMapData, FrameFit, GridPlane, ImageTexture, Mesh, MeshRenderer, PbrMaterial,
    RenderMode, Viewport,
};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::error::AppError;
use crate::stream::FrameData;

/// GPU resources tied to a live window surface.
pub(crate) struct WindowRenderer {
    pub(crate) window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
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
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await?;

        let info = adapter.get_info();
        log::info!(
            "using {:?} adapter \"{}\" ({:?}), driver: {}",
            info.backend,
            info.name,
            info.device_type,
            info.driver_info
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("trd app device"),
                required_features: wgpu::Features::empty(),
                // Use the adapter's real limits so a large / high-DPI window
                // surface fits (downlevel_defaults caps textures at 2048).
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
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
        surface.configure(&device, &config);

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            renderer: None,
        })
    }

    pub(crate) fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Uploads the stream's meshes and builds the scene renderer (each mesh
    /// centered + scaled to fit via its preview base model). Idempotent per
    /// stream: called once when the mesh table first arrives.
    pub(crate) fn set_meshes(&mut self, meshes: &[Mesh]) {
        self.renderer = Some(MeshRenderer::auto_fit(
            &self.device,
            self.config.format,
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
    pub(crate) fn render(
        &mut self,
        frame: Option<&FrameData>,
        mode: RenderMode,
        show_aabb: bool,
        show_axes: bool,
        show_local_axes: bool,
        show_local_grid: Option<GridPlane>,
    ) {
        let (Some(renderer), Some(frame)) = (self.renderer.as_mut(), frame) else {
            return;
        };

        let surface = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface)
            | wgpu::CurrentSurfaceTexture::Suboptimal(surface) => surface,
            // The surface config is stale (e.g. after a resize/minimise or a lost
            // surface); reconfigure and try again on the next redraw.
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                self.window.request_redraw();
                return;
            }
            // Transient (timeout/occluded/other): skip this frame.
            _ => return,
        };

        let view = surface
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

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
            frame_fit,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trd app frame"),
            });
        renderer.encode(
            &self.queue,
            &mut encoder,
            &view,
            frame.params,
            &scene,
            Viewport {
                width: self.config.width,
                height: self.config.height,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        self.queue.present(surface);
    }
}
