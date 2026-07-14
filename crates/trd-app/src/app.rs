//! Native windowed rendering: a winit application that owns a live wgpu surface
//! and drives [`trd_core::render_triangle`] each redraw.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

const INITIAL_WIDTH: u32 = 800;
const INITIAL_HEIGHT: u32 = 600;

/// Errors that can occur while setting up the window or GPU.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    /// The winit event loop could not be created.
    #[error("failed to create the event loop: {0}")]
    EventLoop(#[from] winit::error::EventLoopError),
    /// A wgpu surface could not be created from the window.
    #[error("failed to create a GPU surface: {0}")]
    CreateSurface(#[from] wgpu::CreateSurfaceError),
    /// No GPU adapter could satisfy the request.
    #[error("no suitable GPU adapter found: {0}")]
    RequestAdapter(#[from] wgpu::RequestAdapterError),
    /// The GPU device could not be created.
    #[error("failed to create GPU device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    /// The adapter does not support the window surface.
    #[error("the GPU adapter does not support the window surface")]
    SurfaceUnsupported,
}

/// GPU resources tied to a live window surface.
struct Gpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
}

impl Gpu {
    async fn new(window: Arc<Window>) -> Result<Self, AppError> {
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

        let config = surface
            .get_default_config(&adapter, width, height)
            .ok_or(AppError::SurfaceUnsupported)?;
        surface.configure(&device, &config);

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
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

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        trd_core::render_triangle(
            &self.device,
            &self.queue,
            &view,
            self.config.format,
            trd_core::FrameParams::IDENTITY,
        );

        self.queue.present(frame);
    }
}

/// The winit application: holds the GPU state once the window is created.
#[derive(Default)]
struct App {
    gpu: Option<Gpu>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("trd — hello triangle")
            .with_inner_size(LogicalSize::new(INITIAL_WIDTH, INITIAL_HEIGHT));

        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                log::error!("failed to create window: {err}");
                event_loop.exit();
                return;
            }
        };

        match pollster::block_on(Gpu::new(window.clone())) {
            Ok(gpu) => {
                gpu.window.request_redraw();
                self.gpu = Some(gpu);
            }
            Err(err) => {
                log::error!("failed to initialize GPU: {err}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                gpu.resize(size);
                gpu.window.request_redraw();
            }
            WindowEvent::RedrawRequested => gpu.render(),
            _ => {}
        }
    }
}

/// Runs the interactive window until it is closed.
pub fn run() -> Result<(), AppError> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,trd_app=info,trd_core=info"),
    )
    .init();

    let event_loop = EventLoop::new()?;
    // The hello-triangle is static, so wait for OS events instead of busy-looping.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}
