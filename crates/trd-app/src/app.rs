//! Native windowed streaming viewer: a winit application that owns a live wgpu
//! surface and renders trd frame-params streamed from stdin.
//!
//! A background thread reads the Arrow IPC frame-params stream (the same input
//! `trd-cli` consumes) via [`trd_core::read_frame_stream`] and forwards each
//! decoded [`trd_core::FrameParams`] over a channel. The window plays them at a
//! fixed rate, rendering each with the shared [`trd_core::render_triangle`] — so
//! all rendering logic still lives in `trd-core`.

use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use trd_core::FrameParams;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// Interactive desktop viewer for a trd frame-params stream (protocol 0.0.1).
///
/// Reads an Arrow IPC stream of `{center, size, theta}` rows on stdin and plays
/// them live in a window, e.g. `duckdb ... | trd-app`.
#[derive(Parser)]
#[command(name = "trd-app", version, about)]
struct Cli {
    /// Initial window width in logical pixels.
    #[arg(long, default_value_t = 800, value_parser = clap::value_parser!(u32).range(1..))]
    width: u32,
    /// Initial window height in logical pixels.
    #[arg(long, default_value_t = 600, value_parser = clap::value_parser!(u32).range(1..))]
    height: u32,
    /// Playback rate in frames per second.
    #[arg(long, default_value_t = 30.0)]
    fps: f32,
    /// Play the stream once and hold the last frame instead of looping.
    #[arg(long)]
    once: bool,
}

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

    fn render(&mut self, params: FrameParams) {
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

        trd_core::render_triangle(&self.device, &self.queue, &view, self.config.format, params);

        self.queue.present(frame);
    }
}

/// The winit application: owns the GPU state and drives stream playback.
struct App {
    gpu: Option<Gpu>,
    /// Decoded frames arriving from the stdin reader thread.
    rx: Receiver<FrameParams>,
    /// Every frame received so far, retained so playback can loop.
    frames: Vec<FrameParams>,
    /// Index of the next frame to display.
    cursor: usize,
    /// The frame currently on screen (identity until the first arrives).
    current: FrameParams,
    /// Time between displayed frames (the inverse of the target FPS).
    frame_interval: Duration,
    /// When the next frame should be shown.
    next_tick: Instant,
    /// Restart from the first frame once the stream ends.
    loop_playback: bool,
    /// The reader thread has closed the channel (stream fully consumed).
    stream_done: bool,
    /// Initial window size in logical pixels.
    window_size: (u32, u32),
}

impl App {
    fn new(
        rx: Receiver<FrameParams>,
        window_size: (u32, u32),
        frame_interval: Duration,
        loop_playback: bool,
    ) -> Self {
        Self {
            gpu: None,
            rx,
            frames: Vec::new(),
            cursor: 0,
            current: FrameParams::IDENTITY,
            frame_interval,
            next_tick: Instant::now(),
            loop_playback,
            stream_done: false,
            window_size,
        }
    }

    /// Moves all frames the reader thread has produced into the playback buffer.
    fn drain_stream(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(params) => self.frames.push(params),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.stream_done = true;
                    break;
                }
            }
        }
    }

    /// Advances playback by one frame, returning whether the shown frame changed.
    fn advance(&mut self) -> bool {
        if self.cursor < self.frames.len() {
            self.current = self.frames[self.cursor];
            self.cursor += 1;
            true
        } else if self.stream_done && self.loop_playback && !self.frames.is_empty() {
            // Caught up to the end of a finished stream: loop back to the start.
            self.current = self.frames[0];
            self.cursor = 1;
            true
        } else {
            // Waiting for more frames (or the stream ended without looping):
            // hold the current frame.
            false
        }
    }

    /// True once the stream is finished and there is nothing left to play.
    fn idle(&self) -> bool {
        self.stream_done
            && self.cursor >= self.frames.len()
            && !(self.loop_playback && !self.frames.is_empty())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("trd — stream viewer")
            .with_inner_size(LogicalSize::new(self.window_size.0, self.window_size.1));

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
            WindowEvent::RedrawRequested => gpu.render(self.current),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.drain_stream();

        let now = Instant::now();
        if now >= self.next_tick {
            if self.advance() {
                if let Some(gpu) = self.gpu.as_ref() {
                    gpu.window.request_redraw();
                }
            }
            // Schedule from `now` (not `next_tick`) so a stall never causes a
            // catch-up burst of frames.
            self.next_tick = now + self.frame_interval;
        }

        // Sleep until the next frame is due; go fully idle once playback is done.
        if self.idle() {
            event_loop.set_control_flow(ControlFlow::Wait);
        } else {
            event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_tick));
        }
    }
}

/// Reads the Arrow IPC frame-params stream from stdin on a background thread,
/// forwarding each decoded frame over `tx` until the stream ends.
fn spawn_stdin_reader(tx: mpsc::Sender<FrameParams>) {
    let spawned = std::thread::Builder::new()
        .name("trd-stdin-reader".to_string())
        .spawn(move || {
            let stdin = std::io::stdin().lock();
            // A send error just means the window closed; stop reading in that case.
            if let Err(err) = trd_core::read_frame_stream(stdin, |params| {
                let _ = tx.send(params);
            }) {
                log::error!("input stream error: {err}");
            }
        });
    if let Err(err) = spawned {
        log::error!("failed to spawn stdin reader thread: {err}");
    }
}

/// Runs the interactive stream viewer until the window is closed.
pub fn run() -> Result<(), AppError> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,trd_app=info,trd_core=info"),
    )
    .init();

    let cli = Cli::parse();
    let fps = if cli.fps.is_finite() && cli.fps > 0.0 {
        cli.fps
    } else {
        30.0
    };
    let frame_interval = Duration::from_secs_f32(1.0 / fps);

    let (tx, rx) = mpsc::channel();
    spawn_stdin_reader(tx);

    let event_loop = EventLoop::new()?;
    // Playback is paced with `ControlFlow::WaitUntil` in `about_to_wait`; start
    // by waiting until the app schedules the first frame.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::new(rx, (cli.width, cli.height), frame_interval, !cli.once);
    event_loop.run_app(&mut app)?;
    Ok(())
}
