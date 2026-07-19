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
    /// Playback frame rate (frames per second): sets both the animation speed
    /// (higher = faster) and the present rate. When omitted, the stream's
    /// declared rate (`trd.stream.frame_rate` metadata, default 30) is used.
    #[arg(long)]
    fps: Option<f64>,
    /// Play the stream once and hold the last frame instead of looping.
    #[arg(long)]
    once: bool,
    /// Lock presentation to the monitor refresh (vsync). By default the app
    /// presents at `--fps` decoupled from the refresh rate (non-vsync).
    #[arg(long)]
    vsync: bool,
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
    async fn new(window: Arc<Window>, vsync: bool) -> Result<Self, AppError> {
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

        trd_core::render_triangle(
            &self.device,
            &self.queue,
            &view,
            self.config.format,
            params,
            self.config.width,
            self.config.height,
        );

        self.queue.present(frame);
    }
}

/// A message from the stdin reader thread: the stream's declared playback rate
/// (sent once, before any frames) followed by decoded frames.
enum StreamMsg {
    Rate(f64),
    // Boxed: `FrameParams` is large (camera columns), so an unboxed variant
    // would dwarf `Rate` (clippy::large_enum_variant).
    Frame(Box<FrameParams>),
}

/// The winit application: owns the GPU state and drives stream playback.
struct App {
    gpu: Option<Gpu>,
    /// Rate + frames arriving from the stdin reader thread.
    rx: Receiver<StreamMsg>,
    /// Every frame received so far, retained so playback can loop.
    frames: Vec<FrameParams>,
    /// The frame currently on screen (identity until the first arrives).
    current: FrameParams,
    /// Explicit `--fps` override; when `None`, `stream_rate` drives playback.
    rate_override: Option<f64>,
    /// The stream's declared playback rate (fps); `DEFAULT_FRAME_RATE` until the
    /// reader reports the schema metadata.
    stream_rate: f64,
    /// Wall-clock origin for playback, set when the first frame arrives.
    playback_start: Option<Instant>,
    /// Index of the frame currently shown, to detect when it changes.
    shown_index: Option<usize>,
    /// Restart from the first frame once the stream ends.
    loop_playback: bool,
    /// The reader thread has closed the channel (stream fully consumed).
    stream_done: bool,
    /// Initial window size in logical pixels.
    window_size: (u32, u32),
    /// Whether to lock presentation to the monitor refresh (vsync / Fifo).
    vsync: bool,
}

impl App {
    fn new(
        rx: Receiver<StreamMsg>,
        window_size: (u32, u32),
        rate_override: Option<f64>,
        loop_playback: bool,
        vsync: bool,
    ) -> Self {
        Self {
            gpu: None,
            rx,
            frames: Vec::new(),
            current: FrameParams::IDENTITY,
            rate_override,
            stream_rate: trd_core::DEFAULT_FRAME_RATE,
            playback_start: None,
            shown_index: None,
            loop_playback,
            stream_done: false,
            window_size,
            vsync,
        }
    }

    /// The effective playback rate (fps): the `--fps` override, else the stream's
    /// declared rate.
    fn rate(&self) -> f64 {
        self.rate_override.unwrap_or(self.stream_rate)
    }

    /// Drains the reader channel into the playback buffer and stream rate.
    fn drain_stream(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(StreamMsg::Rate(rate)) => self.stream_rate = rate,
                Ok(StreamMsg::Frame(params)) => self.frames.push(*params),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.stream_done = true;
                    break;
                }
            }
        }
    }

    /// Picks the frame for the current wall-clock time and updates `current`,
    /// returning whether the shown frame changed. Speed is `rate()` frames/sec
    /// regardless of how often this is called (i.e. independent of present fps).
    fn advance(&mut self) -> bool {
        if self.frames.is_empty() {
            return false;
        }
        let start = *self.playback_start.get_or_insert_with(Instant::now);
        let elapsed = start.elapsed().as_secs_f64();
        let target = (elapsed * self.rate()).floor().max(0.0) as usize;

        let index = if self.loop_playback {
            // Only loop across a length known to be complete once the stream ends;
            // while still streaming, clamp to what has arrived.
            if self.stream_done {
                target % self.frames.len()
            } else {
                target.min(self.frames.len() - 1)
            }
        } else {
            target.min(self.frames.len() - 1)
        };

        if self.shown_index == Some(index) {
            return false;
        }
        self.current = self.frames[index];
        self.shown_index = Some(index);
        true
    }

    /// The instant the next frame boundary is due, for scheduling a wakeup.
    fn next_boundary(&self) -> Option<Instant> {
        let start = self.playback_start?;
        let next = self.shown_index.map_or(0, |i| i + 1) as f64;
        Some(start + Duration::from_secs_f64(next / self.rate()))
    }

    /// True once the stream is finished and there is nothing left to play (a
    /// non-looping stream whose last frame is already shown).
    fn idle(&self) -> bool {
        self.stream_done
            && !self.loop_playback
            && self
                .shown_index
                .map_or(self.frames.is_empty(), |i| i + 1 >= self.frames.len())
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

        match pollster::block_on(Gpu::new(window.clone(), self.vsync)) {
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

        // Select the frame for the current wall-clock time (speed = rate()),
        // independent of how often this runs or the present/refresh rate.
        if self.advance() {
            if let Some(gpu) = self.gpu.as_ref() {
                gpu.window.request_redraw();
            }
        }

        // Sleep until the next frame boundary is due; go fully idle once a
        // non-looping stream has shown its last frame.
        if self.idle() {
            event_loop.set_control_flow(ControlFlow::Wait);
        } else if let Some(next) = self.next_boundary() {
            event_loop.set_control_flow(ControlFlow::WaitUntil(next));
        } else {
            // No frames yet: wait for the reader to deliver some.
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }
}

/// Reads the Arrow IPC frame-params stream from stdin on a background thread,
/// forwarding the stream's declared playback rate then each decoded frame over
/// `tx` until the stream ends.
fn spawn_stdin_reader(tx: mpsc::Sender<StreamMsg>) {
    let spawned = std::thread::Builder::new()
        .name("trd-stdin-reader".to_string())
        .spawn(move || {
            let stdin = std::io::stdin().lock();
            let rate_tx = tx.clone();
            // A send error just means the window closed; stop reading in that case.
            if let Err(err) = trd_core::read_frame_stream_with_meta(
                stdin,
                |rate| {
                    let _ = rate_tx.send(StreamMsg::Rate(rate));
                },
                |params| {
                    let _ = tx.send(StreamMsg::Frame(Box::new(params)));
                },
            ) {
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
    let rate_override = cli.fps.filter(|fps| fps.is_finite() && *fps > 0.0);

    let (tx, rx) = mpsc::channel();
    spawn_stdin_reader(tx);

    let event_loop = EventLoop::new()?;
    // Playback is paced with `ControlFlow::WaitUntil` in `about_to_wait`; start
    // by waiting until the app schedules the first frame.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::new(
        rx,
        (cli.width, cli.height),
        rate_override,
        !cli.once,
        cli.vsync,
    );
    event_loop.run_app(&mut app)?;
    Ok(())
}
