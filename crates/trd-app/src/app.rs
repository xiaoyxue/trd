//! Native windowed streaming viewer: a winit application that owns a live wgpu
//! surface and renders trd frame-params streamed from stdin.
//!
//! A background thread reads the Arrow IPC stream (the same `[mesh][params]`
//! input `trd-cli` consumes) via [`trd_core::read_scene_stream_with_meta`],
//! forwarding the decoded mesh table then each frame's params + instanced draw
//! list over a channel. The window plays them at a fixed rate, encoding each
//! frame's [`trd_core::Scene`] with the shared [`trd_core::MeshRenderer`] — so
//! all rendering logic still lives in `trd-core`.

use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use std::path::PathBuf;
use trd_core::{
    build_scene, read_scene_stream_with_meta, Draw, FrameFit, FrameParams, ImageData, ImageTexture,
    Mesh, MeshRenderer, RenderMode, Viewport,
};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

/// Interactive desktop viewer for a trd scene stream (protocol 0.0.3).
///
/// Reads the Arrow IPC `[mesh][params]` stream on stdin — a leading mesh table
/// then per-frame params + instanced draw lists (or a legacy `0.0.1`/`0.0.2`
/// params-only stream → the built-in hello-triangle) — and plays it live in a
/// window, e.g. `trd-render.sh --mesh bunny.obj … | trd-app`.
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
    /// Render meshes as an edge wireframe (line list) instead of filled
    /// triangles (#38).
    #[arg(long)]
    wireframe: bool,
    /// Render meshes textured — sampling the stream's bound texture table at
    /// each vertex UV — instead of the per-vertex color (#20). Requires a
    /// `0.0.4` stream carrying a texture table (else the bound texture is 1×1
    /// white).
    #[arg(long, conflicts_with = "wireframe")]
    textured: bool,
    /// Overlay each drawn mesh's axis-aligned bounding box as a green wireframe
    /// box (#42).
    #[arg(long)]
    aabb: bool,
    /// Overlay a coordinate-axes gizmo (X=red, Y=green, Z=blue) at the world
    /// origin (#42).
    #[arg(long)]
    axes: bool,
    /// Base directory for per-frame background images (`0.0.5`, #63). When set, a
    /// frame's `frame_path` (relative) is joined to this dir, decoded (PNG/JPEG),
    /// and composited beneath the scene as a background frame plane. Without it,
    /// `frame_path` columns are ignored.
    #[arg(long, value_name = "DIR")]
    frames_base: Option<PathBuf>,
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
    /// The scene renderer, built lazily once the stream's mesh table (or the
    /// legacy built-in fallback) has arrived from the reader thread.
    renderer: Option<MeshRenderer>,
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
            renderer: None,
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Uploads the stream's meshes and builds the scene renderer (each mesh
    /// centered + scaled to fit via its preview base model). Idempotent per
    /// stream: called once when the mesh table first arrives.
    fn set_meshes(&mut self, meshes: &[Mesh]) {
        self.renderer = Some(MeshRenderer::with_meshes_preview(
            &self.device,
            self.config.format,
            meshes,
        ));
    }

    /// Binds `texture` as the albedo sampled by [`RenderMode::Textured`] meshes
    /// (`0.0.4`). No-op until the renderer is built; re-uploaded lazily on the
    /// next `render`.
    fn set_texture(&mut self, texture: &ImageTexture) {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_texture(texture);
        }
    }

    /// Renders one frame's [`Scene`](trd_core::Scene) to the window surface.
    /// No-op until the renderer is built and a frame is available.
    fn render(
        &mut self,
        frame: Option<&FrameData>,
        mode: RenderMode,
        show_aabb: bool,
        show_axes: bool,
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
        let scene = build_scene(&frame.draws, mode, show_aabb, show_axes, frame_fit);
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

/// A message from the stdin reader thread: the decoded mesh table (sent once,
/// first), then the optional bound texture (once, only for a `0.0.4` stream
/// carrying a texture table), then the stream's declared playback rate (once),
/// then each decoded frame.
enum StreamMsg {
    Meshes(Vec<Mesh>),
    // Only sent when the stream carries a texture table; small (width/height +
    // an RGBA byte buffer), so it needs no boxing.
    Texture(ImageTexture),
    Rate(f64),
    // Boxed: `FrameData` embeds the large `FrameParams` (camera columns), so an
    // unboxed variant would dwarf `Rate` (clippy::large_enum_variant).
    Frame(Box<FrameData>),
}

/// One decoded frame: its camera/transform params and resolved instanced draw
/// list, built into a [`trd_core::Scene`] at render time. `frame_image` holds a
/// per-frame background image (`0.0.5`, #63) already decoded to RGBA off the
/// render thread (from `frame_path` + `--frames-base`), uploaded + composited
/// beneath the scene at render time; `None` when the frame has no background.
#[derive(Clone)]
struct FrameData {
    params: FrameParams,
    draws: Vec<Draw>,
    frame_image: Option<ImageData>,
}

/// The winit application: owns the GPU state and drives stream playback.
struct App {
    gpu: Option<Gpu>,
    /// Meshes + rate + frames arriving from the stdin reader thread.
    rx: Receiver<StreamMsg>,
    /// The stream's mesh table (or the legacy built-in fallback), held until the
    /// GPU surface exists so the renderer can be built.
    pending_meshes: Option<Vec<Mesh>>,
    /// The stream's bound texture (`0.0.4`), held until the renderer is built so
    /// it can be uploaded; `None` for streams without a texture table.
    pending_texture: Option<ImageTexture>,
    /// Whether `pending_texture` has been applied to the built renderer (so it is
    /// uploaded exactly once, even though it can arrive before or after the mesh
    /// table triggers the renderer build).
    texture_applied: bool,
    /// Every frame received so far, retained so playback can loop.
    frames: Vec<FrameData>,
    /// The frame currently on screen (none until the first arrives).
    current: Option<FrameData>,
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
    /// Render mode (filled / wireframe) applied to every mesh drawable.
    mode: RenderMode,
    /// Overlay each drawn mesh's axis-aligned bounding box (#42).
    show_aabb: bool,
    /// Overlay the origin coordinate-axes gizmo (#42).
    show_axes: bool,
}

impl App {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rx: Receiver<StreamMsg>,
        window_size: (u32, u32),
        rate_override: Option<f64>,
        loop_playback: bool,
        vsync: bool,
        mode: RenderMode,
        show_aabb: bool,
        show_axes: bool,
    ) -> Self {
        Self {
            gpu: None,
            rx,
            pending_meshes: None,
            pending_texture: None,
            texture_applied: false,
            frames: Vec::new(),
            current: None,
            rate_override,
            stream_rate: trd_core::DEFAULT_FRAME_RATE,
            playback_start: None,
            shown_index: None,
            loop_playback,
            stream_done: false,
            window_size,
            vsync,
            mode,
            show_aabb,
            show_axes,
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
                Ok(StreamMsg::Meshes(meshes)) => self.pending_meshes = Some(meshes),
                Ok(StreamMsg::Texture(texture)) => self.pending_texture = Some(texture),
                Ok(StreamMsg::Rate(rate)) => self.stream_rate = rate,
                Ok(StreamMsg::Frame(frame)) => self.frames.push(*frame),
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
        self.current = Some(self.frames[index].clone());
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
            WindowEvent::RedrawRequested => gpu.render(
                self.current.as_ref(),
                self.mode,
                self.show_aabb,
                self.show_axes,
            ),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.drain_stream();

        // Build the scene renderer once both the GPU surface and the stream's
        // mesh table (or built-in fallback) are available; then paint.
        if let Some(gpu) = self.gpu.as_mut() {
            if gpu.renderer.is_none() {
                if let Some(meshes) = self.pending_meshes.as_ref() {
                    gpu.set_meshes(meshes);
                    gpu.window.request_redraw();
                }
            }
            // Upload the stream's bound texture once the renderer exists (the
            // texture can arrive before or after the mesh table).
            if !self.texture_applied && gpu.renderer.is_some() {
                if let Some(texture) = self.pending_texture.as_ref() {
                    gpu.set_texture(texture);
                    self.texture_applied = true;
                    gpu.window.request_redraw();
                }
            }
        }

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
/// `tx` until the stream ends. When `frames_base` is set, a frame's `frame_path`
/// (`0.0.5`, #63) is loaded + decoded to RGBA off the render thread and shipped
/// with the frame for compositing.
fn spawn_stdin_reader(tx: mpsc::Sender<StreamMsg>, frames_base: Option<PathBuf>) {
    let spawned = std::thread::Builder::new()
        .name("trd-stdin-reader".to_string())
        .spawn(move || {
            let stdin = std::io::stdin().lock();
            let meshes_tx = tx.clone();
            let texture_tx = tx.clone();
            let rate_tx = tx.clone();
            // A send error just means the window closed; stop reading in that case.
            if let Err(err) = read_scene_stream_with_meta(
                stdin,
                |meshes| {
                    let _ = meshes_tx.send(StreamMsg::Meshes(meshes));
                },
                |texture| {
                    if let Some(texture) = texture {
                        let _ = texture_tx.send(StreamMsg::Texture(texture));
                    }
                },
                |rate| {
                    let _ = rate_tx.send(StreamMsg::Rate(rate));
                },
                |params, draws, frame_ref| {
                    let frame_image = frame_ref
                        .as_deref()
                        .zip(frames_base.as_ref())
                        .and_then(|(rel, base)| load_frame_image(&base.join(rel)));
                    let _ = tx.send(StreamMsg::Frame(Box::new(FrameData {
                        params,
                        draws,
                        frame_image,
                    })));
                },
            ) {
                log::error!("input stream error: {err}");
            }
        });
    if let Err(err) = spawned {
        log::error!("failed to spawn stdin reader thread: {err}");
    }
}

/// Decodes a background frame image file (PNG/JPEG) to RGBA (#63). Kept in the
/// shell so trd-core does no image I/O; a load failure logs and yields `None`
/// (that frame renders without a background).
fn load_frame_image(path: &std::path::Path) -> Option<ImageData> {
    match image::open(path) {
        Ok(img) => {
            let rgba = img.to_rgba8();
            let (width, height) = rgba.dimensions();
            Some(ImageData {
                width,
                height,
                rgba: rgba.into_raw(),
            })
        }
        Err(err) => {
            log::warn!("skipping frame background {}: {err}", path.display());
            None
        }
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
    let mode = if cli.textured {
        RenderMode::Textured
    } else if cli.wireframe {
        RenderMode::Wireframe
    } else {
        RenderMode::Filled
    };

    let (tx, rx) = mpsc::channel();
    spawn_stdin_reader(tx, cli.frames_base.clone());

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
        mode,
        cli.aabb,
        cli.axes,
    );
    event_loop.run_app(&mut app)?;
    Ok(())
}
