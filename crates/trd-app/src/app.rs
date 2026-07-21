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
use trd_core::{ImageTexture, Mesh, RenderMode};
use winit::application::ApplicationHandler;
#[cfg(not(target_os = "windows"))]
use winit::dpi::LogicalSize;
#[cfg(target_os = "windows")]
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::cli::Cli;
use crate::error::AppError;
use crate::renderer::Gpu;
use crate::stream::{spawn_stdin_reader, FrameData, StreamMsg};

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
    /// Overlay a coordinate-axes gizmo at each drawn object's local (model) frame.
    show_local_axes: bool,
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
        show_local_axes: bool,
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
            show_local_axes,
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

        // The CV camera `k` intrinsics (fx/fy/cx/cy) are render-resolution-specific,
        // so the GPU surface must be exactly the authored `--width`×`--height`. On
        // Windows, per-monitor DPI scaling turns a `LogicalSize` request into a
        // larger physical surface (e.g. 960×540 → 1440×810 at 150%), which
        // misprojects the scene over the stretched background frame (the mesh
        // "floats" off its placement quad). Request the size in physical pixels
        // there so the surface matches the authored resolution; other platforms
        // (validated at 100% scale) keep the logical-size request.
        #[cfg(target_os = "windows")]
        let size_attr = PhysicalSize::new(self.window_size.0, self.window_size.1);
        #[cfg(not(target_os = "windows"))]
        let size_attr = LogicalSize::new(self.window_size.0, self.window_size.1);

        let attributes = Window::default_attributes()
            .with_title("trd — stream viewer")
            .with_inner_size(size_attr);

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
                self.show_local_axes,
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
        cli.axes_local,
    );
    event_loop.run_app(&mut app)?;
    Ok(())
}
