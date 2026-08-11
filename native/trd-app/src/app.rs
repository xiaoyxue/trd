//! Native windowed streaming viewer: a winit application that owns a live wgpu
//! surface and renders trd frame-params streamed from stdin.
//!
//! A background thread reads the Arrow IPC stream (the same `[mesh][params]`
//! input `trd-cli` consumes) via [`trd_core::read_scene_stream_with_meta`],
//! forwarding the decoded mesh table then each frame's params + instanced draw
//! list over a channel. The window plays them at a fixed rate, encoding each
//! frame's [`trd_core::Scene`] with the shared [`trd_core::SceneRenderer`] — so
//! all rendering logic still lives in `trd-core`.

use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use trd_core::{
    DisneyMaterial, EnvMapData, ImageBasedLighting, ImageTexture, Lighting, Mesh, PbrConfig,
    RenderMode, RenderOptions, ToneMapping,
};
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
use crate::renderer::WindowRenderer;
use crate::stream::{spawn_stdin_reader, FrameData, StreamMsg};

/// The winit application: owns the GPU state and drives stream playback.
struct App {
    gpu: Option<WindowRenderer>,
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
    /// Draw mode + every overlay toggle, in the **one** type every front-end uses
    /// to describe a frame's appearance (#180).
    options: RenderOptions,
    /// The Disney PBR material + optional HDR env probe (`--pbr`), held until the
    /// renderer is built so it can be applied; `None` unless `--pbr` is set.
    pbr_config: Option<PbrConfig>,
    /// Whether `pbr_config` has been applied to the built renderer (once).
    pbr_applied: bool,
}

impl App {
    fn new(
        rx: Receiver<StreamMsg>,
        window_size: (u32, u32),
        rate_override: Option<f64>,
        loop_playback: bool,
        vsync: bool,
        options: RenderOptions,
        pbr_config: Option<PbrConfig>,
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
            options,
            pbr_config,
            pbr_applied: false,
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
        // The next absolute frame boundary after *now* (wall-clock), so the wakeup
        // is always in the future. Deriving it from `shown_index` breaks once
        // playback loops (the looped index is small, e.g. 0..len), scheduling an
        // instant in the past that turns the `WaitUntil` sleep into a busy-loop
        // (100% CPU, and the render thread never gets to pace/present cleanly).
        let elapsed = start.elapsed().as_secs_f64();
        let next_frame = (elapsed * self.rate()).floor() + 1.0;
        Some(start + Duration::from_secs_f64(next_frame / self.rate()))
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

        match pollster::block_on(WindowRenderer::new(window.clone(), self.vsync)) {
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
            WindowEvent::RedrawRequested => gpu.render(self.current.as_ref(), &self.options),
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
            // Apply the Disney PBR material + env probe once the renderer exists.
            if !self.pbr_applied && gpu.renderer.is_some() {
                if let Some(pbr) = self.pbr_config.take() {
                    gpu.set_disney_material(pbr.material);
                    gpu.set_lighting(pbr.lighting);
                    gpu.set_image_based_lighting(pbr.ibl);
                    gpu.set_tone_mapping(pbr.tone_mapping);
                    if let Some(env) = pbr.env_map {
                        gpu.set_env_map(env);
                    }
                    gpu.window.request_redraw();
                }
                self.pbr_applied = true;
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
    let mode = if cli.pbr {
        RenderMode::Pbr
    } else if cli.textured {
        RenderMode::Textured
    } else if cli.wireframe {
        RenderMode::Wireframe
    } else {
        RenderMode::Filled
    };

    // Assemble the Disney PBR config (material + optional HDR env probe) when
    // `--pbr` is set. The `.hdr` file is decoded here so trd-core does no
    // file/codec I/O; it is downscaled to the renderer's portable 2048px limit.
    let pbr_config = if cli.pbr {
        let material = DisneyMaterial {
            metallic: cli.metallic,
            roughness: cli.roughness,
            specular: cli.specular,
            clearcoat: cli.clearcoat,
            ..Default::default()
        };
        let lighting = Lighting {
            ambient: cli.ambient,
            ..Default::default()
        };
        let ibl = ImageBasedLighting {
            intensity: cli.env_intensity,
            ..ImageBasedLighting::default()
        };
        let tone_mapping = ToneMapping {
            operator: cli.tonemap.into(),
            exposure: cli.exposure,
        };
        let env_map = match cli.env.as_ref() {
            Some(path) => Some(load_env_map(path)?),
            None => None,
        };
        Some(PbrConfig {
            material,
            lighting,
            ibl,
            tone_mapping,
            env_map,
        })
    } else {
        None
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
        RenderOptions {
            mode,
            show_aabb: cli.aabb,
            show_axes: cli.axes,
            show_local_axes: cli.axes_local,
            show_local_grid: cli.grid_local.map(Into::into),
            show_local_grid_mesh: cli.grid_mesh,
            ..RenderOptions::default()
        },
        pbr_config,
    );
    event_loop.run_app(&mut app)?;
    Ok(())
}

/// Decodes an equirectangular Radiance `.hdr` file into a linear-RGBA f32
/// [`EnvMapData`], downscaled to the renderer's portable 2048px limit. Kept in
/// the app shell so trd-core does no file/codec I/O.
fn load_env_map(path: &std::path::Path) -> Result<EnvMapData, AppError> {
    let img = image::open(path)
        .map_err(|e| AppError::EnvMap(format!("read {}: {e}", path.display())))?
        .to_rgba32f();
    let (w, h) = img.dimensions();
    Ok(EnvMapData::from_rgba32f(w, h, img.into_raw(), 2048))
}
