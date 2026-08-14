//! Shared browser/native video-editing state (#163/#167).
//!
//! This module owns the editor state, the typed render/pick scheduler, and the
//! wasm browser bridge. The surfaces built on top of it live alongside:
//! [`editing_ui`] renders the editor panels and player, [`details_ui`] draws
//! the Details inspector in immediate mode, and [`diagnostics`] holds the pure
//! domain calculations both rely on.

mod details_ui;
mod diagnostics;
mod editing_ui;

pub use diagnostics::{PoseDeltaDiagnostics, QuadFrameDiagnostics, TrackingPlacementError};

use std::cell::{Cell, RefCell};
use std::rc::Rc;
// `std::time::Instant::now()` panics on `wasm32-unknown-unknown` ("time not
// implemented on this platform"), which surfaces in the browser as
// `RuntimeError: unreachable` the moment the first video frame schedules a
// render. `web_time::Instant` is that same type on native and a
// `performance.now()`-backed clock on wasm.
use web_time::Instant;

use diagnostics::{dot3, pose_delta};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CatalogAsset {
    CocaColaCan,
    BeerCan,
    Dragon,
}

impl CatalogAsset {
    pub const ALL: [Self; 3] = [Self::CocaColaCan, Self::BeerCan, Self::Dragon];

    pub const fn label(self) -> &'static str {
        match self {
            Self::CocaColaCan => "Coca-Cola can",
            Self::BeerCan => "Beer can",
            Self::Dragon => "Dragon",
        }
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::CocaColaCan),
            2 => Some(Self::BeerCan),
            3 => Some(Self::Dragon),
            _ => None,
        }
    }

    pub const fn code(self) -> u8 {
        self as u8 + 1
    }
}

const COMMAND_NONE: u8 = 0;
const COMMAND_PICK_VIDEO: u8 = 1;
const COMMAND_PLAY: u8 = 2;
const COMMAND_PAUSE: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoEditingCommand {
    OpenLocalVideo,
    Play,
    Pause,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoSourceKind {
    LocalFile,
    HttpUrl,
}

#[derive(Debug, Clone, PartialEq)]
struct VideoSourceObservation {
    kind: VideoSourceKind,
    name: String,
    byte_length: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct VideoMetadataObservation {
    width: u32,
    height: u32,
    duration_seconds: f64,
}

/// Media-element level state. It is deliberately *not* per-frame: `mediaTime`
/// travels with its own frame (`IncomingVideoFrame::media_time_seconds`) so the
/// timeline diagnostics describe the frame on screen, not a newer one.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct VideoMediaObservation {
    ready_state: u8,
    ended: bool,
}

#[derive(Clone)]
struct IncomingVideoFrame {
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    frame_index: u32,
    media_time_seconds: f64,
    source_generation: u64,
}

struct RenderedVideoFrame {
    frame: IncomingVideoFrame,
    render_revision: u64,
    diagnostics: RenderedFrameDiagnostics,
}

#[derive(Clone)]
struct RenderedFrameDiagnostics {
    media_time_seconds: f64,
    scene: crate::scene::SceneState,
    selected_asset: Option<CatalogAsset>,
    selected_quad: bool,
    move_direction: crate::interaction::MoveDirection,
    playing: bool,
    show_quad: bool,
    show_quad_gizmo: bool,
    draw_model: Option<trd_core::Matrix4>,
    renderer: crate::video_editing_renderer::VideoRendererDiagnostics,
}

#[derive(Clone, Copy)]
struct PickRequest {
    id: u64,
    point: (u32, u32),
    source_generation: u64,
    render_revision: u64,
}

struct PickResult {
    id: u64,
    source_generation: u64,
    render_revision: u64,
    hit: Option<u32>,
}

pub struct VideoEditingShared {
    frame: RefCell<Option<IncomingVideoFrame>>,
    latest_video_frame: RefCell<Option<IncomingVideoFrame>>,
    rendered_frame: RefCell<Option<RenderedVideoFrame>>,
    context: RefCell<Option<egui::Context>>,
    command: Cell<u8>,
    asset_request: Cell<u8>,
    video_url_request: RefCell<Option<String>>,
    seek_frame: Cell<i32>,
    video_loaded: Cell<bool>,
    video_playing: Cell<bool>,
    video_source: RefCell<Option<VideoSourceObservation>>,
    video_metadata: Cell<Option<VideoMetadataObservation>>,
    video_media: Cell<VideoMediaObservation>,
    source_generation: Cell<u64>,
    needs_overlay: Cell<bool>,
    render_revision: Cell<u64>,
    render_in_flight: Cell<bool>,
    render_in_flight_frame: Cell<Option<u32>>,
    last_render_latency_ms: Cell<Option<f64>>,
    render_latency_total_ms: Cell<f64>,
    render_latency_count: Cell<u64>,
    last_render_error: RefCell<Option<String>>,
    pending_pick: Cell<Option<PickRequest>>,
    pick_revision: Cell<u64>,
    pick_in_flight: Cell<bool>,
    pick_result: RefCell<Option<PickResult>>,
    last_pick_error: RefCell<Option<String>>,
    renderer_generation: Cell<u64>,
    renderer: RefCell<Option<crate::video_editing_renderer::VideoPlacementRenderer>>,
    renderer_diagnostics: RefCell<Option<crate::video_editing_renderer::VideoRendererDiagnostics>>,
    asset_defaults: RefCell<Option<(CatalogAsset, trd_core::RenderMode, trd_core::DisneyMaterial)>>,
    error: RefCell<Option<String>>,
}

impl Default for VideoEditingShared {
    fn default() -> Self {
        Self {
            frame: RefCell::new(None),
            latest_video_frame: RefCell::new(None),
            rendered_frame: RefCell::new(None),
            context: RefCell::new(None),
            command: Cell::new(COMMAND_NONE),
            asset_request: Cell::new(0),
            video_url_request: RefCell::new(None),
            seek_frame: Cell::new(-1),
            video_loaded: Cell::new(false),
            video_playing: Cell::new(false),
            video_source: RefCell::new(None),
            video_metadata: Cell::new(None),
            video_media: Cell::new(VideoMediaObservation::default()),
            source_generation: Cell::new(0),
            needs_overlay: Cell::new(false),
            render_revision: Cell::new(0),
            render_in_flight: Cell::new(false),
            render_in_flight_frame: Cell::new(None),
            last_render_latency_ms: Cell::new(None),
            render_latency_total_ms: Cell::new(0.0),
            render_latency_count: Cell::new(0),
            last_render_error: RefCell::new(None),
            pending_pick: Cell::new(None),
            pick_revision: Cell::new(0),
            pick_in_flight: Cell::new(false),
            pick_result: RefCell::new(None),
            last_pick_error: RefCell::new(None),
            renderer_generation: Cell::new(0),
            renderer: RefCell::new(None),
            renderer_diagnostics: RefCell::new(None),
            asset_defaults: RefCell::new(None),
            error: RefCell::new(None),
        }
    }
}

impl VideoEditingShared {
    pub fn update_video_frame_rgba(
        &self,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        frame_index: u32,
        media_time_seconds: f64,
    ) -> Result<(), String> {
        let expected = width as usize * height as usize * 4;
        if rgba.len() != expected {
            return Err(format!(
                "video RGBA length {} != {width}x{height}x4",
                rgba.len()
            ));
        }
        self.frame.replace(Some(IncomingVideoFrame {
            rgba,
            width,
            height,
            frame_index,
            media_time_seconds,
            source_generation: self.source_generation.get(),
        }));
        self.request_repaint();
        Ok(())
    }

    /// Takes the pending playback command as its **wire code**, leaving
    /// `COMMAND_NONE`.
    ///
    /// These raw accessors exist so the browser bridge polls the shared state
    /// through an API instead of reaching into its cells — the bridge lives in the
    /// wasm surface crate now (#180). They are the untyped twins of
    /// [`take_command`](Self::take_command) and friends, which the Rust app uses;
    /// JS gets the code because that is what crosses the ABI.
    pub fn take_command_code(&self) -> u8 {
        self.command.replace(COMMAND_NONE)
    }

    /// Takes the pending catalog-asset request as its wire code, leaving `0`
    /// ("none").
    pub fn take_asset_request_code(&self) -> u8 {
        self.asset_request.replace(0)
    }

    /// Takes the pending seek target as its wire code, leaving `-1` ("no seek").
    pub fn take_seek_frame_code(&self) -> i32 {
        self.seek_frame.replace(-1)
    }

    pub fn set_video_status(&self, loaded: bool, playing: bool) {
        if !loaded {
            self.source_generation
                .set(self.source_generation.get().wrapping_add(1));
            self.frame.replace(None);
            self.latest_video_frame.replace(None);
            self.rendered_frame.replace(None);
            self.pending_pick.set(None);
            self.pick_result.replace(None);
            self.last_pick_error.replace(None);
            self.last_render_error.replace(None);
            self.needs_overlay.set(false);
            self.render_in_flight_frame.set(None);
            self.video_media.set(VideoMediaObservation::default());
        }
        self.video_loaded.set(loaded);
        self.video_playing.set(playing);
        if !loaded {
            self.error.replace(None);
        }
        self.request_repaint();
    }

    pub fn set_video_source_observation(
        &self,
        kind: VideoSourceKind,
        name: impl Into<String>,
        byte_length: Option<u64>,
    ) {
        self.video_metadata.set(None);
        self.video_media.set(VideoMediaObservation::default());
        self.video_source.replace(Some(VideoSourceObservation {
            kind,
            name: name.into(),
            byte_length,
        }));
        self.request_repaint();
    }

    pub fn set_video_metadata_observation(&self, width: u32, height: u32, duration_seconds: f64) {
        self.video_metadata.set(Some(VideoMetadataObservation {
            width,
            height,
            duration_seconds,
        }));
        self.request_repaint();
    }

    pub fn set_video_media_observation(&self, ready_state: u8, ended: bool) {
        self.video_media
            .set(VideoMediaObservation { ready_state, ended });
        self.request_repaint();
    }

    pub fn set_error(&self, message: impl Into<String>) {
        self.error.replace(Some(message.into()));
        self.request_repaint();
    }

    pub fn clear_error(&self) {
        self.error.replace(None);
    }

    pub fn take_command(&self) -> Option<VideoEditingCommand> {
        match self.command.replace(COMMAND_NONE) {
            COMMAND_PICK_VIDEO => Some(VideoEditingCommand::OpenLocalVideo),
            COMMAND_PLAY => Some(VideoEditingCommand::Play),
            COMMAND_PAUSE => Some(VideoEditingCommand::Pause),
            _ => None,
        }
    }

    pub fn take_asset_request(&self) -> Option<CatalogAsset> {
        CatalogAsset::from_code(self.asset_request.replace(0))
    }

    pub fn take_video_url_request(&self) -> Option<String> {
        self.video_url_request.borrow_mut().take()
    }

    pub fn take_seek_frame(&self) -> Option<u32> {
        let frame = self.seek_frame.replace(-1);
        (frame >= 0).then_some(frame as u32)
    }

    pub fn set_renderer(&self, renderer: crate::video_editing_renderer::VideoPlacementRenderer) {
        self.renderer_generation
            .set(self.renderer_generation.get().wrapping_add(1));
        self.renderer_diagnostics
            .replace(Some(renderer.diagnostics()));
        self.renderer.replace(Some(renderer));
        self.request_overlay();
        self.request_repaint();
    }

    pub fn set_catalog_renderer(
        &self,
        asset: CatalogAsset,
        renderer: crate::video_editing_renderer::VideoPlacementRenderer,
    ) {
        let (mode, material) = renderer.defaults();
        self.asset_defaults.replace(Some((asset, mode, material)));
        self.set_renderer(renderer);
    }

    pub fn request_repaint(&self) {
        if let Some(context) = self.context.borrow().as_ref() {
            context.request_repaint();
        }
    }

    fn request_overlay(&self) {
        self.render_revision
            .set(self.render_revision.get().wrapping_add(1));
        self.needs_overlay.set(true);
    }

    fn request_pick(&self, point: (u32, u32)) {
        let id = self.pick_revision.get().wrapping_add(1);
        self.pick_revision.set(id);
        self.pending_pick.set(Some(PickRequest {
            id,
            point,
            source_generation: self.source_generation.get(),
            render_revision: self.render_revision.get(),
        }));
    }

    fn accepts_render(&self, rendered: &RenderedVideoFrame) -> bool {
        rendered.frame.source_generation == self.source_generation.get()
            && rendered.render_revision == self.render_revision.get()
    }

    fn accepts_pick(&self, result: &PickResult) -> bool {
        result.id == self.pick_revision.get()
            && result.source_generation == self.source_generation.get()
            && result.render_revision == self.render_revision.get()
    }

    fn record_render_latency(&self, started: Instant) {
        let latency_ms = started.elapsed().as_secs_f64() * 1_000.0;
        self.last_render_latency_ms.set(Some(latency_ms));
        self.render_latency_total_ms
            .set(self.render_latency_total_ms.get() + latency_ms);
        self.render_latency_count
            .set(self.render_latency_count.get().saturating_add(1));
    }
}

pub struct VideoEditingApp {
    document: trd_core::VideoEditingDocument,
    display_image: egui::ColorImage,
    display_texture: Option<egui::TextureHandle>,
    current_frame_index: u32,
    displayed_frame_index: u32,
    displayed_frame_ready: bool,
    last_rendered_frame_index: Option<u32>,
    displayed_diagnostics: Option<RenderedFrameDiagnostics>,
    display_size: (u32, u32),
    shared: Rc<VideoEditingShared>,
    controller: crate::interaction::InteractionController,
    selected_quad: bool,
    show_quad_gizmo: bool,
    was_playing: bool,
    selected_asset: Option<CatalogAsset>,
    image_sizing: crate::ui::ImageSizing,
    fitted_render_size: (u32, u32),
    show_video_source_dialog: bool,
    video_url: String,
    pending_seek_target: Option<u32>,
    last_pick_result: Option<Option<u32>>,
}

impl VideoEditingApp {
    pub fn new(document: trd_core::VideoEditingDocument, shared: Rc<VideoEditingShared>) -> Self {
        let source_size = (document.video.width, document.video.height);
        let scene = crate::scene::SceneState::default();
        let mut controller = crate::interaction::InteractionController::new(scene);
        controller.target = crate::interaction::InteractionTarget::Object;
        controller.move_direction = crate::interaction::MoveDirection::Reference1;
        controller.move_reference_axes = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
        controller.state.camera.distance = 1.0;
        Self {
            document,
            display_image: egui::ColorImage::from_rgba_unmultiplied([1, 1], &[0, 0, 0, 0]),
            display_texture: None,
            current_frame_index: 0,
            displayed_frame_index: 0,
            displayed_frame_ready: false,
            last_rendered_frame_index: None,
            displayed_diagnostics: None,
            display_size: source_size,
            shared,
            controller,
            selected_quad: false,
            show_quad_gizmo: false,
            was_playing: false,
            selected_asset: None,
            image_sizing: crate::ui::ImageSizing::FitCanvas,
            fitted_render_size: source_size,
            show_video_source_dialog: false,
            video_url: String::new(),
            pending_seek_target: None,
            last_pick_result: None,
        }
    }

    fn ensure_texture(&mut self, context: &egui::Context) {
        if self.display_texture.is_none() {
            self.display_texture = Some(context.load_texture(
                "video-editing-frame",
                self.display_image.clone(),
                egui::TextureOptions::LINEAR,
            ));
        }
    }

    fn video_source_dialog(&mut self, context: &egui::Context) {
        if !self.show_video_source_dialog {
            return;
        }
        let mut open = true;
        let mut close = false;
        egui::Window::new("Open video")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(context, |ui| {
                ui.set_min_width(420.0);
                ui.label("Select the video matched by this editing document.");
                if ui.button("Select local file...").clicked() {
                    self.shared.command.set(COMMAND_PICK_VIDEO);
                    close = true;
                }
                ui.separator();
                ui.label("Video URL");
                let response = ui.add(
                    egui::TextEdit::singleline(&mut self.video_url)
                        .hint_text("https://example.com/video.mp4")
                        .desired_width(f32::INFINITY),
                );
                let submit =
                    response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
                if ui.button("Load URL").clicked() || submit {
                    let url = self.video_url.trim();
                    if url.starts_with("https://") || url.starts_with("http://") {
                        self.shared.video_url_request.replace(Some(url.to_owned()));
                        close = true;
                    } else {
                        self.shared.error.replace(Some(
                            "video URL must start with http:// or https://".to_owned(),
                        ));
                    }
                }
                ui.weak("The URL must allow cross-origin video frame access.");
            });
        self.show_video_source_dialog = open && !close;
    }

    fn set_display_frame(&mut self, rendered: RenderedVideoFrame) {
        let frame = &rendered.frame;
        self.display_size = (frame.width, frame.height);
        self.displayed_frame_index = frame.frame_index;
        self.displayed_frame_ready = true;
        self.last_rendered_frame_index = Some(frame.frame_index);
        self.displayed_diagnostics = Some(rendered.diagnostics);
        if self.pending_seek_target == Some(frame.frame_index) {
            self.pending_seek_target = None;
        }
        self.display_image = egui::ColorImage::from_rgba_unmultiplied(
            [self.display_size.0 as usize, self.display_size.1 as usize],
            &frame.rgba,
        );
        if let Some(texture) = self.display_texture.as_mut() {
            texture.set(self.display_image.clone(), egui::TextureOptions::LINEAR);
        }
    }

    fn consume_video_frame(&mut self) {
        let frame = self.shared.frame.borrow_mut().take();
        let Some(frame) = frame else {
            return;
        };
        self.current_frame_index = frame.frame_index;
        self.shared.latest_video_frame.replace(Some(frame));
        self.shared.request_overlay();
        self.schedule_overlay();
    }

    fn consume_rendered_frame(&mut self) {
        let rendered = self.shared.rendered_frame.borrow_mut().take();
        let Some(rendered) = rendered else {
            return;
        };
        if self.shared.accepts_render(&rendered) {
            self.set_display_frame(rendered);
        }
    }

    fn consume_asset_defaults(&mut self) {
        let Some((asset, mode, material)) = self.shared.asset_defaults.borrow_mut().take() else {
            return;
        };
        if self.selected_asset == Some(asset) {
            self.controller.state.modes[0] = mode;
            self.controller.state.materials[0] = material;
            self.controller.state.environment_available = true;
            self.controller.state.lighting = match asset {
                CatalogAsset::Dragon => trd_core::Lighting {
                    ambient: 0.0,
                    scale: 0.0,
                    ..trd_core::Lighting::default()
                },
                CatalogAsset::CocaColaCan | CatalogAsset::BeerCan => trd_core::Lighting::default(),
            };
            self.controller.rebase_reset();
            self.shared.request_overlay();
        }
    }

    fn consume_pick_result(&mut self) {
        let Some(result) = self.shared.pick_result.borrow_mut().take() else {
            return;
        };
        if !self.shared.accepts_pick(&result) {
            return;
        }
        let hit = result.hit;
        self.last_pick_result = Some(hit);
        if hit != self.controller.state.selected {
            self.controller.state.selected = hit;
            self.shared.request_overlay();
        }
    }

    fn schedule_overlay(&self) {
        if !self.shared.needs_overlay.get()
            || self.shared.render_in_flight.get()
            || self.shared.pick_in_flight.get()
        {
            return;
        }

        let Some(video) = self.shared.latest_video_frame.borrow().clone() else {
            return;
        };
        let Some(background_frame) = self
            .document
            .frames
            .get(video.frame_index as usize)
            .cloned()
        else {
            return;
        };
        let quad_frame = self.quad_frame_at(video.frame_index);
        let show_quad = !self.shared.video_playing.get() && background_frame.tracked;
        let quad_model = quad_frame
            .filter(|_| show_quad)
            .map(trd_placement::quad_outline_model);
        let quad_axes = quad_frame
            .filter(|_| show_quad)
            .map(trd_placement::quad_axes_model);
        let show_object =
            self.selected_asset.is_some() && self.selected_quad && background_frame.tracked;
        let placement_frame = show_object.then_some(background_frame.clone());
        let model = if show_object {
            self.placement_model_at(video.frame_index)
        } else {
            None
        };
        let Some(mut renderer) = self.shared.renderer.borrow_mut().take() else {
            return;
        };
        let source_size = (self.document.video.width, self.document.video.height);
        let requested_size = match self.image_sizing {
            crate::ui::ImageSizing::FitCanvas => (
                self.fitted_render_size.0.min(source_size.0).max(1),
                self.fitted_render_size.1.min(source_size.1).max(1),
            ),
            crate::ui::ImageSizing::OriginalResolution => source_size,
        };
        if let Err(error) = renderer.resize(requested_size.0, requested_size.1) {
            self.shared.renderer.replace(Some(renderer));
            self.shared.error.replace(Some(error));
            return;
        }
        self.shared
            .renderer_diagnostics
            .replace(Some(renderer.diagnostics()));
        let render_size = renderer.size();
        self.shared.needs_overlay.set(false);
        let mut state = self.controller.state.clone();
        let rendered_playing = self.shared.video_playing.get();
        if rendered_playing {
            state.selected = None;
            state.show_aabb = false;
            state.show_axes = false;
            state.show_local_axes = false;
            state.show_world_grid = false;
            state.show_local_grid = false;
        }
        self.shared.render_in_flight.set(true);
        self.shared
            .render_in_flight_frame
            .set(Some(video.frame_index));
        let shared = self.shared.clone();
        let render_revision = shared.render_revision.get();
        let source_generation = video.source_generation;
        let renderer_generation = shared.renderer_generation.get();
        let width = self.document.video.width;
        let height = self.document.video.height;
        let show_quad_gizmo = self.show_quad_gizmo;
        let selected_asset = self.selected_asset;
        let selected_quad = self.selected_quad;
        let move_direction = self.controller.move_direction;
        let rendered_model = model;
        let background_frame_index = video.frame_index;
        let background_media_time = video.media_time_seconds;
        let render_started = Instant::now();
        let render = async move {
            let result = renderer
                .render(
                    &video.rgba,
                    video.width,
                    video.height,
                    (width, height),
                    &background_frame,
                    quad_model,
                    quad_axes,
                    show_quad_gizmo,
                    placement_frame.as_ref(),
                    model,
                    &state,
                )
                .await;
            let renderer_diagnostics = renderer.diagnostics();
            if shared.renderer_generation.get() != renderer_generation {
                shared.render_in_flight.set(false);
                shared.render_in_flight_frame.set(None);
                shared.request_repaint();
                return;
            }
            shared.renderer.replace(Some(renderer));
            let current = source_generation == shared.source_generation.get()
                && render_revision == shared.render_revision.get();
            match (current, result) {
                (true, Ok(rgba)) => {
                    shared.record_render_latency(render_started);
                    shared
                        .renderer_diagnostics
                        .replace(Some(renderer_diagnostics.clone()));
                    shared.last_render_error.replace(None);
                    shared.rendered_frame.replace(Some(RenderedVideoFrame {
                        frame: IncomingVideoFrame {
                            rgba,
                            width: render_size.0,
                            height: render_size.1,
                            frame_index: background_frame_index,
                            media_time_seconds: background_media_time,
                            source_generation,
                        },
                        render_revision,
                        diagnostics: RenderedFrameDiagnostics {
                            media_time_seconds: background_media_time,
                            scene: state,
                            selected_asset,
                            selected_quad,
                            move_direction,
                            playing: rendered_playing,
                            show_quad,
                            show_quad_gizmo,
                            draw_model: rendered_model,
                            renderer: renderer_diagnostics,
                        },
                    }));
                }
                (true, Err(error)) => {
                    shared.record_render_latency(render_started);
                    shared
                        .renderer_diagnostics
                        .replace(Some(renderer_diagnostics));
                    shared.last_render_error.replace(Some(error.clone()));
                    shared.error.replace(Some(error));
                }
                (false, _) => {}
            }
            shared.render_in_flight.set(false);
            shared.render_in_flight_frame.set(None);
            if let Some(context) = shared.context.borrow().as_ref() {
                context.request_repaint();
            }
        };
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(render);
        #[cfg(not(target_arch = "wasm32"))]
        pollster::block_on(render);
    }

    fn schedule_pick(&self) {
        if self.shared.render_in_flight.get() || self.shared.pick_in_flight.get() {
            return;
        }
        let Some(request) = self.shared.pending_pick.take() else {
            return;
        };
        let Some(frame) = self
            .document
            .frames
            .get(self.displayed_frame_index as usize)
            .cloned()
        else {
            return;
        };
        let Some(model) = self.placement_model_at(self.displayed_frame_index) else {
            return;
        };
        let Some(mut renderer) = self.shared.renderer.borrow_mut().take() else {
            self.shared.pending_pick.set(Some(request));
            return;
        };
        let source_size = (self.document.video.width, self.document.video.height);
        let render_size = renderer.size();
        let target_point = (
            request.point.0 * render_size.0 / self.display_size.0.max(1),
            request.point.1 * render_size.1 / self.display_size.1.max(1),
        );
        self.shared.pick_in_flight.set(true);
        let shared = self.shared.clone();
        let renderer_generation = shared.renderer_generation.get();
        let pick = async move {
            let result = renderer
                .pick(&frame, source_size, model, target_point)
                .await;
            let renderer_diagnostics = renderer.diagnostics();
            if shared.renderer_generation.get() != renderer_generation {
                shared.pick_in_flight.set(false);
                shared.request_repaint();
                return;
            }
            shared.renderer.replace(Some(renderer));
            let current = request.id == shared.pick_revision.get()
                && request.source_generation == shared.source_generation.get()
                && request.render_revision == shared.render_revision.get();
            match (current, result) {
                (true, Ok(hit)) => {
                    shared
                        .renderer_diagnostics
                        .replace(Some(renderer_diagnostics));
                    shared.last_pick_error.replace(None);
                    shared.pick_result.replace(Some(PickResult {
                        id: request.id,
                        source_generation: request.source_generation,
                        render_revision: request.render_revision,
                        hit,
                    }));
                }
                (true, Err(error)) => {
                    shared
                        .renderer_diagnostics
                        .replace(Some(renderer_diagnostics));
                    shared.last_pick_error.replace(Some(error.clone()));
                    shared.error.replace(Some(error));
                }
                (false, _) => {}
            }
            shared.pick_in_flight.set(false);
            if let Some(context) = shared.context.borrow().as_ref() {
                context.request_repaint();
            }
        };
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(pick);
        #[cfg(not(target_arch = "wasm32"))]
        pollster::block_on(pick);
    }

    fn quad_frame_at(&self, frame_index: u32) -> Option<trd_placement::QuadFrame> {
        self.quad_frame_result_at(frame_index).ok()
    }

    fn quad_frame_result_at(
        &self,
        frame_index: u32,
    ) -> Result<trd_placement::QuadFrame, TrackingPlacementError> {
        let frame = self
            .document
            .frames
            .get(frame_index as usize)
            .ok_or(TrackingPlacementError::FrameOutOfRange)?;
        let k = frame.k.ok_or(TrackingPlacementError::MissingIntrinsics)?;
        let placement_quad = frame
            .placement_quad
            .ok_or(TrackingPlacementError::MissingQuad)?;
        trd_placement::quad_frame(
            trd_placement::CameraIntrinsics { row_major: k },
            trd_placement::PlacementQuad {
                points_px: placement_quad,
            },
        )
        .map_err(TrackingPlacementError::from)
    }

    fn placement_model_at(&self, frame_index: u32) -> Option<trd_core::Matrix4> {
        let frame = self.quad_frame_at(frame_index)?;
        let placement = trd_placement::LocalPlacement {
            offset_e1: 1.3,
            offset_e2: -1.7,
            size_factor: 0.24,
            ..Default::default()
        };
        let quad_basis = trd_placement::placement_model(frame, placement).ok()?;
        let object = self.controller.state.objects.first()?;
        let object_model = trd_core::Matrix4::from_cols_array(&object.model_matrix());
        Some(quad_basis * object_model)
    }

    /// Resolves the values the Details panel cannot simply read off the app:
    /// they need the displayed-frame pin plus a little domain math. Everything
    /// else the panel shows — document metadata and live host observations — is
    /// read directly at draw time, so nothing is copied twice.
    ///
    /// Computed once per Details draw, and only while the panel is open.
    pub(super) fn displayed_facts(&self) -> DisplayedFacts {
        let displayed_frame_index = self
            .displayed_frame_ready
            .then_some(self.displayed_frame_index);
        let timeline_frame =
            displayed_frame_index.and_then(|index| self.document.frames.get(index as usize));
        // Media time rides with its own frame, so the timeline block describes
        // the frame actually on screen rather than a newer presented one.
        let media_time_seconds = self
            .displayed_frame_ready
            .then(|| {
                self.displayed_diagnostics
                    .as_ref()
                    .map(|displayed| displayed.media_time_seconds)
            })
            .flatten();
        let presented_frame_index = self
            .shared
            .latest_video_frame
            .borrow()
            .as_ref()
            .map(|frame| frame.frame_index);
        let in_flight_frame_index = self.shared.render_in_flight_frame.get();
        let coalesced_frame_index = in_flight_frame_index.and_then(|in_flight| {
            presented_frame_index.filter(|presented| *presented != in_flight)
        });

        let (quad, placement_error) = match displayed_frame_index {
            Some(index) if timeline_frame.is_some_and(|frame| frame.tracked) => {
                match self.quad_frame_result_at(index) {
                    Ok(frame) => (Some(frame), None),
                    Err(error) => (None, Some(error)),
                }
            }
            _ => (None, None),
        };
        let previous_quad = displayed_frame_index.and_then(|index| {
            (0..index).rev().find_map(|previous_index| {
                self.document
                    .frames
                    .get(previous_index as usize)
                    .filter(|frame| frame.tracked)
                    .and_then(|_| {
                        self.quad_frame_at(previous_index)
                            .map(|frame| (previous_index, frame))
                    })
            })
        });
        let pose_delta =
            quad.zip(previous_quad)
                .map(|(current, (previous_frame_index, previous))| {
                    pose_delta(previous_frame_index, previous, current)
                });
        let normal_sign_warning = quad
            .zip(previous_quad)
            .is_some_and(|(current, (_, previous))| dot3(current.e3, previous.e3) < 0.0);

        let displayed = self.displayed_diagnostics.as_ref();
        let scene = displayed.map_or(&self.controller.state, |displayed| &displayed.scene);
        let selected_asset = displayed.map_or(self.selected_asset, |d| d.selected_asset);
        let selected_quad = displayed.map_or(self.selected_quad, |d| d.selected_quad);
        let playing = displayed.is_some_and(|d| d.playing);
        let renderer = displayed
            .map(|d| d.renderer.clone())
            .or_else(|| self.shared.renderer_diagnostics.borrow().clone());

        let tracked = timeline_frame.is_some_and(|frame| frame.tracked);
        let visibility_reason = if playing {
            "playing"
        } else if selected_asset.is_none() {
            "no asset"
        } else if !tracked {
            "untracked tail"
        } else if !selected_quad {
            "no quad selected"
        } else {
            "tracked"
        };
        let object_visible = visibility_reason == "tracked";
        let draw_model = displayed
            .and_then(|d| d.draw_model)
            .or_else(|| {
                displayed_frame_index
                    .filter(|_| object_visible)
                    .and_then(|index| self.placement_model_at(index))
            })
            .map(trd_core::Matrix4::to_cols_array);
        let move_direction = displayed.map_or(self.controller.move_direction, |d| d.move_direction);
        let movement_basis = match move_direction {
            crate::interaction::MoveDirection::LocalX
            | crate::interaction::MoveDirection::LocalY
            | crate::interaction::MoveDirection::LocalZ => ["object X", "object Y", "object Z"],
            _ => ["quad e1", "quad e2", "quad e3"],
        };
        let imported_material = renderer
            .as_ref()
            .and_then(|facts| facts.asset.as_ref())
            .map(|facts| &facts.imported_material);
        let reflective_tracking_warning = imported_material
            .is_some_and(|imported| imported.metallic >= 0.7 || imported.auxiliary.textures.normal)
            && pose_delta.as_ref().is_some_and(|delta| {
                delta.rotation_degrees >= 1.0
                    || quad.is_some_and(|quad| delta.translation >= quad.axis_length * 0.02)
            });

        let show_quad = displayed.is_some_and(|d| d.show_quad);
        let show_quad_gizmo = displayed.is_some_and(|d| d.show_quad_gizmo);
        let background_drawables =
            1 + u32::from(show_quad) + if show_quad && show_quad_gizmo { 2 } else { 0 };
        let foreground_drawables = if object_visible {
            1 + u32::from(scene.show_local_axes)
                + u32::from(scene.show_axes)
                + u32::from(scene.show_local_grid)
                + u32::from(scene.show_world_grid)
        } else {
            0
        };
        let selection_drawables =
            u32::from(object_visible && (scene.show_aabb || scene.selected == Some(0)));
        let render_target_size = renderer
            .as_ref()
            .map(|facts| facts.target_size)
            .unwrap_or(self.display_size);

        DisplayedFacts {
            frame_index: displayed_frame_index,
            media_time_seconds,
            timeline_frame: timeline_frame.cloned(),
            presented_frame_index,
            in_flight_frame_index,
            coalesced_frame_index,
            quad,
            placement_error,
            pose_delta,
            normal_sign_warning,
            scene: scene.clone(),
            selected_asset,
            selected_quad,
            visibility_reason,
            draw_model,
            movement_basis,
            reflective_tracking_warning,
            background_drawables,
            foreground_drawables,
            selection_drawables,
            render_target_size,
            renderer,
            requested_frame_index: self.current_frame_index,
            rendered_frame_index: self.last_rendered_frame_index,
            seek_target: self.pending_seek_target,
            latest_pick_result: self.last_pick_result,
            shared: self.shared.clone(),
        }
    }
}

/// The subset of Details values that must be derived rather than read.
///
/// Deliberately *not* a snapshot of everything the panel shows: static document
/// metadata and live host observations are read straight from the app while
/// drawing, so there is exactly one representation of each value.
pub(super) struct DisplayedFacts {
    pub frame_index: Option<u32>,
    pub media_time_seconds: Option<f64>,
    pub timeline_frame: Option<trd_core::VideoEditingFrame>,
    pub presented_frame_index: Option<u32>,
    pub in_flight_frame_index: Option<u32>,
    pub coalesced_frame_index: Option<u32>,
    pub quad: Option<trd_placement::QuadFrame>,
    pub placement_error: Option<TrackingPlacementError>,
    pub pose_delta: Option<PoseDeltaDiagnostics>,
    pub normal_sign_warning: bool,
    pub scene: crate::scene::SceneState,
    pub selected_asset: Option<CatalogAsset>,
    pub selected_quad: bool,
    pub visibility_reason: &'static str,
    pub draw_model: Option<[f32; 16]>,
    pub movement_basis: [&'static str; 3],
    pub reflective_tracking_warning: bool,
    pub background_drawables: u32,
    pub foreground_drawables: u32,
    pub selection_drawables: u32,
    pub render_target_size: (u32, u32),
    pub renderer: Option<crate::video_editing_renderer::VideoRendererDiagnostics>,
    pub requested_frame_index: u32,
    pub rendered_frame_index: Option<u32>,
    pub seek_target: Option<u32>,
    pub latest_pick_result: Option<Option<u32>>,
    pub shared: Rc<VideoEditingShared>,
}

/// Maps a media-clock time to the nearest zero-based video frame, clamped to
/// the editing document's declared frame range.
pub fn frame_index_at_media_time(
    media_time_seconds: f64,
    fps_num: u32,
    fps_den: u32,
    frame_count: u32,
) -> u32 {
    let frame = (media_time_seconds * f64::from(fps_num) / f64::from(fps_den.max(1)))
        .round()
        .max(0.0) as u32;
    frame.min(frame_count.saturating_sub(1))
}

/// Maps a zero-based video frame to its media-clock time, clamped to the editing
/// document's declared frame range.
pub fn media_time_at_frame(frame_index: u32, fps_num: u32, fps_den: u32, frame_count: u32) -> f64 {
    let frame = frame_index.min(frame_count.saturating_sub(1));
    f64::from(frame) * f64::from(fps_den) / f64::from(fps_num.max(1))
}

pub(super) fn point_in_quad(point: [f32; 2], quad: [[f32; 2]; 4]) -> bool {
    let mut inside = false;
    let mut previous = quad[3];
    for current in quad {
        if (current[1] > point[1]) != (previous[1] > point[1])
            && point[0]
                < (previous[0] - current[0]) * (point[1] - current[1]) / (previous[1] - current[1])
                    + current[0]
        {
            inside = !inside;
        }

        previous = current;
    }
    inside
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(super) fn document() -> trd_core::VideoEditingDocument {
        trd_core::VideoEditingDocument {
            video: trd_core::VideoInfo {
                source_name: "shot.mp4".to_owned(),
                mime: "video/mp4".to_owned(),
                codec: "h264".to_owned(),
                sha256: "unused".to_owned(),
                byte_length: 1,
                width: 1920,
                height: 1080,
                fps_num: 24,
                fps_den: 1,
                frame_count: 288,
                duration_us: 12_000_000,
            },
            poster_bytes: vec![1, 2, 3],
            frames: vec![trd_core::VideoEditingFrame {
                video_frame_index: 0,
                present_index: 0,
                timestamp_us: 0,
                k: None,
                placement_quad: None,
                tracked: false,
            }],
        }
    }

    #[test]
    fn unloaded_editor_starts_without_a_frame_or_texture() {
        let shared = Rc::new(VideoEditingShared::default());
        let app = VideoEditingApp::new(document(), shared.clone());
        assert!(shared.latest_video_frame.borrow().is_none());
        assert!(app.display_texture.is_none());
        assert_eq!(app.display_size, (1920, 1080));
    }

    #[test]
    fn newest_incoming_frame_replaces_the_pending_frame() {
        let shared = VideoEditingShared::default();
        shared
            .update_video_frame_rgba(vec![1, 2, 3, 4], 1, 1, 7, 0.25)
            .unwrap();
        shared
            .update_video_frame_rgba(vec![5, 6, 7, 8], 1, 1, 9, 0.5)
            .unwrap();

        let frame = shared.frame.borrow_mut().take().unwrap();
        assert_eq!(frame.frame_index, 9);
        assert_eq!(frame.rgba, vec![5, 6, 7, 8]);
    }

    #[test]
    fn invalid_incoming_frame_does_not_replace_the_pending_frame() {
        let shared = VideoEditingShared::default();
        shared
            .update_video_frame_rgba(vec![1, 2, 3, 4], 1, 1, 7, 0.25)
            .unwrap();
        assert!(shared
            .update_video_frame_rgba(vec![5, 6, 7], 1, 1, 9, 0.5)
            .is_err());

        assert_eq!(
            shared
                .frame
                .borrow()
                .as_ref()
                .map(|frame| frame.frame_index),
            Some(7)
        );
    }

    #[test]
    fn one_slot_commands_and_seek_requests_keep_the_newest_value() {
        let shared = VideoEditingShared::default();
        shared.command.set(COMMAND_PLAY);
        shared.command.set(COMMAND_PAUSE);
        assert_eq!(shared.take_command(), Some(VideoEditingCommand::Pause));
        assert_eq!(shared.take_command(), None);

        shared.seek_frame.set(12);
        shared.seek_frame.set(42);
        assert_eq!(shared.take_seek_frame(), Some(42));
        assert_eq!(shared.take_seek_frame(), None);
    }

    #[test]
    fn media_time_frame_mapping_rounds_and_clamps_at_boundaries() {
        assert_eq!(frame_index_at_media_time(-1.0, 24, 1, 288), 0);
        assert_eq!(frame_index_at_media_time(1.0 / 48.0, 24, 1, 288), 1);
        assert_eq!(frame_index_at_media_time(30.0, 24, 1, 288), 287);
        assert_eq!(media_time_at_frame(288, 24, 1, 288), 287.0 / 24.0);
    }

    #[test]
    fn source_reset_invalidates_frames_renders_and_picks() {
        let shared = VideoEditingShared::default();
        shared.set_video_status(true, false);
        shared
            .update_video_frame_rgba(vec![1, 2, 3, 4], 1, 1, 7, 0.25)
            .unwrap();
        let source_generation = shared.source_generation.get();
        shared.request_overlay();
        let render_revision = shared.render_revision.get();
        shared.request_pick((3, 4));
        let pick = shared.pending_pick.get().unwrap();
        let rendered = RenderedVideoFrame {
            frame: IncomingVideoFrame {
                rgba: vec![1, 2, 3, 4],
                width: 1,
                height: 1,
                frame_index: 7,
                media_time_seconds: 0.25,
                source_generation,
            },
            render_revision,
            diagnostics: test_rendered_frame_diagnostics(),
        };
        let pick_result = PickResult {
            id: pick.id,
            source_generation,
            render_revision,
            hit: Some(0),
        };
        assert!(shared.accepts_render(&rendered));
        assert!(shared.accepts_pick(&pick_result));

        shared.set_video_status(false, false);
        assert!(!shared.accepts_render(&rendered));
        assert!(!shared.accepts_pick(&pick_result));
        assert!(shared.frame.borrow().is_none());
        assert!(shared.latest_video_frame.borrow().is_none());
        assert!(shared.pending_pick.get().is_none());
    }

    #[test]
    fn newer_scene_revision_invalidates_render_and_pick_completions() {
        let shared = VideoEditingShared::default();
        shared.request_overlay();
        let revision = shared.render_revision.get();
        shared.request_pick((3, 4));
        let pick = shared.pending_pick.get().unwrap();
        let rendered = RenderedVideoFrame {
            frame: IncomingVideoFrame {
                rgba: vec![1, 2, 3, 4],
                width: 1,
                height: 1,
                frame_index: 7,
                media_time_seconds: 0.25,
                source_generation: shared.source_generation.get(),
            },
            render_revision: revision,
            diagnostics: test_rendered_frame_diagnostics(),
        };
        let pick_result = PickResult {
            id: pick.id,
            source_generation: shared.source_generation.get(),
            render_revision: revision,
            hit: Some(0),
        };
        shared.request_overlay();
        assert!(!shared.accepts_render(&rendered));
        assert!(!shared.accepts_pick(&pick_result));
    }

    #[test]
    fn a_pick_click_stays_valid_when_the_same_frame_also_needs_a_render() {
        let shared = Rc::new(VideoEditingShared::default());
        let mut app = VideoEditingApp::new(document(), shared.clone());
        app.selected_quad = true;
        app.selected_asset = Some(CatalogAsset::CocaColaCan);

        // A primary click makes `image_panel` report *both* `needs_render` and a
        // pick point, so drive the real `settle_frame` with exactly that pair
        // rather than hand-rolling the order here — hand-rolling would pass even
        // if the app went back to handling the pick before the revision settled,
        // which is the bug (#205).
        let before = shared.render_revision.get();
        app.settle_frame(&egui::Context::default(), true, Some((3, 4)), None);
        let settled = shared.render_revision.get();
        assert_ne!(settled, before, "the render request must bump the revision");

        let pick = shared
            .pending_pick
            .get()
            .expect("the click requested a pick");
        assert_eq!(
            pick.render_revision, settled,
            "the pick must capture the revision the same frame's render request settled on"
        );
        assert_eq!(
            shared.render_revision.get(),
            settled,
            "the pick path must not bump the scene revision"
        );
        let result = PickResult {
            id: pick.id,
            source_generation: pick.source_generation,
            render_revision: pick.render_revision,
            hit: Some(0),
        };
        assert!(shared.accepts_pick(&result));
    }

    #[test]
    fn newer_pick_request_invalidates_older_pick_completion() {
        let shared = VideoEditingShared::default();
        shared.request_pick((1, 2));
        let first = shared.pending_pick.get().unwrap();
        shared.request_pick((3, 4));
        let result = PickResult {
            id: first.id,
            source_generation: first.source_generation,
            render_revision: first.render_revision,
            hit: Some(0),
        };
        assert!(!shared.accepts_pick(&result));
        assert_eq!(shared.pending_pick.get().unwrap().point, (3, 4));
    }

    #[test]
    fn diagnostics_keep_the_scene_bound_to_the_displayed_render() {
        let shared = Rc::new(VideoEditingShared::default());
        let mut app = VideoEditingApp::new(document(), shared);
        let mut rendered = test_rendered_frame_diagnostics();
        rendered.selected_asset = Some(CatalogAsset::Dragon);
        rendered.scene.materials[0].metallic = 0.25;
        rendered.scene.lighting = trd_core::Lighting {
            ambient: 0.0,
            scale: 0.0,
            ..trd_core::Lighting::default()
        };
        rendered.scene.environment_available = true;
        rendered.renderer.asset = Some(crate::video_editing_renderer::ImportedAssetDiagnostics {
            source_format: "GLB",
            aabb_min: [-1.0; 3],
            aabb_max: [1.0; 3],
            preview_scale: 1.0,
            imported_material: trd_core::DisneyMaterial {
                metallic: 1.0,
                auxiliary: trd_core::Auxiliary {
                    textures: trd_core::MaterialTextures {
                        metallic_roughness: true,
                        normal: true,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
        });
        app.displayed_frame_ready = true;
        app.displayed_frame_index = 0;
        app.last_rendered_frame_index = Some(0);
        app.displayed_diagnostics = Some(rendered);
        app.controller.state.materials[0].metallic = 0.9;

        let facts = app.displayed_facts();
        let imported = facts
            .renderer
            .as_ref()
            .and_then(|renderer| renderer.asset.as_ref())
            .map(|asset| &asset.imported_material)
            .unwrap();
        assert_eq!(facts.frame_index, Some(0));
        assert_eq!(facts.scene.materials[0].metallic, 0.25);
        assert_eq!(imported.metallic, 1.0);
        assert!(imported.auxiliary.textures.metallic_roughness);
        assert!(imported.auxiliary.textures.normal);
        assert_eq!(facts.scene.lighting.scale, 0.0);
        assert!(facts.scene.environment_available);
    }

    #[test]
    fn diagnostics_media_time_tracks_the_displayed_frame_not_a_newer_one() {
        let shared = Rc::new(VideoEditingShared::default());
        let mut app = VideoEditingApp::new(document(), shared.clone());
        assert_eq!(app.displayed_facts().media_time_seconds, None);

        let mut rendered = test_rendered_frame_diagnostics();
        rendered.media_time_seconds = 0.0;
        app.displayed_frame_ready = true;
        app.displayed_frame_index = 0;
        app.last_rendered_frame_index = Some(0);
        app.displayed_diagnostics = Some(rendered);

        // A newer frame arrives but has not reached the screen: the timeline
        // block must still describe frame 0, delta included.
        shared
            .update_video_frame_rgba(vec![1, 2, 3, 4], 1, 1, 5, 5.0 / 24.0)
            .unwrap();
        shared.set_video_media_observation(4, false);

        let facts = app.displayed_facts();
        assert_eq!(facts.presented_frame_index, None);
        assert_eq!(facts.frame_index, Some(0));
        assert_eq!(facts.media_time_seconds, Some(0.0));
        assert_eq!(
            facts
                .timeline_frame
                .as_ref()
                .map(|frame| frame.timestamp_us),
            Some(0)
        );
        assert_eq!(shared.video_media.get().ready_state, 4);
    }

    fn test_rendered_frame_diagnostics() -> RenderedFrameDiagnostics {
        RenderedFrameDiagnostics {
            media_time_seconds: 0.25,
            scene: crate::scene::SceneState::default(),
            selected_asset: None,
            selected_quad: false,
            move_direction: crate::interaction::MoveDirection::Reference1,
            playing: false,
            show_quad: false,
            show_quad_gizmo: false,
            draw_model: None,
            renderer: crate::video_editing_renderer::VideoRendererDiagnostics {
                identity: Rc::new(crate::video_editing_renderer::RendererIdentity {
                    adapter_name: "test".to_owned(),
                    backend: "test".to_owned(),
                    device_type: "test".to_owned(),
                }),
                target_size: (1, 1),
                pick_target_size: None,
                msaa_samples: 4,
                asset: None,
            },
        }
    }
}
