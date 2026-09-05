//! Shared browser/native video-editing state (#163/#167).
//! Sub-modules: [`editing_ui`] (panels/player), [`details_ui`] (inspector), [`diagnostics`] (domain math).

mod details_ui;
mod diagnostics;
mod editing_ui;
mod export;

pub use diagnostics::{PoseDeltaDiagnostics, QuadFrameDiagnostics, TrackingPlacementError};
pub use export::{decode_video_editing_input, ArrowExport, ArrowScene, VideoEditingInput};

use std::cell::{Cell, RefCell};
use std::rc::Rc;
// `std::time::Instant::now()` panics on wasm32; `web_time::Instant` is `performance.now()`-backed there.
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

/// Which path a surfaced error came from. A success clears only its own scope,
/// leaving other paths' errors standing (#329).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ErrorScope {
    Media,
    Catalog,
    Document,
    Render,
    Pick,
    Export,
}

impl ErrorScope {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Media => "media",
            Self::Catalog => "catalog",
            Self::Document => "document",
            Self::Render => "render",
            Self::Pick => "pick",
            Self::Export => "export",
        }
    }

    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::Media),
            2 => Some(Self::Catalog),
            3 => Some(Self::Document),
            4 => Some(Self::Render),
            5 => Some(Self::Pick),
            6 => Some(Self::Export),
            _ => None,
        }
    }

    pub const fn code(self) -> u8 {
        self as u8 + 1
    }
}

impl std::fmt::Display for ErrorScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

const COMMAND_NONE: u8 = 0;
const COMMAND_PICK_VIDEO: u8 = 1;
const COMMAND_PLAY: u8 = 2;
const COMMAND_PAUSE: u8 = 3;
const COMMAND_PICK_DOCUMENT: u8 = 4;
const COMMAND_LOAD_SELECTION: u8 = 5;
const COMMAND_EXPORT_ARROW: u8 = 6;

/// A source the dialog has selected but not loaded: picking and loading are separate steps (#264).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingSource {
    pub kind: VideoSourceKind,
    /// File name (local) or full URL (used as display text and fetch target).
    pub name: String,
}

/// Annotation-document formats accepted by the Open dialog. Extension is a hint only; the real loader sniffs bytes (#264).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentFormat {
    ArrowIpc,
    Parquet,
}

impl DocumentFormat {
    pub const EXTENSIONS: [&'static str; 2] = ["arrow", "parquet"];

    /// The format an extension suggests, or `None` for anything else.
    pub fn from_name(name: &str) -> Option<Self> {
        let extension = name.rsplit_once('.')?.1.to_ascii_lowercase();
        match extension.as_str() {
            "arrow" => Some(Self::ArrowIpc),
            "parquet" => Some(Self::Parquet),
            _ => None,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::ArrowIpc => "Arrow IPC",
            Self::Parquet => "Parquet",
        }
    }
}

/// One row's status line: the last validation verdict if there is one, else a
/// plain description of what is currently selected.
fn selection_label(
    ui: &mut egui::Ui,
    status: Option<&Result<String, String>>,
    fallback: impl FnOnce() -> String,
) {
    match status {
        Some(Ok(message)) => {
            ui.colored_label(egui::Color32::LIGHT_GREEN, format!("Selected: {message}"));
        }
        Some(Err(error)) => {
            ui.colored_label(egui::Color32::LIGHT_RED, error);
        }
        None => {
            ui.weak(fallback());
        }
    }
}

/// Whether a string is an `http`/`https` URL (the only accepted schemes).
pub fn is_http_url(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}

/// Why the placement overlay is or is not drawing at the current frame.
/// Three of the four silent reasons are normal for a sparse document (#264).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayState {
    /// The toggle is off.
    Hidden,
    /// No annotation document: there is nothing to draw anywhere.
    NoDocument,
    /// A document is loaded but does not annotate this frame.
    NotAnnotated(u32),
    /// Annotated, but marked video-only — the tail of a shot's tracking.
    VideoOnly(u32),
    /// Drawing.
    Drawing(u32),
}

impl OverlayState {
    /// What the panel says under the toggle.
    pub fn label(self) -> String {
        match self {
            Self::Hidden => "Overlay off: annotated frames play without their quad".to_owned(),
            Self::NoDocument => "Nothing to draw: no annotation document is loaded".to_owned(),
            Self::NotAnnotated(frame) => {
                format!("Nothing to draw: frame {frame} is not annotated (plain video)")
            }
            Self::VideoOnly(frame) => {
                format!("Nothing to draw: frame {frame} is annotated but not tracked")
            }
            Self::Drawing(frame) => format!("Drawing the quad on frame {frame}"),
        }
    }
}

/// Resolves the overlay's state. `tracked` is `None` for an unannotated frame; `show_overlay` is the combined toggle.
pub fn overlay_state(
    show_overlay: bool,
    has_document: bool,
    frame_index: u32,
    tracked: Option<bool>,
) -> OverlayState {
    if !show_overlay {
        return OverlayState::Hidden;
    }
    match (has_document, tracked) {
        (false, _) => OverlayState::NoDocument,
        (true, None) => OverlayState::NotAnnotated(frame_index),
        (true, Some(false)) => OverlayState::VideoOnly(frame_index),
        (true, Some(true)) => OverlayState::Drawing(frame_index),
    }
}

/// Whether the Load button has anything to commit: a freshly selected video,
/// or an already-playing video (so a document can be attached without re-picking, #264).
pub fn load_is_available(video_selected: bool, video_loaded: bool) -> bool {
    video_selected || video_loaded
}

/// What a loaded annotation document says about itself, compared against the playing video.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentSummary {
    /// The video the document was authored against.
    pub describes: String,
    /// How many frames carry placement data, and where they are.
    pub annotated: String,
    /// Set when the document is for a different video: a foreign document loads without error
    /// but silently places quads on the wrong frames (#264).
    pub mismatch: Option<String>,
}

/// Summarises `document` for display, against the `playing` timeline.
pub fn document_summary(
    document: &trd_core::VideoEditingDocument,
    playing: &trd_core::VideoInfo,
) -> DocumentSummary {
    let authored = &document.video;
    let describes = format!(
        "Authored for {} · {}x{} · {}/{} fps · {} frames",
        authored.source_name,
        authored.width,
        authored.height,
        authored.fps_num,
        authored.fps_den,
        authored.frame_count
    );

    let shots = document.shots();
    let ranges = shots
        .iter()
        .take(4)
        .map(|shot| format!("{}-{}", shot.start_frame, shot.end_frame))
        .collect::<Vec<_>>()
        .join(", ");
    let annotated = if shots.is_empty() {
        "No annotated frames: every frame is plain video".to_owned()
    } else {
        format!(
            "{} annotated frames in {} shot{}: {}{}",
            document.frames.len(),
            shots.len(),
            if shots.len() == 1 { "" } else { "s" },
            ranges,
            if shots.len() > 4 { ", …" } else { "" }
        )
    };

    // Resolution is compared, not names: a URL-opened video has its URL as name.
    let last_annotated = shots.last().map_or(0, |shot| shot.end_frame);
    let mismatch = if authored.width != playing.width || authored.height != playing.height {
        Some(format!(
            "This document is for {}x{} video, but {}x{} is playing — its quads belong to another clip",
            authored.width, authored.height, playing.width, playing.height
        ))
    } else if playing.frame_count <= last_annotated {
        Some(format!(
            "This document annotates up to frame {last_annotated}, but the video has only {} frames",
            playing.frame_count
        ))
    } else {
        None
    };

    DocumentSummary {
        describes,
        annotated,
        mismatch,
    }
}

/// Validates an annotation-document URL and infers its format from the suffix.
pub fn document_url_selection(url: &str) -> Result<DocumentFormat, String> {
    let url = url.trim();
    if url.is_empty() {
        return Err("enter a document URL, or leave it empty to play without one".to_owned());
    }
    if !is_http_url(url) {
        return Err("document URL must start with http:// or https://".to_owned());
    }
    // The path, without a query string or fragment — `?v=2` is not an extension.
    let path = url
        .split(['?', '#'])
        .next()
        .unwrap_or(url)
        .rsplit('/')
        .next()
        .unwrap_or(url);
    DocumentFormat::from_name(path).ok_or_else(|| {
        format!(
            "document URL should name a .{} or .{} file",
            DocumentFormat::EXTENSIONS[0],
            DocumentFormat::EXTENSIONS[1]
        )
    })
}

pub(crate) fn protocol_k_from_row_major(k: [f32; 9]) -> [f32; 9] {
    [k[0], k[3], k[6], k[1], k[4], k[7], k[2], k[5], k[8]]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoEditingCommand {
    OpenLocalVideo,
    /// Pick a local annotation document (`.arrow` / `.parquet`); optional (#264).
    OpenLocalDocument,
    /// Load the dialog's selection (video + optional document). Picking alone never loads.
    LoadSelection,
    Play,
    Pause,
    ExportArrow,
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

/// Media-element level state (not per-frame; `mediaTime` rides with the frame it belongs to).
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
    /// On-screen duration per container. Zero when unavailable.
    /// Reported only; seek retirement uses `answers_seek`, not this (#322).
    duration_seconds: f64,
    source_generation: u64,
    /// Seek generation this frame answers — the newest seek taken when the frame
    /// was handed over. Stamped at take time, so no older frame can slip through (#322).
    answers_seek: u64,
}

struct RenderedVideoFrame {
    frame: IncomingVideoFrame,
    render_revision: u64,
    diagnostics: RenderedFrameDiagnostics,
}

#[derive(Clone)]
struct RenderedFrameDiagnostics {
    media_time_seconds: f64,
    /// Pinned to the displayed frame (same reason as `media_time_seconds`).
    duration_seconds: f64,
    scene: crate::scene::SceneState,
    selected_asset: Option<CatalogAsset>,
    selected_quad: bool,
    move_direction: crate::interaction::MoveDirection,
    playing: bool,
    show_quad: bool,
    show_gizmos: bool,
    hovered_quad: bool,
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
    /// True when the render texture is bound directly into egui (no pixel readback).
    skip_readback: Cell<bool>,
    /// The toolkit's GPU context when the renderer shares eframe's device.
    /// Held (not just flagged) so rebuilt renderers reuse the same device.
    shared_gpu: RefCell<Option<std::sync::Arc<trd_core::GpuContext>>>,
    /// GPU-resident frame from the browser (always `None` natively, #229). Trait object to avoid
    /// naming `web_sys::VideoFrame` here (#302). Cloned rather than taken so repaints find it.
    external_frame: RefCell<Option<Rc<dyn trd_core::ExternalFrame>>>,
    /// Timeline probed from the container, consumed on the next frame (arrives late, after `moov`, #264).
    pending_video_info: RefCell<Option<trd_core::VideoInfo>>,
    /// Decoded document from the shell, or `Some(None)` to clear. Consumed next frame.
    /// Distinct from `pending_document` (the dialog's selection — this is the loaded result).
    incoming_document: RefCell<Option<Option<trd_core::VideoEditingDocument>>>,
    incoming_scene: RefCell<Option<Option<Rc<ArrowScene>>>>,
    command: Cell<u8>,
    asset_request: Cell<u8>,

    /// Selected but not yet loaded (shells fill local-file entries; the dialog fills URLs).
    pending_video: RefCell<Option<PendingSource>>,
    pending_document: RefCell<Option<PendingSource>>,
    seek_frame: Cell<i32>,
    /// Counts the seeks the timeline has **asked** for. A pending seek carries
    /// the value this had when it was made, which is what a delivered frame is
    /// matched against (#322).
    seek_generation: Cell<u64>,
    /// Newest seek actually taken by a shell. Frames are stamped with it.
    /// Differs from `seek_generation` while a request is queued: frames in that gap are old.
    dispatched_seek: Cell<u64>,
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
    export_asset: RefCell<Option<Rc<crate::video_editing_renderer::VideoExportAsset>>>,
    pending_export: RefCell<Option<ArrowExport>>,
    export_status: RefCell<Option<Result<String, String>>>,
    error: RefCell<Option<(ErrorScope, String)>>,
}

impl Default for VideoEditingShared {
    fn default() -> Self {
        Self {
            frame: RefCell::new(None),
            latest_video_frame: RefCell::new(None),
            rendered_frame: RefCell::new(None),
            context: RefCell::new(None),
            skip_readback: Cell::new(false),
            shared_gpu: RefCell::new(None),
            external_frame: RefCell::new(None),
            pending_video_info: RefCell::new(None),
            incoming_document: RefCell::new(None),
            incoming_scene: RefCell::new(None),
            command: Cell::new(COMMAND_NONE),
            asset_request: Cell::new(0),

            pending_video: RefCell::new(None),
            pending_document: RefCell::new(None),
            seek_frame: Cell::new(-1),
            seek_generation: Cell::new(0),
            dispatched_seek: Cell::new(0),
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
            export_asset: RefCell::new(None),
            pending_export: RefCell::new(None),
            export_status: RefCell::new(None),
            error: RefCell::new(None),
        }
    }
}

impl VideoEditingShared {
    /// Disables pixel readback when the render texture is bound directly into egui.
    pub fn set_skip_readback(&self, skip: bool) {
        self.skip_readback.set(skip);
    }

    /// Declares the toolkit's GPU context so rebuilt renderers reuse it.
    /// Must be set by the shell (wgpu `Device` has no identity comparison).
    pub fn set_shared_gpu(&self, gpu: std::sync::Arc<trd_core::GpuContext>) {
        self.shared_gpu.replace(Some(gpu));
    }

    /// Returns the toolkit's GPU context; rebuilt renderers must use it or their texture is unusable.
    pub fn shared_gpu(&self) -> Option<std::sync::Arc<trd_core::GpuContext>> {
        self.shared_gpu.borrow().clone()
    }

    /// Renderer's target view/size/identity for direct egui binding. `None` while in-flight or absent.
    pub fn target_binding(&self) -> Option<(wgpu::TextureView, (u32, u32), usize)> {
        let renderer = self.renderer.borrow();
        let renderer = renderer.as_ref()?;
        Some((
            renderer.target_view(),
            renderer.size(),
            renderer.renderer_generation_key(),
        ))
    }

    /// GPU-resident frame, cloned rather than taken so repeated repaints find it.
    fn external_frame(&self) -> Option<Rc<dyn trd_core::ExternalFrame>> {
        self.external_frame.borrow().clone()
    }

    /// Publishes a GPU-resident frame. Held until superseded (repaints reuse it).
    /// Requires an empty RGBA buffer; see also `update_video_frame_rgba` (#302).
    pub fn present_external_frame(
        &self,
        frame: Rc<dyn trd_core::ExternalFrame>,
        frame_index: u32,
        media_time_seconds: f64,
        duration_seconds: f64,
    ) -> Result<(), String> {
        let (width, height) = frame.size();
        if width == 0 || height == 0 {
            return Err(format!("video frame size {width}x{height} is degenerate"));
        }
        self.external_frame.replace(Some(frame));
        self.frame.replace(Some(IncomingVideoFrame {
            rgba: Vec::new(),
            width,
            height,
            frame_index,
            media_time_seconds,
            duration_seconds,
            source_generation: self.source_generation.get(),
            answers_seek: self.dispatched_seek.get(),
        }));
        self.request_repaint();
        Ok(())
    }

    pub fn update_video_frame_rgba(
        &self,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        frame_index: u32,
        media_time_seconds: f64,
        duration_seconds: f64,
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
            duration_seconds,
            source_generation: self.source_generation.get(),
            answers_seek: self.dispatched_seek.get(),
        }));
        self.request_repaint();
        Ok(())
    }

    /// Takes the pending command as its wire code, leaving `COMMAND_NONE`. Used by the JS bridge (#180).
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
        self.take_seek_frame().map_or(-1, |frame| frame as i32)
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

    pub fn set_error(&self, scope: ErrorScope, message: impl Into<String>) {
        self.error.replace(Some((scope, message.into())));
        self.request_repaint();
    }

    /// Clears only `scope`'s error; leaves other scopes' errors untouched (#329).
    pub fn clear_error(&self, scope: ErrorScope) {
        let held = self
            .error
            .borrow()
            .as_ref()
            .is_some_and(|(held, _)| *held == scope);
        if held {
            self.error.replace(None);
        }
    }

    /// The error to display, prefixed with the path that produced it.
    pub fn error_text(&self) -> Option<String> {
        self.error
            .borrow()
            .as_ref()
            .map(|(scope, message)| format!("{scope}: {message}"))
    }

    pub fn take_command(&self) -> Option<VideoEditingCommand> {
        match self.command.replace(COMMAND_NONE) {
            COMMAND_PICK_VIDEO => Some(VideoEditingCommand::OpenLocalVideo),
            COMMAND_PICK_DOCUMENT => Some(VideoEditingCommand::OpenLocalDocument),
            COMMAND_LOAD_SELECTION => Some(VideoEditingCommand::LoadSelection),
            COMMAND_PLAY => Some(VideoEditingCommand::Play),
            COMMAND_PAUSE => Some(VideoEditingCommand::Pause),
            COMMAND_EXPORT_ARROW => Some(VideoEditingCommand::ExportArrow),
            _ => None,
        }
    }

    pub fn take_asset_request(&self) -> Option<CatalogAsset> {
        CatalogAsset::from_code(self.asset_request.replace(0))
    }

    /// Hands the editor a timeline probed from the container. Applied on the
    /// next frame, since the shell learns it after the editor has started.
    pub fn set_pending_video_info(&self, video: trd_core::VideoInfo) {
        self.pending_video_info.replace(Some(video));
        self.request_repaint();
    }

    fn take_pending_video_info(&self) -> Option<trd_core::VideoInfo> {
        self.pending_video_info.borrow_mut().take()
    }

    pub fn queue_annotation_document(&self, document: trd_core::VideoEditingDocument) {
        self.incoming_document.replace(Some(Some(document)));
        self.request_repaint();
    }

    pub fn queue_arrow_scene(&self, scene: Rc<ArrowScene>) {
        self.incoming_scene.replace(Some(Some(scene)));
        self.request_repaint();
    }

    /// Drops the current annotation document or replay scene; the video keeps playing.
    pub fn clear_document(&self) {
        self.incoming_document.replace(Some(None));
        self.incoming_scene.replace(Some(None));
        self.request_repaint();
    }

    fn take_incoming_document(&self) -> Option<Option<trd_core::VideoEditingDocument>> {
        self.incoming_document.borrow_mut().take()
    }

    fn take_incoming_scene(&self) -> Option<Option<Rc<ArrowScene>>> {
        self.incoming_scene.borrow_mut().take()
    }

    /// Records a file-picker result without loading; displayed by the dialog until Load is pressed.
    pub fn set_pending_video(&self, source: Option<PendingSource>) {
        self.pending_video.replace(source);
    }

    pub fn pending_video(&self) -> Option<PendingSource> {
        self.pending_video.borrow().clone()
    }

    /// The optional annotation document selected alongside the video.
    pub fn set_pending_document(&self, source: Option<PendingSource>) {
        self.pending_document.replace(source);
    }

    pub fn pending_document(&self) -> Option<PendingSource> {
        self.pending_document.borrow().clone()
    }

    /// Records a seek and returns its id. Multiple requests before one take coalesce onto the newest id.
    fn request_seek(&self, frame_index: u32) -> u64 {
        let id = self.seek_generation.get().wrapping_add(1);
        self.seek_generation.set(id);
        self.seek_frame.set(frame_index as i32);
        id
    }

    /// Takes the pending seek target. Taking is the dispatch: all subsequent frames answer this seek.
    pub fn take_seek_frame(&self) -> Option<u32> {
        let frame = self.seek_frame.replace(-1);
        if frame < 0 {
            return None;
        }
        self.dispatched_seek.set(self.seek_generation.get());
        Some(frame as u32)
    }

    pub fn set_renderer(&self, renderer: crate::video_editing_renderer::VideoPlacementRenderer) {
        self.export_asset.replace(renderer.export_asset());
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
    /// Timeline of the playing video. Always present; the document is optional (#264).
    video: trd_core::VideoInfo,
    /// The annotation rows, when a document was supplied. `None` means the
    /// editor is a plain player: the placement UI is inert and every frame is
    /// just video.
    document: Option<trd_core::VideoEditingDocument>,
    /// A finished protocol scene replayed over the same source video.
    arrow_scene: Option<Rc<ArrowScene>>,
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
    /// The pointer is over the tracked quad. Purely a highlight, recomputed from
    /// the pointer every frame, so it never has to be cleared by hand.
    hovered_quad: bool,
    /// Whether the local grid + basis axes are drawn (independent of placement quads).
    show_gizmos: bool,
    /// Whether placement quads are drawn, including during playback (#264).
    show_placement_quads: bool,
    was_playing: bool,
    selected_asset: Option<CatalogAsset>,
    image_sizing: crate::ui::ImageSizing,
    fitted_render_size: (u32, u32),
    show_video_source_dialog: bool,
    video_url: String,
    /// URL input and last validation result (`Ok` = selected, `Err` = rejected).
    video_status: Option<Result<String, String>>,
    document_url: String,
    document_status: Option<Result<String, String>>,
    pending_seek: Option<PendingSeek>,
    last_pick_result: Option<Option<u32>>,
    /// The rendered texture bound directly into egui, when the shell shares
    /// trd's `wgpu::Device`. `None` means the portable readback path.
    native_texture: Option<egui::TextureId>,
    /// The `(renderer identity, size)` the current `native_texture` was
    /// registered for, so a resize or asset swap re-registers instead of
    /// sampling a freed view.
    native_texture_key: Option<(usize, (u32, u32))>,
}

impl VideoEditingApp {
    /// The editor over an annotation document — the document's own video
    /// metadata is the timeline.
    pub fn new(document: trd_core::VideoEditingDocument, shared: Rc<VideoEditingShared>) -> Self {
        Self::with_timeline(document.video.clone(), Some(document), shared)
    }

    /// The editor as a plain player: container timeline, no annotation document (#264).
    pub fn player(video: trd_core::VideoInfo, shared: Rc<VideoEditingShared>) -> Self {
        Self::with_timeline(video, None, shared)
    }

    fn with_timeline(
        video: trd_core::VideoInfo,
        document: Option<trd_core::VideoEditingDocument>,
        shared: Rc<VideoEditingShared>,
    ) -> Self {
        let source_size = (video.width.max(1), video.height.max(1));
        let scene = crate::scene::SceneState::default();
        let mut controller = crate::interaction::InteractionController::new(scene);
        controller.target = crate::interaction::InteractionTarget::Object;
        controller.move_direction = crate::interaction::MoveDirection::Reference1;
        controller.move_reference_axes = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
        controller.state.camera.distance = 1.0;
        Self {
            video,
            document,
            arrow_scene: None,
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
            hovered_quad: false,
            show_gizmos: false,
            show_placement_quads: true,
            was_playing: false,
            selected_asset: None,
            image_sizing: crate::ui::ImageSizing::FitCanvas,
            fitted_render_size: source_size,
            show_video_source_dialog: false,
            video_url: String::new(),
            video_status: None,
            document_url: String::new(),
            document_status: None,
            pending_seek: None,
            last_pick_result: None,
            native_texture: None,
            native_texture_key: None,
        }
    }

    /// Returns the annotation row for `frame_index`, or `None` for unannotated frames (normal for sparse docs).
    pub fn frame_row(&self, frame_index: u32) -> Option<&trd_core::VideoEditingFrame> {
        self.document.as_ref()?.frame(frame_index)
    }

    /// The document's annotated runs — what the Shots list navigates. Empty
    /// without a document.
    pub fn shots(&self) -> Vec<trd_core::Shot> {
        self.document
            .as_ref()
            .map(|document| document.shots())
            .unwrap_or_default()
    }

    /// Whether an annotation document is loaded at all. Drives the "inert but
    /// honest" state of the placement UI.
    pub fn has_document(&self) -> bool {
        self.document.is_some()
    }

    /// Adopts or drops an annotation document while the video keeps playing.
    /// Selection/object/pick are cleared (they refer to the old document's quads, #264).
    pub fn set_document(&mut self, document: Option<trd_core::VideoEditingDocument>) {
        self.document = document;
        self.arrow_scene = None;
        self.selected_quad = false;
        self.hovered_quad = false;
        self.selected_asset = None;
        self.last_pick_result = None;
        self.controller.state.selected = None;
        self.controller.state.objects[0] = crate::scene::ObjectTransform::default();
        self.shared.clear_export_asset();
        self.shared.cancel_arrow_export();
        self.shared.request_overlay();
    }

    pub fn set_arrow_scene(&mut self, scene: Option<Rc<ArrowScene>>) {
        if self.shared.video_loaded.get() {
            if let Some(error) = scene
                .as_deref()
                .and_then(|scene| self.arrow_scene_validation_error(scene))
            {
                self.shared.set_error(ErrorScope::Document, error);
                return;
            }
        }
        self.document = None;
        if let Some(scene) = scene.as_ref() {
            if let Some(operator) = scene.tonemap {
                self.controller.state.tone_mappings[0].operator = operator;
                self.controller
                    .state
                    .environment_background_tone_mapping
                    .operator = operator;
            }
            if let Some(mode) = scene
                .frames
                .iter()
                .filter_map(|frame| frame.draws.as_ref())
                .flatten()
                .find_map(|draw| match draw.selection {
                    trd_core::DrawSelection::Mesh(Some(mode)) => Some(mode),
                    _ => None,
                })
            {
                self.controller.state.modes[0] = mode;
            }
            if let Some(renderer) = self.shared.renderer.borrow().as_ref() {
                let (material, lighting) = renderer.replay_defaults();
                self.controller.state.materials[0] = material;
                self.controller.state.lighting = lighting;
                self.controller.state.environment_available = true;
            }
        }
        self.arrow_scene = scene;
        self.selected_quad = false;
        self.hovered_quad = false;
        self.selected_asset = None;
        self.last_pick_result = None;
        self.controller.state.selected = None;
        self.controller.state.objects[0] = crate::scene::ObjectTransform::default();
        self.shared.clear_export_asset();
        self.shared.cancel_arrow_export();
        self.shared.clear_error(ErrorScope::Document);
        self.shared.request_overlay();
    }

    /// Resets all editing state (selection, asset, transform, overlays, pick) without touching the media.
    pub fn reset_all(&mut self) {
        self.selected_quad = false;
        self.hovered_quad = false;
        self.selected_asset = None;
        self.last_pick_result = None;
        self.show_placement_quads = true;
        self.show_gizmos = false;
        self.controller.state = crate::scene::SceneState::default();
        self.controller.target = crate::interaction::InteractionTarget::Camera;
        self.controller.mode = crate::interaction::TransformMode::default();
        self.controller.move_direction = crate::interaction::MoveDirection::default();
        self.controller.rebase_reset();
        self.shared.clear_export_asset();
        self.shared.cancel_arrow_export();
        // Drop the renderer to clear the GPU-side asset too.
        self.shared.renderer.borrow_mut().take();
        self.shared.asset_request.set(0);
        self.shared.request_overlay();
    }

    /// Replaces the timeline (arrives late after moov). Clamps the playhead to the new range.
    pub fn set_video_info(&mut self, video: trd_core::VideoInfo) {
        let last = video.frame_count.saturating_sub(1);
        self.video = video;
        self.current_frame_index = self.current_frame_index.min(last);
        self.displayed_frame_index = self.displayed_frame_index.min(last);
        self.pending_seek = None;
        if let Some(error) = self
            .arrow_scene
            .as_deref()
            .and_then(|scene| self.arrow_scene_validation_error(scene))
        {
            self.arrow_scene = None;
            self.shared.set_error(ErrorScope::Document, error);
        }
    }

    fn arrow_scene_validation_error(&self, scene: &ArrowScene) -> Option<String> {
        let stored = self.video.frame_count as usize;
        if scene.is_frame_indexed() {
            return scene
                .frames
                .iter()
                .filter_map(|frame| frame.video_frame_index)
                .find(|index| *index as usize >= stored)
                .map(|index| {
                    format!(
                        "protocol scene references video frame {index}, but the video stores \
                         {stored} frames"
                    )
                });
        }
        let allowed_tail = self
            .video
            .unpresented_tail
            .map_or(1, |tail| tail.samples.max(1)) as usize;
        let missing_tail = stored.saturating_sub(scene.frames.len());
        (scene.frames.is_empty() || scene.frames.len() > stored || missing_tail > allowed_tail)
            .then(|| {
                format!(
                    "protocol scene has {} params rows, but the video stores {stored} frames \
                     (at most {allowed_tail} trailing unpresented frame(s) may be omitted)",
                    scene.frames.len(),
                )
            })
    }

    /// Registers the render texture directly in egui (no readback) when sharing a device.
    /// Keyed on (identity, size): resize or asset swap triggers re-registration.
    fn sync_native_texture(&mut self, frame: &mut eframe::Frame) {
        if self.shared.shared_gpu().is_none() {
            return;
        }
        let Some(state) = frame.wgpu_render_state() else {
            return;
        };
        let Some((view, size, key)) = self.shared.target_binding() else {
            return;
        };
        if self.native_texture_key == Some((key, size)) {
            return;
        }
        let mut renderer = state.renderer.write();
        if let Some(old) = self.native_texture.take() {
            renderer.free_texture(&old);
        }
        let id = renderer.register_native_texture(&state.device, &view, wgpu::FilterMode::Linear);
        drop(renderer);
        self.native_texture = Some(id);
        self.native_texture_key = Some((key, size));
        self.shared.set_skip_readback(true);
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
        egui::Window::new("Open source")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(context, |ui| {
                ui.set_min_width(460.0);
                // The rows scroll; the Load button does not. A dialog whose
                // commit point can be pushed off-screen by its own explanatory
                // text is a dialog with no commit point.
                egui::ScrollArea::vertical()
                    .max_height(360.0)
                    .show(ui, |ui| {
                        self.video_source_row(ui);
                        ui.separator();
                        self.document_source_row(ui);
                    });
                ui.separator();
                close = self.load_row(ui);
            });
        self.show_video_source_dialog = open && !close;
    }

    /// The video row. Selecting a file or URL waits for the Load button (#264).
    fn video_source_row(&mut self, ui: &mut egui::Ui) {
        ui.heading("Video");
        ui.label("The video to play. Required.");
        ui.horizontal(|ui| {
            if ui.button("Select local file...").clicked() {
                self.shared.command.set(COMMAND_PICK_VIDEO);
            }
            if ui.button("Clear").clicked() {
                self.video_url.clear();
                self.video_status = None;
                self.shared.set_pending_video(None);
            }
        });

        ui.label("Video URL");
        let response = ui.add(
            egui::TextEdit::singleline(&mut self.video_url)
                .hint_text("https://example.com/video.mp4")
                .desired_width(f32::INFINITY),
        );
        let submit = response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
        if ui.button("Use this URL").clicked() || submit {
            let url = self.video_url.trim().to_owned();
            self.video_status = Some(if is_http_url(&url) {
                self.shared.set_pending_video(Some(PendingSource {
                    kind: VideoSourceKind::HttpUrl,
                    name: url.clone(),
                }));
                Ok(url)
            } else {
                Err("video URL must start with http:// or https://".to_owned())
            });
        }
        selection_label(ui, self.video_status.as_ref(), || {
            match self.shared.pending_video() {
                Some(source) => format!("Selected: {}", source.name),
                None => "No video selected".to_owned(),
            }
        });
        ui.weak("A URL must allow cross-origin video frame access.");
    }

    /// Optional annotation document or exported protocol scene.
    fn document_source_row(&mut self, ui: &mut egui::Ui) {
        ui.heading("Arrow input (optional)");
        ui.label("An annotation Arrow/Parquet document, or an exported protocol 0.0.6 scene.");
        ui.weak("Annotations are editable; an exported scene replays over the selected video.");

        ui.horizontal(|ui| {
            if ui.button("Select local file...").clicked() {
                self.shared.command.set(COMMAND_PICK_DOCUMENT);
            }
            if ui.button("Clear").clicked() {
                self.document_url.clear();
                self.document_status = None;
                self.shared.set_pending_document(None);
            }
        });
        ui.weak("Load applies the whole selection: with no Arrow input the video plays unchanged.");

        ui.label("Arrow input URL");
        let response = ui.add(
            egui::TextEdit::singleline(&mut self.document_url)
                .hint_text("https://example.com/shot.arrow")
                .desired_width(f32::INFINITY),
        );
        let submit = response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
        if ui.button("Use this URL").clicked() || submit {
            let url = self.document_url.trim().to_owned();
            self.document_status = Some(match document_url_selection(&url) {
                Ok(format) => {
                    self.shared.set_pending_document(Some(PendingSource {
                        kind: VideoSourceKind::HttpUrl,
                        name: url.clone(),
                    }));
                    Ok(format!("{url} · {}", format.label()))
                }
                Err(error) => Err(error),
            });
        }
        selection_label(ui, self.document_status.as_ref(), || {
            match self.shared.pending_document() {
                Some(source) => format!("Selected: {}", source.name),
                None => "No Arrow input — the video plays as-is".to_owned(),
            }
        });
        ui.weak("Format is decided by the file's contents, not its name.");
    }

    /// The single commit point. Returns whether it was pressed, so the dialog
    /// closes only on an actual load.
    fn load_row(&mut self, ui: &mut egui::Ui) -> bool {
        let ready = load_is_available(
            self.shared.pending_video().is_some(),
            self.shared.video_loaded.get(),
        );
        let clicked = ui
            .add_enabled(ready, egui::Button::new("Load"))
            .on_disabled_hover_text("Select a video first — the document is optional")
            .clicked();
        if clicked {
            self.shared.command.set(COMMAND_LOAD_SELECTION);
        }
        if !ready {
            ui.weak("Load becomes available once a video is selected.");
        }
        clicked
    }

    fn set_display_frame(&mut self, rendered: RenderedVideoFrame) {
        let frame = &rendered.frame;
        self.display_size = (frame.width, frame.height);
        self.displayed_frame_index = frame.frame_index;
        self.displayed_frame_ready = true;
        self.last_rendered_frame_index = Some(frame.frame_index);
        self.displayed_diagnostics = Some(rendered.diagnostics);
        if let Some(pending) = self.pending_seek {
            if pending.settled_by(frame.answers_seek) {
                self.pending_seek = None;
            }
        }
        // Shared-device path: no readback, so no pixels to upload. Skipping is a panic.
        if frame.rgba.is_empty() {
            return;
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
        let background_frame = self.frame_row(video.frame_index).cloned();
        let replay_frame = self
            .arrow_scene
            .as_ref()
            .and_then(|scene| scene.frame(video.frame_index))
            .cloned();
        let quad_frame = self.quad_frame_at(video.frame_index);
        // Overlay follows the toggles, not play state (#264).
        let tracked = background_frame.as_ref().is_some_and(|frame| frame.tracked);
        let show_quad = self.show_placement_quads && tracked;
        let show_gizmos = self.show_gizmos && tracked;
        let quad_overlay = crate::video_editing_renderer::QuadOverlay {
            model: quad_frame.map(trd_placement::quad_outline_model),
            axes: quad_frame.map(trd_placement::quad_axes_model),
            show_quads: show_quad,
            show_gizmos,
            hovered: self.hovered_quad,
            selected: self.selected_quad,
        };
        let show_object = self.selected_asset.is_some()
            && background_frame.as_ref().is_some_and(|frame| frame.tracked);
        let placement_frame = show_object.then(|| background_frame.clone()).flatten();
        let model = if show_object {
            self.placement_model_at(video.frame_index)
        } else {
            None
        };
        let Some(mut renderer) = self.shared.renderer.borrow_mut().take() else {
            return;
        };
        // Size from the decoded frame (not the document): `--preview-width` may decode below
        // source res. Calibration K stays at document size; `frame_camera` rescales it (#168/#170).
        let decoded_size = (video.width.max(1), video.height.max(1));
        let requested_size = match self.image_sizing {
            crate::ui::ImageSizing::FitCanvas => (
                self.fitted_render_size.0.min(decoded_size.0).max(1),
                self.fitted_render_size.1.min(decoded_size.1).max(1),
            ),
            crate::ui::ImageSizing::OriginalResolution => decoded_size,
        };
        if let Err(error) = renderer.resize(requested_size.0, requested_size.1) {
            self.shared.renderer.replace(Some(renderer));
            self.shared.set_error(ErrorScope::Render, error);
            return;
        }
        self.shared
            .renderer_diagnostics
            .replace(Some(renderer.diagnostics()));
        let render_size = renderer.size();
        self.shared.needs_overlay.set(false);
        let mut state = self.controller.state.clone();
        let replay_tonemap = state.tone_mappings[0].operator;
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
        let width = self.video.width;
        let height = self.video.height;
        let selected_asset = self.selected_asset;
        let selected_quad = self.selected_quad;
        let hovered_quad = self.hovered_quad;
        let move_direction = self.controller.move_direction;
        let rendered_model = replay_frame
            .as_ref()
            .and_then(|frame| frame.draws.as_ref())
            .and_then(|draws| draws.first())
            .map(|draw| draw.model)
            .or(model);
        let background_frame_index = video.frame_index;
        let background_media_time = video.media_time_seconds;
        let background_duration = video.duration_seconds;
        let answers_seek = video.answers_seek;
        let render_started = Instant::now();
        // GPU-path: `present_external_frame` frame has no RGBA bytes; always None natively (#302).
        let external_frame = shared.external_frame().filter(|_| video.rgba.is_empty());
        let render = async move {
            let source = match external_frame.as_deref() {
                Some(frame) => crate::video_editing_renderer::FrameSource::External(frame),
                None => crate::video_editing_renderer::FrameSource::Rgba(&video.rgba),
            };
            let result = if shared.skip_readback.get() {
                // Shared-device: no readback; empty Vec signals `set_display_frame` to skip upload.
                match replay_frame.as_ref() {
                    Some(frame) => renderer.draw_scene_frame(
                        source,
                        video.width,
                        video.height,
                        (width, height),
                        frame,
                        replay_tonemap,
                    ),
                    None => renderer.draw(
                        source,
                        video.width,
                        video.height,
                        (width, height),
                        background_frame.as_ref(),
                        quad_overlay,
                        placement_frame.as_ref(),
                        model,
                        &state,
                    ),
                }
                .map(|()| Vec::new())
            } else {
                match replay_frame.as_ref() {
                    Some(frame) => {
                        renderer
                            .render_scene_frame(
                                &video.rgba,
                                video.width,
                                video.height,
                                (width, height),
                                frame,
                                replay_tonemap,
                            )
                            .await
                    }
                    None => {
                        renderer
                            .render(
                                &video.rgba,
                                video.width,
                                video.height,
                                (width, height),
                                background_frame.as_ref(),
                                quad_overlay,
                                placement_frame.as_ref(),
                                model,
                                &state,
                            )
                            .await
                    }
                }
            };
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
                    shared.clear_error(ErrorScope::Render);
                    shared.rendered_frame.replace(Some(RenderedVideoFrame {
                        frame: IncomingVideoFrame {
                            rgba,
                            width: render_size.0,
                            height: render_size.1,
                            frame_index: background_frame_index,
                            media_time_seconds: background_media_time,
                            duration_seconds: background_duration,
                            source_generation,
                            answers_seek,
                        },
                        render_revision,
                        diagnostics: RenderedFrameDiagnostics {
                            media_time_seconds: background_media_time,
                            duration_seconds: background_duration,
                            scene: state,
                            selected_asset,
                            selected_quad,
                            move_direction,
                            playing: rendered_playing,
                            show_quad,
                            show_gizmos,
                            hovered_quad,
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
                    shared.set_error(ErrorScope::Render, error);
                }
                (false, _) => {}
            }
            shared.render_in_flight.set(false);
            shared.render_in_flight_frame.set(None);
            if let Some(context) = shared.context.borrow().as_ref() {
                context.request_repaint();
            }
        };
        crate::platform::drive(render);
    }

    fn schedule_pick(&self) {
        if self.shared.render_in_flight.get() || self.shared.pick_in_flight.get() {
            return;
        }
        let Some(request) = self.shared.pending_pick.take() else {
            return;
        };
        let Some(frame) = self.frame_row(self.displayed_frame_index).cloned() else {
            return;
        };
        let Some(model) = self.placement_model_at(self.displayed_frame_index) else {
            return;
        };
        let Some(mut renderer) = self.shared.renderer.borrow_mut().take() else {
            self.shared.pending_pick.set(Some(request));
            return;
        };
        let source_size = (self.video.width, self.video.height);
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
                    shared.clear_error(ErrorScope::Pick);
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
                    shared.set_error(ErrorScope::Pick, error);
                }
                (false, _) => {}
            }
            shared.pick_in_flight.set(false);
            if let Some(context) = shared.context.borrow().as_ref() {
                context.request_repaint();
            }
        };
        crate::platform::drive(pick);
    }

    fn quad_frame_at(&self, frame_index: u32) -> Option<trd_placement::QuadFrame> {
        self.quad_frame_result_at(frame_index).ok()
    }

    fn quad_frame_result_at(
        &self,
        frame_index: u32,
    ) -> Result<trd_placement::QuadFrame, TrackingPlacementError> {
        let frame = self
            .frame_row(frame_index)
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
        let object_model = object.model_matrix();
        Some(quad_basis * object_model)
    }

    /// Derives the values the Details panel can't read directly: displayed-frame pin + domain math.
    pub(super) fn displayed_facts(&self) -> DisplayedFacts {
        let displayed_frame_index = self
            .displayed_frame_ready
            .then_some(self.displayed_frame_index);
        let timeline_frame = displayed_frame_index.and_then(|index| self.frame_row(index));
        let media_time_seconds = self
            .displayed_frame_ready
            .then(|| {
                self.displayed_diagnostics
                    .as_ref()
                    .map(|displayed| displayed.media_time_seconds)
            })
            .flatten();
        let frame_duration_seconds = self
            .displayed_diagnostics
            .as_ref()
            .map(|displayed| displayed.duration_seconds)
            .filter(|duration| *duration > 0.0);
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
                self.frame_row(previous_index)
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
        let show_gizmos = displayed.is_some_and(|d| d.show_gizmos);
        let quad_washed = show_quad && displayed.is_some_and(|d| d.hovered_quad || d.selected_quad);
        let background_drawables =
            1 + u32::from(show_quad) + u32::from(quad_washed) + if show_gizmos { 2 } else { 0 };
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
            frame_duration_seconds,
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
            seek_target: self.pending_seek.map(|pending| pending.frame_index),
            latest_pick_result: self.last_pick_result,
            shared: self.shared.clone(),
        }
    }
}

/// Derived Details values (live observations are read directly at draw time).
pub(super) struct DisplayedFacts {
    pub frame_index: Option<u32>,
    pub media_time_seconds: Option<f64>,
    /// The displayed frame's own declared duration, or `None` when the shell
    /// could not say.
    pub frame_duration_seconds: Option<f64>,
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

/// An outstanding seek: carries the id minted at request time, not a timestamp.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PendingSeek {
    frame_index: u32,
    id: u64,
}

impl PendingSeek {
    /// True when `answers_seek >= self.id` — the frame answers this seek or a superseding one (#322).
    fn settled_by(self, answers_seek: u64) -> bool {
        answers_seek >= self.id
    }
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
                unpresented_tail: None,
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
    fn a_decoded_frame_clears_only_the_media_error() {
        let shared = VideoEditingShared::default();
        shared.set_error(ErrorScope::Catalog, "catalog asset missing");

        shared.clear_error(ErrorScope::Media);

        assert_eq!(
            shared.error_text().as_deref(),
            Some("catalog: catalog asset missing")
        );
    }

    #[test]
    fn a_scope_clears_its_own_error() {
        let shared = VideoEditingShared::default();
        shared.set_error(ErrorScope::Media, "decoder ended early");

        shared.clear_error(ErrorScope::Media);

        assert_eq!(shared.error_text(), None);
    }

    #[test]
    fn error_scope_codes_round_trip() {
        for scope in [
            ErrorScope::Media,
            ErrorScope::Catalog,
            ErrorScope::Document,
            ErrorScope::Render,
            ErrorScope::Pick,
            ErrorScope::Export,
        ] {
            assert_eq!(ErrorScope::from_code(scope.code()), Some(scope));
        }
        assert_eq!(ErrorScope::from_code(0), None);
    }

    #[test]
    fn newest_incoming_frame_replaces_the_pending_frame() {
        let shared = VideoEditingShared::default();
        shared
            .update_video_frame_rgba(vec![1, 2, 3, 4], 1, 1, 7, 0.25, 0.0)
            .unwrap();
        shared
            .update_video_frame_rgba(vec![5, 6, 7, 8], 1, 1, 9, 0.5, 0.0)
            .unwrap();

        let frame = shared.frame.borrow_mut().take().unwrap();
        assert_eq!(frame.frame_index, 9);
        assert_eq!(frame.rgba, vec![5, 6, 7, 8]);
    }

    #[test]
    fn invalid_incoming_frame_does_not_replace_the_pending_frame() {
        let shared = VideoEditingShared::default();
        shared
            .update_video_frame_rgba(vec![1, 2, 3, 4], 1, 1, 7, 0.25, 0.0)
            .unwrap();
        assert!(shared
            .update_video_frame_rgba(vec![5, 6, 7], 1, 1, 9, 0.5, 0.0)
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

        shared.request_seek(12);
        shared.request_seek(42);
        assert_eq!(shared.take_seek_frame(), Some(42));
        assert_eq!(shared.take_seek_frame(), None);
    }

    #[test]
    fn overlay_state_names_each_reason_it_draws_nothing() {
        assert_eq!(
            overlay_state(false, true, 7, Some(true)),
            OverlayState::Hidden,
            "the toggle wins over everything else"
        );
        assert_eq!(
            overlay_state(true, false, 7, None),
            OverlayState::NoDocument
        );
        assert_eq!(
            overlay_state(true, true, 7, None),
            OverlayState::NotAnnotated(7),
            "a sparse document simply has no row here"
        );
        assert_eq!(
            overlay_state(true, true, 7, Some(false)),
            OverlayState::VideoOnly(7)
        );
        assert_eq!(
            overlay_state(true, true, 7, Some(true)),
            OverlayState::Drawing(7)
        );

        assert!(overlay_state(true, true, 7, None).label().contains("7"));
        assert!(
            overlay_state(true, false, 7, None)
                .label()
                .contains("no annotation document"),
            "the commonest case names the missing document"
        );
    }

    /// Document summary flags a foreign document (different clip = wrong frames).
    #[test]
    fn document_summary_reports_contents_and_flags_a_foreign_document() {
        let authored = trd_core::VideoInfo {
            source_name: "shot_0001.mp4".to_owned(),
            mime: "video/mp4".to_owned(),
            codec: "h264".to_owned(),
            sha256: String::new(),
            byte_length: 6_664_274,
            width: 1920,
            height: 1080,
            fps_num: 24,
            fps_den: 1,
            frame_count: 288,
            duration_us: 12_000_000,
            unpresented_tail: None,
        };
        let document = trd_core::VideoEditingDocument {
            video: authored.clone(),
            poster_bytes: Vec::new(),
            frames: (0..3)
                .chain(10..12)
                .map(|index| trd_core::VideoEditingFrame {
                    video_frame_index: index,
                    present_index: index,
                    timestamp_us: 0,
                    k: None,
                    placement_quad: None,
                    tracked: true,
                })
                .collect(),
        };

        let matching = document_summary(&document, &authored);
        assert_eq!(
            matching.describes,
            "Authored for shot_0001.mp4 · 1920x1080 · 24/1 fps · 288 frames"
        );
        assert_eq!(
            matching.annotated,
            "5 annotated frames in 2 shots: 0-2, 10-11"
        );
        assert_eq!(
            matching.mismatch, None,
            "the document describes the video that is playing"
        );

        // The case this session hit: a 1080p document attached to a 4K recording.
        let four_k = trd_core::VideoInfo {
            source_name: "2026-07-16 20-52-51.mp4".to_owned(),
            width: 3840,
            height: 2160,
            frame_count: 694_840,
            ..authored.clone()
        };
        let foreign = document_summary(&document, &four_k);
        assert!(
            foreign
                .mismatch
                .as_deref()
                .is_some_and(|text| text.contains("1920x1080") && text.contains("3840x2160")),
            "a resolution difference names both sides: {:?}",
            foreign.mismatch
        );

        // Same resolution, but the video is too short for the rows it carries.
        let truncated = trd_core::VideoInfo {
            frame_count: 5,
            ..authored.clone()
        };
        assert!(
            document_summary(&document, &truncated)
                .mismatch
                .as_deref()
                .is_some_and(|text| text.contains("frame 11")),
            "rows past the end of the video are named"
        );

        let empty = trd_core::VideoEditingDocument {
            video: authored.clone(),
            poster_bytes: Vec::new(),
            frames: Vec::new(),
        };
        assert_eq!(
            document_summary(&empty, &authored).annotated,
            "No annotated frames: every frame is plain video"
        );
    }

    #[test]
    fn load_is_available_for_a_new_selection_or_an_already_playing_video() {
        assert!(
            load_is_available(true, false),
            "a freshly selected video is the ordinary case"
        );
        assert!(
            load_is_available(false, true),
            "a document alone must commit against the video already on screen"
        );
        assert!(load_is_available(true, true));
        assert!(
            !load_is_available(false, false),
            "with neither, Load has nothing to act on"
        );
    }

    #[test]
    fn document_url_selection_names_the_format_or_says_why_not() {
        assert_eq!(
            document_url_selection("https://example.com/shot.parquet"),
            Ok(DocumentFormat::Parquet)
        );
        assert_eq!(
            document_url_selection("  http://example.com/a/b/shot.ARROW  "),
            Ok(DocumentFormat::ArrowIpc),
            "extensions are case-insensitive and the input is trimmed"
        );
        assert_eq!(
            document_url_selection("https://example.com/shot.arrow?v=2#row"),
            Ok(DocumentFormat::ArrowIpc),
            "a query string is not part of the extension"
        );

        assert!(document_url_selection("").is_err());
        assert!(document_url_selection("file:///tmp/shot.arrow").is_err());
        assert!(document_url_selection("https://example.com/shot.mp4").is_err());
    }

    #[test]
    fn document_format_follows_the_extension_only_as_a_hint() {
        assert_eq!(
            DocumentFormat::from_name("fiba-shot1.arrow"),
            Some(DocumentFormat::ArrowIpc)
        );
        assert_eq!(
            DocumentFormat::from_name("tracks.Parquet"),
            Some(DocumentFormat::Parquet)
        );
        assert_eq!(DocumentFormat::from_name("tracks"), None);
        assert_eq!(DocumentFormat::from_name("tracks.csv"), None);
    }

    #[test]
    fn picking_a_document_is_its_own_command() {
        let shared = VideoEditingShared::default();
        shared.command.set(COMMAND_PICK_DOCUMENT);
        assert_eq!(
            shared.take_command(),
            Some(VideoEditingCommand::OpenLocalDocument)
        );
        shared.command.set(COMMAND_LOAD_SELECTION);
        assert_eq!(
            shared.take_command(),
            Some(VideoEditingCommand::LoadSelection)
        );
    }

    #[test]
    fn pending_sources_are_selected_independently() {
        let shared = VideoEditingShared::default();
        assert_eq!(shared.pending_video(), None);

        shared.set_pending_video(Some(PendingSource {
            kind: VideoSourceKind::LocalFile,
            name: "shot.mp4".to_owned(),
        }));
        shared.set_pending_document(Some(PendingSource {
            kind: VideoSourceKind::HttpUrl,
            name: "https://example.com/shot.parquet".to_owned(),
        }));

        shared.set_pending_document(None);
        assert_eq!(
            shared.pending_video().map(|source| source.name).as_deref(),
            Some("shot.mp4"),
            "clearing the document dropped the video selection"
        );
        assert_eq!(shared.pending_document(), None);
    }

    #[test]
    fn media_time_frame_mapping_rounds_and_clamps_at_boundaries() {
        assert_eq!(frame_index_at_media_time(-1.0, 24, 1, 288), 0);
        assert_eq!(frame_index_at_media_time(1.0 / 48.0, 24, 1, 288), 1);
        assert_eq!(frame_index_at_media_time(30.0, 24, 1, 288), 287);
        assert_eq!(media_time_at_frame(288, 24, 1, 288), 287.0 / 24.0);
    }

    /// Recording timescale from #317/#319/#322 (nominally 60fps, `stts` alternates 272/256 units/frame).
    const TIMESCALE: f64 = 16_000.0;

    /// Delivers `(frame_index, media_time, duration)` as a shell would; 0×0 needs no egui texture.
    fn deliver(app: &mut VideoEditingApp, shared: &Rc<VideoEditingShared>, frame: (u32, f64, f64)) {
        let (frame_index, media_time_seconds, duration_seconds) = frame;
        shared
            .update_video_frame_rgba(
                Vec::new(),
                0,
                0,
                frame_index,
                media_time_seconds,
                duration_seconds,
            )
            .expect("an empty 0x0 frame is well formed");
        let frame = shared.frame.borrow().clone().expect("just published");
        app.set_display_frame(RenderedVideoFrame {
            frame,
            render_revision: shared.render_revision.get(),
            diagnostics: test_rendered_frame_diagnostics(),
        });
    }

    /// Runs one seek round-trip and reports whether it retired.
    fn seek_answered_by(requested: u32, delivered: (u32, f64, f64)) -> bool {
        let shared = Rc::new(VideoEditingShared::default());
        let mut app = VideoEditingApp::new(document(), shared.clone());
        app.pending_seek = Some(PendingSeek {
            frame_index: requested,
            id: shared.request_seek(requested),
        });
        assert_eq!(
            shared.take_seek_frame(),
            Some(requested),
            "the shell has to be handed the seek before it can answer one"
        );

        deliver(&mut app, &shared, delivered);
        app.pending_seek.is_none()
    }

    /// #322: ffmpeg overshoots (asked 6311, got 6312). The old window rule left the seek pending
    /// because delivered duration < gap. Identity-based retirement fixes this.
    #[test]
    fn an_overshooting_seek_retires_though_the_frame_landed_on_is_shorter() {
        let skipped_pts = 1_682_933.0 / TIMESCALE;
        let requested_time = skipped_pts + (16.0 / 3.0) / TIMESCALE;
        let delivered_pts = skipped_pts + 272.0 / TIMESCALE;
        let delivered_duration = 256.0 / TIMESCALE;

        // Verify the fixture falls outside the window rule (otherwise it proves nothing).
        assert!(
            (delivered_pts - requested_time).abs() >= delivered_duration,
            "fixture must fall outside the delivered frame's own presentation window"
        );

        assert!(
            seek_answered_by(6311, (6312, delivered_pts, delivered_duration)),
            "the frame the reader returned for this seek is its answer, whatever its timestamp says"
        );
    }

    /// #317: mediabunny undershoots (asked 95838, got 95837 at 1597.283 s).
    #[test]
    fn an_undershooting_seek_still_retires() {
        assert!(
            seek_answered_by(95_838, (95_837, 1597.283, 272.0 / TIMESCALE)),
            "the reader answered with the frame covering the instant — that is the answer"
        );
    }

    /// Frame decoded for the previous position must not retire a new seek that hasn't been taken yet.
    #[test]
    fn a_frame_delivered_before_the_shell_took_the_seek_does_not_retire_it() {
        let shared = Rc::new(VideoEditingShared::default());
        let mut app = VideoEditingApp::new(document(), shared.clone());
        assert_eq!(shared.take_seek_frame(), None);

        app.pending_seek = Some(PendingSeek {
            frame_index: 7517,
            id: shared.request_seek(7517),
        });

        deliver(&mut app, &shared, (7480, 124.7, 272.0 / TIMESCALE));
        assert!(
            app.pending_seek.is_some(),
            "a frame decoded before the shell took the seek is not an answer to it"
        );

        assert_eq!(shared.take_seek_frame(), Some(7517));
        deliver(&mut app, &shared, (7518, 125.3, 256.0 / TIMESCALE));
        assert!(
            app.pending_seek.is_none(),
            "the frame delivered for the seek retires it"
        );
    }

    /// Multiple seeks before one take coalesce; one frame retires them all.
    #[test]
    fn requests_coalesced_into_one_take_are_answered_by_a_single_frame() {
        let shared = Rc::new(VideoEditingShared::default());
        let mut app = VideoEditingApp::new(document(), shared.clone());
        shared.request_seek(6311);
        app.pending_seek = Some(PendingSeek {
            frame_index: 7517,
            id: shared.request_seek(7517),
        });

        assert_eq!(
            shared.take_seek_frame(),
            Some(7517),
            "the newest target wins the slot"
        );
        deliver(&mut app, &shared, (7518, 125.3, 256.0 / TIMESCALE));

        assert!(app.pending_seek.is_none());
    }

    /// A newer seek's answer also retires older outstanding seeks.
    #[test]
    fn a_newer_seeks_answer_retires_an_older_wait() {
        let older = PendingSeek {
            frame_index: 6311,
            id: 4,
        };

        assert!(older.settled_by(4), "its own answer");
        assert!(older.settled_by(5), "a newer seek has been answered since");
        assert!(
            !older.settled_by(3),
            "a frame from before this seek answers nothing"
        );
    }

    #[test]
    fn source_reset_invalidates_frames_renders_and_picks() {
        let shared = VideoEditingShared::default();
        shared.set_video_status(true, false);
        shared
            .update_video_frame_rgba(vec![1, 2, 3, 4], 1, 1, 7, 0.25, 0.0)
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
                duration_seconds: 0.0,
                source_generation,
                answers_seek: 0,
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
                duration_seconds: 0.0,
                source_generation: shared.source_generation.get(),
                answers_seek: 0,
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

        // Drive `settle_frame` directly so both render + pick are submitted in the real order (#205).
        let before = shared.render_revision.get();
        app.settle_frame(&egui::Context::default(), true, Some((3, 4)), None, None);
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
    fn selecting_a_quad_reveals_its_gizmos() {
        let shared = Rc::new(VideoEditingShared::default());
        let mut app = VideoEditingApp::new(document(), shared.clone());
        app.display_size = (1920, 1080);
        app.show_gizmos = false;
        let quad = Some([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]);

        app.handle_pick((50, 50), quad);
        assert!(app.selected_quad, "clicking inside selects the quad");
        assert!(app.show_gizmos, "selection reveals the local frame");

        app.handle_pick((500, 500), quad);
        assert!(!app.selected_quad, "clicking outside deselects it");
        assert!(!app.show_gizmos, "and takes its local frame away again");
    }

    #[test]
    fn hovering_the_quad_requests_one_overlay_per_change() {
        let shared = Rc::new(VideoEditingShared::default());
        let mut app = VideoEditingApp::new(document(), shared.clone());
        app.display_size = (1920, 1080);
        let quad = Some([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]);
        let ctx = egui::Context::default();

        let before = shared.render_revision.get();
        app.settle_frame(&ctx, false, None, Some((50, 50)), quad);
        assert!(app.hovered_quad, "the pointer is inside the quad");
        let entered = shared.render_revision.get();
        assert_ne!(entered, before, "entering asks for a new overlay");

        app.settle_frame(&ctx, false, None, Some((60, 60)), quad);
        assert!(app.hovered_quad);
        assert_eq!(
            shared.render_revision.get(),
            entered,
            "moving within the quad changes nothing to draw"
        );

        app.settle_frame(&ctx, false, None, Some((500, 500)), quad);
        assert!(!app.hovered_quad, "the pointer left the quad");
        assert_ne!(
            shared.render_revision.get(),
            entered,
            "leaving asks for a new overlay"
        );

        app.settle_frame(&ctx, false, None, None, quad);
        assert!(!app.hovered_quad, "off the image is not hovering either");
    }

    #[test]
    fn a_placed_object_keeps_its_quad_selected() {
        let shared = Rc::new(VideoEditingShared::default());
        let mut app = VideoEditingApp::new(document(), shared.clone());
        app.display_size = (1920, 1080);
        let quad = Some([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]);

        app.handle_pick((50, 50), quad);
        assert!(app.selected_quad);
        assert!(app.show_gizmos);

        app.selected_asset = Some(CatalogAsset::CocaColaCan);
        app.handle_pick((500, 500), quad);
        assert!(app.selected_quad, "the object's frame stays selected");
        assert!(app.show_gizmos, "and its basis stays visible");
        assert!(
            shared.pending_pick.get().is_some(),
            "the click asks the id pass about the object"
        );
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
            .update_video_frame_rgba(vec![1, 2, 3, 4], 1, 1, 5, 5.0 / 24.0, 0.0)
            .unwrap();
        shared.set_video_media_observation(4, false);

        let facts = app.displayed_facts();
        assert_eq!(facts.presented_frame_index, None);
        assert_eq!(facts.frame_index, Some(0));
        assert_eq!(facts.media_time_seconds, Some(0.0));
        assert_eq!(
            facts.frame_duration_seconds,
            Some(272.0 / TIMESCALE),
            "the displayed frame's own declared duration, not the newer frame's"
        );
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
            duration_seconds: 272.0 / TIMESCALE,
            scene: crate::scene::SceneState::default(),
            selected_asset: None,
            selected_quad: false,
            move_direction: crate::interaction::MoveDirection::Reference1,
            playing: false,
            show_quad: false,
            show_gizmos: false,
            hovered_quad: false,
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
                transfers: crate::video_editing_renderer::TransferCounts::default(),
            },
        }
    }
}
