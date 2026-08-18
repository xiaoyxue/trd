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
const COMMAND_PICK_DOCUMENT: u8 = 4;
const COMMAND_LOAD_SELECTION: u8 = 5;

/// A source the dialog has **selected but not loaded**.
///
/// Picking a file and loading it are separate steps because the annotation
/// document is optional *and* independent: the user chooses a video, maybe a
/// document, sees both, and then commits with one Load. A picker that loaded
/// immediately would make "video + document" impossible to express (#264).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingSource {
    pub kind: VideoSourceKind,
    /// A file name (local) or the URL itself. Display text *and* — for a URL —
    /// what the shell fetches.
    pub name: String,
}

/// The annotation-document formats the Open dialog accepts.
///
/// Extension matching is a **hint for the file picker only**: the real loader
/// must sniff magic bytes, because a URL need not carry a useful suffix and a
/// mislabelled file should still be read correctly (#264).
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

/// Whether a string is an `http`/`https` URL, the only schemes either source/// accepts — a browser cannot fetch anything else cross-origin, and a `file:`
/// URL would silently mean the wrong thing on each platform.
pub fn is_http_url(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}

/// Why the placement overlay is or is not drawing at the current frame.
///
/// It draws nothing for four different reasons, three of which are the ordinary
/// state of a **sparse** document — most frames of a long recording carry no
/// row at all (#264). A checkbox that appears to do nothing is indistinguishable
/// from a broken one, so the reason is stated rather than left to be inferred.
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

/// Resolves the overlay's state. `tracked` is `None` when the document has no
/// row for this frame at all. `show_overlay` is the combined toggle state: with
/// both the quads and the gizmos switched off there is nothing to draw, and the
/// reason is the toggle rather than the document.
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

/// Whether the dialog's Load button has anything to commit.
///
/// A newly selected video is the obvious case. A video that is **already
/// playing** is the other one: the annotation document is optional and
/// independent, so it has to be attachable to the video already on screen
/// (#264). Requiring a fresh video selection made that impossible — a video
/// opened from `?video=`, which never goes through the picker, left Load
/// permanently disabled and a picked `.arrow` with no way to apply it.
pub fn load_is_available(video_selected: bool, video_loaded: bool) -> bool {
    video_selected || video_loaded
}

/// What a loaded annotation document says about itself, read next to the video
/// that is actually playing.
///
/// Pure, so the readout is pinned by tests instead of by looking at the UI, and
/// so both shells show the same thing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentSummary {
    /// The video the document was authored against.
    pub describes: String,
    /// How many frames carry placement data, and where they are.
    pub annotated: String,
    /// Set when the document does not describe the video on screen. Rows are
    /// keyed by frame number, so a document from another clip does not fail to
    /// load — it silently lines its quads up with the wrong frames, which is
    /// worth saying out loud (#264).
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

    // Resolution is the objective signal. The *names* cannot be compared: a
    // video opened from a URL is labelled with the whole URL, so a match there
    // would be luck rather than evidence.
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

/// Validates a typed annotation-document URL, naming the format its suffix
/// suggests.
///
/// Pure so the dialog's rules are testable without a UI, and so both shells
/// agree about what is acceptable before anything is fetched.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoEditingCommand {
    OpenLocalVideo,
    /// Pick a local annotation document (`.arrow` / `.parquet`). Optional by
    /// design: without one the video simply plays (#264).
    OpenLocalDocument,
    /// Load what the dialog has selected — the picked local video, plus the
    /// document if one was chosen. Picking never loads on its own.
    LoadSelection,
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
    show_gizmos: bool,
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
    /// Set when the host binds the render texture directly into egui: the render
    /// task then draws without reading the pixels back.
    skip_readback: Cell<bool>,
    /// The **UI toolkit's own** GPU context, when the shell built the renderer on
    /// it (`eframe`'s `wgpu_render_state`).
    ///
    /// Held rather than merely flagged for two reasons. It is *declared* by the
    /// shell because `wgpu 30`'s `Device` has no identity comparison — nothing
    /// can be detected — and keeping the context itself means every **rebuilt**
    /// renderer (a catalog asset swap rebuilds one) lands on the same device
    /// instead of quietly opening another, which would make the bound texture
    /// come from a device egui knows nothing about.
    shared_gpu: RefCell<Option<std::sync::Arc<trd_core::GpuContext>>>,
    /// The browser `<video>` element whose decoded frames are copied GPU→GPU.
    ///
    /// Handed over once by the JS bootstrap; its presence is what selects the
    /// zero-upload path. Web-only: native frames arrive from an ffmpeg pipe as
    /// CPU bytes and have nothing to keep on the GPU (#229).
    #[cfg(target_arch = "wasm32")]
    /// The decoded frame waiting to be drawn. Owned here: it is closed after
    /// the upload, or when a newer frame replaces it.
    video_frame: RefCell<Option<web_sys::VideoFrame>>,
    /// A timeline the shell probed from the container, waiting to be adopted.    ///
    /// The browser learns the real frame rate only after `moov` has been read,
    /// which happens *after* the editor starts — so this arrives late by
    /// construction and the app consumes it on its next frame (#264).
    pending_video_info: RefCell<Option<trd_core::VideoInfo>>,
    /// An annotation document the shell fetched, waiting to be adopted — or
    /// `Some(None)` to drop the current one.
    ///
    /// A slot rather than a direct call because only the shell can read a file
    /// or a URL, while only the app owns the document; the app takes it on its
    /// next frame (#264). Distinct from `pending_document`, which is the
    /// dialog's *selection* — this is the decoded result of loading it.
    incoming_document: RefCell<Option<Option<trd_core::VideoEditingDocument>>>,
    command: Cell<u8>,
    asset_request: Cell<u8>,

    /// What the dialog has **selected but not yet loaded**. The shells fill the
    /// local-file entries in (only they run a file picker); the dialog fills in
    /// URLs itself.
    pending_video: RefCell<Option<PendingSource>>,
    pending_document: RefCell<Option<PendingSource>>,
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
            skip_readback: Cell::new(false),
            shared_gpu: RefCell::new(None),
            #[cfg(target_arch = "wasm32")]
            video_frame: RefCell::new(None),
            pending_video_info: RefCell::new(None),
            incoming_document: RefCell::new(None),
            command: Cell::new(COMMAND_NONE),
            asset_request: Cell::new(0),

            pending_video: RefCell::new(None),
            pending_document: RefCell::new(None),
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
    /// Switches the render task off the readback path, for a host that binds the
    /// rendered texture directly into egui.
    pub fn set_skip_readback(&self, skip: bool) {
        self.skip_readback.set(skip);
    }

    /// Declares that the renderer was built on the **UI toolkit's own device**,
    /// and keeps that context so rebuilt renderers stay on it.
    ///
    /// Declared by the shell rather than detected: `wgpu 30`'s `Device` derives
    /// only `Debug, Clone` — no `PartialEq`, no `global_id()` — so there is
    /// nothing to compare, and the shell is the component that *chose* the
    /// device anyway. Binding a texture from a different device is undefined.
    pub fn set_shared_gpu(&self, gpu: std::sync::Arc<trd_core::GpuContext>) {
        self.shared_gpu.replace(Some(gpu));
    }

    /// The toolkit's GPU context, if this editor shares one. A renderer rebuilt
    /// for a catalog asset **must** use it, or its texture belongs to a device
    /// egui cannot sample.
    pub fn shared_gpu(&self) -> Option<std::sync::Arc<trd_core::GpuContext>> {
        self.shared_gpu.borrow().clone()
    }

    /// The current renderer's target view + size + identity, for a host binding
    /// it into its UI toolkit. `None` while a render is in flight (the renderer
    /// is moved out for the duration) or before one exists.
    pub fn target_binding(&self) -> Option<(wgpu::TextureView, (u32, u32), usize)> {
        let renderer = self.renderer.borrow();
        let renderer = renderer.as_ref()?;
        Some((
            renderer.target_view(),
            renderer.size(),
            renderer.renderer_generation_key(),
        ))
    }

    /// Hands over the decoded frame whose pixels are copied GPU→GPU, so the
    /// render task can present it without any crossing the wasm boundary.
    ///
    /// Borrows rather than takes: a render can run more than once for the same
    /// frame — any UI change repaints — and a taken frame would leave the second
    /// pass with an empty RGBA buffer. The clone is another handle to the same
    /// frame, not WebCodecs' `clone()`, so it costs no extra pool slot; the
    /// frame is released when a newer one replaces it.
    #[cfg(target_arch = "wasm32")]
    fn video_frame(&self) -> Option<web_sys::VideoFrame> {
        self.video_frame.borrow().clone()
    }

    /// Publishes a decoded frame that lives **only on the GPU**: the
    /// `VideoFrame` holds the pixels, so the editor is told which timeline row
    /// is on screen and nothing else.
    ///
    /// **Takes ownership of the frame.** A `VideoFrame` holds a slot in a small
    /// decoder-side pool, so it is closed once uploaded — and any frame still
    /// pending here is closed when this one replaces it, since the newer one is
    /// what will be drawn.
    ///
    /// A separate entry point from
    /// [`update_video_frame_rgba`](Self::update_video_frame_rgba) rather than a
    /// flag on it, because the two have genuinely different preconditions — this
    /// one *requires* an empty buffer, and that buffer reaching `epaint` would
    /// panic if the display path had not been taught to skip it.
    #[cfg(target_arch = "wasm32")]
    pub fn present_video_frame(
        &self,
        frame: web_sys::VideoFrame,
        frame_index: u32,
        media_time_seconds: f64,
    ) -> Result<(), String> {
        let width = frame.display_width();
        let height = frame.display_height();
        if width == 0 || height == 0 {
            frame.close();
            return Err(format!("video frame size {width}x{height} is degenerate"));
        }
        if let Some(dropped) = self.video_frame.replace(Some(frame)) {
            dropped.close();
        }
        self.frame.replace(Some(IncomingVideoFrame {
            rgba: Vec::new(),
            width,
            height,
            frame_index,
            media_time_seconds,
            source_generation: self.source_generation.get(),
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
            COMMAND_PICK_DOCUMENT => Some(VideoEditingCommand::OpenLocalDocument),
            COMMAND_LOAD_SELECTION => Some(VideoEditingCommand::LoadSelection),
            COMMAND_PLAY => Some(VideoEditingCommand::Play),
            COMMAND_PAUSE => Some(VideoEditingCommand::Pause),
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

    /// Decodes `bytes` as an annotation document and hands it to the editor.
    ///
    /// Decoding here rather than in each shell keeps one implementation of the
    /// contract — and one error message — for native and web alike. A failure
    /// leaves the current document untouched: a bad file must not empty the
    /// editor (#264).
    pub fn load_document_bytes(&self, bytes: &[u8]) -> Result<(), String> {
        let document =
            trd_core::decode_video_editing_document(bytes).map_err(|error| error.to_string())?;
        self.incoming_document.replace(Some(Some(document)));
        self.request_repaint();
        Ok(())
    }

    /// Drops the current annotation document: the video keeps playing, as plain
    /// video.
    pub fn clear_document(&self) {
        self.incoming_document.replace(Some(None));
        self.request_repaint();
    }

    fn take_incoming_document(&self) -> Option<Option<trd_core::VideoEditingDocument>> {
        self.incoming_document.borrow_mut().take()
    }

    /// Records what a shell's file picker returned, **without loading it**. The
    /// dialog shows it and enables Load; nothing happens until Load is pressed.
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
    /// What the editor knows about the video **being played** — its size, rate,
    /// frame count and identity.
    ///
    /// Held separately from the document because the document is optional: with
    /// one it comes from the document, without one the shell synthesizes it from
    /// the container (ffprobe natively, `mp4_probe` in the browser). The editor
    /// only ever reads the timeline from here, so "no document" is not a special
    /// case anywhere else (#264).
    video: trd_core::VideoInfo,
    /// The annotation rows, when a document was supplied. `None` means the
    /// editor is a plain player: the placement UI is inert and every frame is
    /// just video.
    document: Option<trd_core::VideoEditingDocument>,
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
    /// Whether the local grid + basis axes are drawn. Independent of
    /// [`show_placement_quads`](Self::show_placement_quads): the quad says where
    /// an object may be placed, the gizmos describe the basis it is placed in,
    /// and either is useful without the other.
    show_gizmos: bool,
    /// Whether the placement quads are drawn at all — including **during
    /// playback**, which is the point: an annotated frame shows its quad as it
    /// plays past, and this is how you turn that off (#264).
    show_placement_quads: bool,
    was_playing: bool,
    selected_asset: Option<CatalogAsset>,
    image_sizing: crate::ui::ImageSizing,
    fitted_render_size: (u32, u32),
    show_video_source_dialog: bool,
    video_url: String,
    /// The URLs being typed, and the last thing the dialog said about each —
    /// `Ok` describes what is selected, `Err` why it was rejected. Cleared to
    /// `None` once the row falls back to reporting the pending selection.
    video_status: Option<Result<String, String>>,
    document_url: String,
    document_status: Option<Result<String, String>>,
    pending_seek_target: Option<u32>,
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

    /// The editor as a **plain player**: a timeline the shell probed from the
    /// container, and no annotation document.
    ///
    /// This is the video-first entry point (#264). Everything downstream reads
    /// the timeline from `video` and the rows through
    /// [`frame_row`](Self::frame_row), so "no document" needs no special case
    /// beyond an inert placement UI.
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
            show_gizmos: true,
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
            pending_seek_target: None,
            last_pick_result: None,
            native_texture: None,
            native_texture_key: None,
        }
    }

    /// The annotation row for `frame_index`, if the document has one.
    ///
    /// An `Option`, not an index: with no document — and, with sparse rows, on
    /// any unannotated frame — the absence of a row is the **normal** state, not
    /// an error. Every caller therefore has to say what it draws for a plain
    /// video frame (#264).
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

    /// Adopts (or drops) an annotation document **while the video keeps
    /// playing**.
    ///
    /// The video position is deliberately kept: attaching a document is an
    /// annotation change, not a source change. Everything derived from the old
    /// document is not — the selection, the placed object and any pick in flight
    /// all refer to a quad that may no longer exist, so they are cleared (#264).
    pub fn set_document(&mut self, document: Option<trd_core::VideoEditingDocument>) {
        self.document = document;
        self.selected_quad = false;
        self.selected_asset = None;
        self.last_pick_result = None;
        self.controller.state.selected = None;
        self.controller.state.objects[0] = crate::scene::ObjectTransform::default();
        self.shared.request_overlay();
    }

    /// Replaces the timeline once the shell has probed the real container.    ///
    /// The browser learns the frame rate only after `moov` has been read, which
    /// happens *after* the editor starts, so the timeline arrives late by
    /// construction. Clamps the playhead, since a shorter video may not contain
    /// the frame currently displayed.
    pub fn set_video_info(&mut self, video: trd_core::VideoInfo) {
        let last = video.frame_count.saturating_sub(1);
        self.video = video;
        self.current_frame_index = self.current_frame_index.min(last);
        self.displayed_frame_index = self.displayed_frame_index.min(last);
        self.pending_seek_target = None;
    }

    /// Binds trd's render texture straight into egui when both share a device.
    ///
    /// This is what removes the readback: instead of copying the rendered pixels
    /// GPU→CPU and re-uploading them through egui, the texture trd just drew
    /// into is registered once and sampled in place. It lives on the app rather
    /// than in a shell so **native and web share one implementation**.
    ///
    /// Registration is keyed on `(renderer identity, size)`, so a resize or an
    /// asset swap — both of which recreate the target — re-registers instead of
    /// sampling a freed view.
    ///
    /// A no-op unless the shell built the renderer on the toolkit's own device;
    /// two devices cannot share a texture, so those shells keep the readback.
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

    /// The video row. Choosing a file or accepting a URL only **selects** it —
    /// loading waits for the Load button, so a document can be chosen in the
    /// same visit (#264).
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

    /// The **optional** annotation-document row: a single `.arrow` or `.parquet`,
    /// local or over HTTP, plus Clear.
    ///
    /// Mock for now — it validates and reports what *would* be loaded. Landing
    /// the shape first keeps the loading slices free of layout churn (#264).
    fn document_source_row(&mut self, ui: &mut egui::Ui) {
        ui.heading("Annotation document (optional)");
        ui.label("Arrow IPC or Parquet rows naming the frames that carry placement data.");
        ui.weak("Without one the video simply plays; with one, those frames become editable.");

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
        ui.weak("Load applies the whole selection: with no document the video plays unannotated.");

        ui.label("Document URL");
        let response = ui.add(
            egui::TextEdit::singleline(&mut self.document_url)
                .hint_text("https://example.com/shot.parquet")
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
                None => "No document — the video plays as-is".to_owned(),
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
        if self.pending_seek_target == Some(frame.frame_index) {
            self.pending_seek_target = None;
        }
        // On the shared-device path the pixels were never read back — the
        // rendered texture is bound directly — so there is nothing to upload.
        // Missing this is a panic, not a silent bug: `ColorImage` asserts the
        // buffer matches the size (`epaint/src/image.rs`).
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
        let quad_frame = self.quad_frame_at(video.frame_index);
        // The overlay follows the **toggles**, not the play state: an annotated
        // frame shows its quad as it plays past, which is how a sparse document
        // announces itself during ordinary playback (#264).
        let tracked = background_frame.as_ref().is_some_and(|frame| frame.tracked);
        let show_quad = self.show_placement_quads && tracked;
        let show_gizmos = self.show_gizmos && tracked;
        let quad_overlay = crate::video_editing_renderer::QuadOverlay {
            model: quad_frame.map(trd_placement::quad_outline_model),
            axes: quad_frame.map(trd_placement::quad_axes_model),
            show_quads: show_quad,
            show_gizmos,
            selected: self.selected_quad,
        };
        let show_object = self.selected_asset.is_some()
            && self.selected_quad
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
        let source_size = (self.video.width, self.video.height);
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
        let width = self.video.width;
        let height = self.video.height;
        let selected_asset = self.selected_asset;
        let selected_quad = self.selected_quad;
        let move_direction = self.controller.move_direction;
        let rendered_model = model;
        let background_frame_index = video.frame_index;
        let background_media_time = video.media_time_seconds;
        let render_started = Instant::now();
        // A frame published by `present_video_frame` carries no pixels: the
        // decoded `VideoFrame` still holds them on the GPU, so the source is the
        // frame itself and nothing crosses the wasm boundary.
        #[cfg(target_arch = "wasm32")]
        let video_frame = shared.video_frame().filter(|_| video.rgba.is_empty());
        let render = async move {
            #[cfg(target_arch = "wasm32")]
            let source = match video_frame.as_ref() {
                Some(frame) => crate::video_editing_renderer::FrameSource::VideoFrame(frame),
                None => crate::video_editing_renderer::FrameSource::Rgba(&video.rgba),
            };
            #[cfg(not(target_arch = "wasm32"))]
            let source = crate::video_editing_renderer::FrameSource::Rgba(&video.rgba);
            let result = if shared.skip_readback.get() {
                // Shared-device path: the rendered texture is bound straight into
                // egui, so there is nothing to read back. The empty `Vec` is the
                // frame's payload, and `set_display_frame` skips the upload for
                // exactly that reason.
                renderer
                    .draw(
                        source,
                        video.width,
                        video.height,
                        (width, height),
                        background_frame.as_ref(),
                        quad_overlay,
                        placement_frame.as_ref(),
                        model,
                        &state,
                    )
                    .map(|()| Vec::new())
            } else {
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
                            show_gizmos,
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
        let timeline_frame = displayed_frame_index.and_then(|index| self.frame_row(index));
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
        let show_gizmos = displayed.is_some_and(|d| d.show_gizmos);
        // The gizmos no longer ride on the quad: each toggle contributes its own
        // drawables, so the count follows them independently.
        let background_drawables = 1 + u32::from(show_quad) + if show_gizmos { 2 } else { 0 };
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

    /// Why the overlay is drawing nothing, pinned without a UI. Three of the
    /// four silences are the ordinary state of a sparse document, which is
    /// exactly why the toggle looked broken.
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

    /// The document readout, pinned without a UI. The mismatch line is the point:
    /// a document from another clip loads perfectly well and then annotates the
    /// wrong frames, so nothing else would tell the user.
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

    /// Load's precondition, pinned without a UI. The second case is the one that
    /// was missing: a document has to be attachable to a video that is already
    /// playing, including one opened from `?video=` rather than the picker.
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

    /// The document row's rules, pinned without a UI: what the dialog accepts is
    /// what both shells will have to accept when the loading path is real.
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

        // An empty box is not an error state: no document is the default.
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

    /// Selecting is not loading, and the two sources are independent: clearing
    /// the document must leave the video's selection alone (#264).
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
            show_gizmos: false,
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
