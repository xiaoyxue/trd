use std::io::Read;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant};

use crate::error::NativeVideoEditingError;
use crate::media::{preview_size, DecodedFrame, NativeVideo, NativeVideoSource};
use trd_gui::video_editing::{
    CatalogAsset, ErrorScope, VideoEditingApp, VideoEditingCommand, VideoEditingShared,
    VideoSourceKind,
};
use trd_gui::video_editing_renderer::VideoPlacementRenderer;

#[derive(Clone, Copy)]
struct PlaybackClock {
    start_frame: u32,
    started: Instant,
}

impl PlaybackClock {
    fn new(start_frame: u32) -> Self {
        Self {
            start_frame,
            started: Instant::now(),
        }
    }

    fn target_frame(self, now: Instant, fps_num: u32, fps_den: u32, last_frame: u32) -> u32 {
        let elapsed = now.saturating_duration_since(self.started).as_secs_f64();
        let advanced = (elapsed * f64::from(fps_num) / f64::from(fps_den)).floor() as u32;
        self.start_frame.saturating_add(advanced).min(last_frame)
    }
}

pub struct NativeVideoEditingApp {
    /// The timeline in force: the document's when there is one, otherwise what
    /// ffprobe read from the container (#264).
    video_info: trd_core::VideoInfo,
    document: Option<trd_core::VideoEditingDocument>,
    shared: Rc<VideoEditingShared>,
    editor: VideoEditingApp,
    video_source: Option<NativeVideoSource>,
    video: Option<NativeVideo>,
    preview_width: u32,
    frame_index: u32,
    playback: Option<PlaybackClock>,
    pending_frame: Option<DecodedFrame>,
    assets_root: PathBuf,
    env_bytes: Option<Vec<u8>>,
    /// What the dialog picked but has not loaded: the paths behind the shared
    /// [`PendingSource`](trd_gui::video_editing::PendingSource) names.
    picked_video: Option<PathBuf>,
    picked_document: Option<PathBuf>,
}

/// Downloads an annotation document over HTTP(S).
///
/// A blocking one-shot rather than a client the app holds: a document is fetched
/// at most once per Open, and the UI thread is already blocked by the file
/// dialog next to it. The size cap keeps a mistyped URL from streaming a video
/// into memory — an annotation document is kilobytes to a few megabytes.
fn fetch_document(url: &str) -> Result<Vec<u8>, String> {
    const MAX_BYTES: u64 = 256 * 1024 * 1024;
    let mut response = ureq::get(url)
        .call()
        .map_err(|error| format!("{url}: {error}"))?;
    let mut bytes = Vec::new();
    response
        .body_mut()
        .as_reader()
        .take(MAX_BYTES)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("{url}: {error}"))?;
    Ok(bytes)
}

/// The timeline before anything is open: one frame of nothing, replaced as soon
/// as a video is loaded through the dialog.
fn empty_video_info() -> trd_core::VideoInfo {
    trd_core::VideoInfo {
        source_name: String::new(),
        mime: String::new(),
        codec: String::new(),
        sha256: String::new(),
        byte_length: 0,
        width: 16,
        height: 9,
        fps_num: 25,
        fps_den: 1,
        frame_count: 1,
        duration_us: 0,
        unpresented_tail: None,
    }
}

impl NativeVideoEditingApp {
    pub fn new(
        input: Option<trd_gui::video_editing::VideoEditingInput>,
        video_source: Option<NativeVideoSource>,
        preview_width: u32,
        gpu: Option<std::sync::Arc<trd_core::GpuContext>>,
    ) -> Result<Self, NativeVideoEditingError> {
        let (document, arrow_scene) = match input {
            Some(trd_gui::video_editing::VideoEditingInput::Annotation(document)) => {
                (Some(document), None)
            }
            Some(trd_gui::video_editing::VideoEditingInput::Scene(scene)) => {
                let mut scene = scene;
                resolve_arrow_scene(&mut scene).map_err(NativeVideoEditingError::Input)?;
                (None, Some(Rc::new(scene)))
            }
            None => (None, None),
        };
        // With a document the video is validated against it; without one the
        // container *is* the timeline, so the probe supplies it (#264).
        let (mut video, video_info) = match (video_source.clone(), document.as_ref()) {
            (Some(source), Some(document)) => {
                let (video, unpresented_tail) =
                    NativeVideo::open(source, &document.video, preview_width)?;
                let mut info = document.video.clone();
                info.unpresented_tail = unpresented_tail.or(info.unpresented_tail);
                (Some(video), info)
            }
            (Some(source), None) => {
                let (video, info) = NativeVideo::probe(source, preview_width)?;
                (Some(video), info)
            }
            (None, Some(document)) => (None, document.video.clone()),
            // Neither yet: an empty timeline the Open dialog will replace.
            (None, None) => (None, empty_video_info()),
        };
        let initial_frame = video
            .as_ref()
            .map(|video| video.decode_one(0))
            .transpose()?;
        let render_size = video
            .as_ref()
            .map(|video| (video.width, video.height))
            .unwrap_or_else(|| preview_size(&video_info, preview_width));
        if let Some(video) = &mut video {
            video.stop();
        }
        let assets_root =
            std::env::current_dir().map_err(|source| NativeVideoEditingError::Read {
                path: "current working directory".to_owned(),
                source,
            })?;
        let replay_env = arrow_scene
            .as_ref()
            .map(|_| read_asset(&assets_root, Path::new("assets/envmap/uffizi-large.hdr")))
            .transpose()
            .map_err(NativeVideoEditingError::Renderer)?;

        let shared = Rc::new(VideoEditingShared::default());
        // With eframe's device the rendered texture is bound straight into egui;
        // without one (no wgpu render state) the portable readback path stands.
        let renderer = match (gpu.clone(), arrow_scene.as_ref()) {
            (Some(gpu), Some(scene)) => VideoPlacementRenderer::new_scene_with_gpu(
                gpu,
                &scene
                    .mesh_assets()
                    .map_err(NativeVideoEditingError::Input)?,
                replay_env.as_deref().expect("scene env loaded above"),
                render_size.0,
                render_size.1,
            ),
            (None, Some(scene)) => pollster::block_on(VideoPlacementRenderer::new_scene(
                &scene
                    .mesh_assets()
                    .map_err(NativeVideoEditingError::Input)?,
                replay_env.as_deref().expect("scene env loaded above"),
                render_size.0,
                render_size.1,
            )),
            (Some(gpu), None) => {
                VideoPlacementRenderer::new_empty_with_gpu(gpu, render_size.0, render_size.1)
            }
            (None, None) => pollster::block_on(VideoPlacementRenderer::new_empty(
                render_size.0,
                render_size.1,
            )),
        }
        .map_err(NativeVideoEditingError::Renderer)?;
        shared.set_renderer(renderer);
        if let Some(gpu) = gpu {
            shared.set_shared_gpu(gpu);
        }
        let mut editor = match document.clone() {
            Some(document) => VideoEditingApp::new(document, shared.clone()),
            None => VideoEditingApp::player(video_info.clone(), shared.clone()),
        };
        if let Some(scene) = arrow_scene {
            editor.set_arrow_scene(Some(scene));
        }
        let mut app = Self {
            video_info,
            document,
            shared,
            editor,
            video_source,
            video,
            preview_width,
            frame_index: 0,
            playback: None,
            pending_frame: None,
            assets_root,
            env_bytes: None,
            picked_video: None,
            picked_document: None,
        };
        if app.video_source.is_some() {
            app.shared.set_video_status(false, false);
            app.sync_source_observation();
            app.shared.set_video_metadata_observation(
                app.video_info.width,
                app.video_info.height,
                app.video_info.duration_us as f64 / 1_000_000.0,
            );
        }
        if let Some(frame) = initial_frame {
            app.submit_frame(frame);
        }
        app.sync_video_status();
        Ok(app)
    }

    fn submit_frame(&mut self, frame: DecodedFrame) {
        let Some((width, height)) = self.video.as_ref().map(|video| (video.width, video.height))
        else {
            return;
        };
        self.frame_index = frame.index;
        match self.shared.update_video_frame_rgba(
            frame.rgba,
            width,
            height,
            frame.index,
            // Both read off the frame ffmpeg handed back, rather than computed
            // from the index that was asked for (#319).
            frame.media_time_seconds,
            frame.duration_seconds,
        ) {
            Ok(()) => {
                self.shared.clear_error(ErrorScope::Media);
                self.shared.set_video_media_observation(4, false);
            }
            Err(error) => {
                self.shared.set_error(ErrorScope::Media, error);
                self.stop_playback();
            }
        }
    }

    fn update_playback(&mut self) {
        let Some(clock) = self.playback else {
            return;
        };
        let last_frame = self.video_info.frame_count.saturating_sub(1);
        let target = clock.target_frame(
            Instant::now(),
            self.video_info.fps_num,
            self.video_info.fps_den,
            last_frame,
        );

        loop {
            if self.pending_frame.is_none() {
                match self.video.as_mut().and_then(NativeVideo::try_frame) {
                    Some(Ok(frame)) => self.pending_frame = Some(frame),
                    Some(Err(error)) => {
                        self.shared.set_error(ErrorScope::Media, error);
                        self.stop_playback();
                        return;
                    }
                    None => break,
                }
            }
            if self
                .pending_frame
                .as_ref()
                .is_none_or(|frame| frame.index > target)
            {
                break;
            }
            let frame = self.pending_frame.take().expect("checked above");
            self.submit_frame(frame);
        }

        if self.frame_index >= last_frame {
            self.stop_playback();
        } else if self.pending_frame.is_none()
            && self
                .video
                .as_ref()
                .is_some_and(|video| !video.is_streaming())
        {
            self.shared.set_error(
                ErrorScope::Media,
                format!(
                    "decoder ended at frame {} before final frame {last_frame}",
                    self.frame_index
                ),
            );
            self.stop_playback();
        }
    }

    fn service_editor_requests(&mut self) {
        if let Some(command) = self.shared.take_command() {
            match command {
                VideoEditingCommand::OpenLocalVideo => self.pick_local_video(),
                VideoEditingCommand::OpenLocalDocument => self.pick_local_document(),
                VideoEditingCommand::LoadSelection => self.load_selection(),
                VideoEditingCommand::Play => self.play(),
                VideoEditingCommand::Pause => self.pause(),
                VideoEditingCommand::ExportArrow => self.save_arrow_export(),
            }
        }

        if let Some(index) = self.shared.take_seek_frame() {
            self.seek(index);
        }
        if let Some(asset) = self.shared.take_asset_request() {
            // Logged, not merely surfaced in the UI. A catalog load reads and
            // decodes up to tens of megabytes and rebuilds the renderer, so it
            // is worth being able to see that it was asked for and that it
            // finished.
            log::info!("loading catalog asset {asset:?}");
            match self.load_catalog_asset(asset) {
                Ok(()) => {
                    log::info!("catalog asset {asset:?} loaded");
                    self.shared.clear_error(ErrorScope::Catalog);
                }
                Err(error) => {
                    log::error!("catalog asset {asset:?} failed to load: {error}");
                    self.shared.set_error(ErrorScope::Catalog, error);
                }
            }
        }
    }

    /// Picks a local annotation document. **Mock**: the choice is recorded and
    /// shown, but nothing is decoded yet — the loading path is its own slice, so
    /// this one stays reviewable as pure UI (#264).
    fn pick_local_document(&mut self) {
        let dialog = rfd::FileDialog::new().add_filter(
            "Arrow input",
            &trd_gui::video_editing::DocumentFormat::EXTENSIONS,
        );
        if let Some(path) = dialog.pick_file() {
            log::info!(
                "annotation document selected: {} (not loaded yet)",
                path.display()
            );
            self.shared
                .set_pending_document(Some(trd_gui::video_editing::PendingSource {
                    kind: VideoSourceKind::LocalFile,
                    name: path.display().to_string(),
                }));
            self.picked_document = Some(path);
        }
    }

    /// Picks a local video **without opening it**: the dialog stays up so an
    /// optional document can be chosen too, and Load commits both (#264).
    fn pick_local_video(&mut self) {
        let mut dialog = rfd::FileDialog::new().add_filter("MP4 video", &["mp4"]);
        if let Some(NativeVideoSource::Local(path)) = &self.video_source {
            if let Some(parent) = path.parent() {
                dialog = dialog.set_directory(parent);
            }
        }
        if let Some(path) = dialog.pick_file() {
            self.shared
                .set_pending_video(Some(trd_gui::video_editing::PendingSource {
                    kind: VideoSourceKind::LocalFile,
                    name: path.display().to_string(),
                }));
            self.picked_video = Some(path);
        }
    }

    fn save_arrow_export(&self) {
        let Some(export) = self.shared.take_arrow_export() else {
            self.shared.complete_arrow_export(Err(
                "the editor requested an export without queued Arrow bytes".to_owned(),
            ));
            return;
        };
        let path = rfd::FileDialog::new()
            .add_filter("Arrow scene", &["arrow"])
            .set_file_name(&export.filename)
            .save_file();
        let Some(path) = path else {
            self.shared.cancel_arrow_export();
            return;
        };
        match write_arrow_export(&path, &export.bytes) {
            Ok(()) => {
                log::info!(
                    "saved Arrow scene to {} ({} bytes)",
                    path.display(),
                    export.bytes.len()
                );
                self.shared.complete_arrow_export(Ok(format!(
                    "Saved {} bytes to {}",
                    export.bytes.len(),
                    path.display()
                )));
            }
            Err(error) => {
                log::error!("{error}");
                self.shared.complete_arrow_export(Err(error));
            }
        }
    }

    /// Loads whatever the dialog selected: the picked local video or the typed
    /// URL, plus the optional annotation document.
    ///
    /// Both are applied in one act, so "open this video *with* this document" is
    /// expressible — which is why picking never loads on its own (#264).
    fn load_selection(&mut self) {
        let Some(pending) = self.shared.pending_video() else {
            return;
        };
        let source = match pending.kind {
            VideoSourceKind::LocalFile => match self.picked_video.clone() {
                Some(path) => NativeVideoSource::Local(path),
                None => return,
            },
            VideoSourceKind::HttpUrl => NativeVideoSource::Url(pending.name),
        };
        self.open_video_source(source);
        self.load_selected_document();
    }

    /// Reads the selected annotation document — a local file or an HTTP(S) URL —
    /// and hands its bytes to the editor.
    ///
    /// A failure is reported and **leaves the current document in place**: a
    /// mistyped URL should not empty the editor.
    fn load_selected_document(&mut self) {
        let Some(pending) = self.shared.pending_document() else {
            // Load applies the *whole* selection: no document selected means the
            // video plays unannotated, even if one was loaded before. Keeping a
            // document authored against a different source would be worse than
            // dropping it (#264).
            self.shared.clear_document();
            self.document = None;
            self.picked_document = None;
            return;
        };
        let bytes = match pending.kind {
            VideoSourceKind::LocalFile => {
                let Some(path) = self.picked_document.clone() else {
                    return;
                };
                std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))
            }
            VideoSourceKind::HttpUrl => fetch_document(&pending.name),
        };
        let result = bytes.and_then(|bytes| self.load_input_bytes(&bytes));
        match result {
            Ok(kind) => {
                log::info!("{kind} loaded from {}", pending.name);
                self.shared.clear_error(ErrorScope::Document);
            }
            Err(error) => self.shared.set_error(ErrorScope::Document, error),
        }
    }

    fn load_input_bytes(&mut self, bytes: &[u8]) -> Result<&'static str, String> {
        match trd_gui::video_editing::decode_video_editing_input(bytes)? {
            trd_gui::video_editing::VideoEditingInput::Annotation(document) => {
                self.document = Some(document.clone());
                self.shared.queue_annotation_document(document);
                Ok("annotation document")
            }
            trd_gui::video_editing::VideoEditingInput::Scene(scene) => {
                let mut scene = scene;
                resolve_arrow_scene(&mut scene)?;
                let assets = scene.mesh_assets()?;
                if self.env_bytes.is_none() {
                    self.env_bytes = Some(read_asset(
                        &self.assets_root,
                        Path::new("assets/envmap/uffizi-large.hdr"),
                    )?);
                }
                let env = self.env_bytes.as_deref().expect("loaded above");
                let (width, height) = self
                    .video
                    .as_ref()
                    .map(|video| (video.width, video.height))
                    .unwrap_or_else(|| preview_size(&self.video_info, self.preview_width));
                let renderer = match self.shared.shared_gpu() {
                    Some(gpu) => {
                        VideoPlacementRenderer::new_scene_with_gpu(gpu, &assets, env, width, height)
                    }
                    None => pollster::block_on(VideoPlacementRenderer::new_scene(
                        &assets, env, width, height,
                    )),
                }?;
                self.document = None;
                self.shared.set_renderer(renderer);
                self.shared.queue_arrow_scene(Rc::new(scene));
                Ok("protocol scene")
            }
        }
    }

    fn open_video_source(&mut self, source: NativeVideoSource) {
        self.stop_playback();
        // With a document the source must match it; without one the container is
        // the timeline, so probe and adopt what it says (#264).
        let opened = match self.document.as_ref() {
            Some(document) => {
                NativeVideo::open(source.clone(), &document.video, self.preview_width).map(
                    |(video, unpresented_tail)| {
                        let mut info = document.video.clone();
                        info.unpresented_tail = unpresented_tail.or(info.unpresented_tail);
                        (video, info)
                    },
                )
            }
            None => NativeVideo::probe(source.clone(), self.preview_width),
        };
        let (video, info) = match opened {
            Ok(opened) => opened,
            Err(error) => {
                self.shared.set_error(ErrorScope::Media, error.to_string());
                return;
            }
        };
        let frame = match video.decode_one(0) {
            Ok(frame) => frame,
            Err(error) => {
                self.shared.set_error(ErrorScope::Media, error.to_string());
                return;
            }
        };
        self.shared.set_video_status(false, false);
        self.video_info = info;
        self.editor.set_video_info(self.video_info.clone());
        self.video_source = Some(source);
        self.video = Some(video);
        self.sync_source_observation();
        self.shared.set_video_metadata_observation(
            self.video_info.width,
            self.video_info.height,
            self.video_info.duration_us as f64 / 1_000_000.0,
        );
        self.frame_index = 0;
        self.pending_frame = None;
        self.submit_frame(frame);
        self.sync_video_status();
    }

    fn play(&mut self) {
        if self.video.is_none() {
            self.shared.set_error(
                ErrorScope::Media,
                "native playback requires a source passed with --video",
            );
            return;
        }
        let last_frame = self.video_info.frame_count.saturating_sub(1);
        let start_frame = replay_start(self.frame_index, last_frame);
        if start_frame != self.frame_index {
            self.seek(start_frame);
        }
        let result = self
            .video
            .as_mut()
            .expect("checked above")
            .play_from(start_frame);
        match result {
            Ok(()) => {
                self.playback = Some(PlaybackClock::new(start_frame));
                self.pending_frame = None;
                self.shared.clear_error(ErrorScope::Media);
                self.sync_video_status();
            }
            Err(error) => self.shared.set_error(ErrorScope::Media, error.to_string()),
        }
    }

    fn pause(&mut self) {
        self.stop_playback();
    }

    fn stop_playback(&mut self) {
        if let Some(video) = &mut self.video {
            video.stop();
        }
        self.playback = None;
        self.pending_frame = None;
        self.sync_video_status();
    }

    fn seek(&mut self, index: u32) {
        let Some(video) = &mut self.video else {
            self.frame_index = index.min(self.video_info.frame_count.saturating_sub(1));
            self.shared.set_error(
                ErrorScope::Media,
                "native seeking requires a source passed with --video",
            );
            return;
        };
        video.stop();
        self.playback = None;
        self.pending_frame = None;
        match video.decode_one(index) {
            Ok(frame) => self.submit_frame(frame),
            Err(error) => self.shared.set_error(ErrorScope::Media, error.to_string()),
        }
        self.sync_video_status();
    }

    fn load_catalog_asset(&mut self, asset: CatalogAsset) -> Result<(), String> {
        let (model_path, texture_path) = catalog_paths(asset);
        let source = trd_core::MeshReference::new(
            Some(model_path.to_string_lossy().replace('\\', "/")),
            None,
        )
        .expect("catalog path is non-empty");
        let model_bytes = read_asset(&self.assets_root, model_path)?;
        let texture_bytes = texture_path
            .map(|path| read_asset(&self.assets_root, path))
            .transpose()?
            .unwrap_or_default();
        if self.env_bytes.is_none() {
            self.env_bytes = Some(read_asset(
                &self.assets_root,
                Path::new("assets/envmap/uffizi-large.hdr"),
            )?);
        }
        let (width, height) = self
            .video
            .as_ref()
            .map(|video| (video.width, video.height))
            .unwrap_or_else(|| preview_size(&self.video_info, self.preview_width));
        // A catalog swap rebuilds the renderer, so it has to land on the *same*
        // device egui samples — otherwise the re-registered texture belongs to a
        // device the toolkit knows nothing about.
        let renderer = match self.shared.shared_gpu() {
            Some(gpu) => VideoPlacementRenderer::new_with_gpu(
                gpu,
                asset,
                source.clone(),
                &model_bytes,
                &texture_bytes,
                self.env_bytes.as_deref().expect("loaded above"),
                width,
                height,
            ),
            None => pollster::block_on(VideoPlacementRenderer::new(
                asset,
                source,
                &model_bytes,
                &texture_bytes,
                self.env_bytes.as_deref().expect("loaded above"),
                width,
                height,
            )),
        }?;
        self.shared.set_catalog_renderer(asset, renderer);
        Ok(())
    }

    fn sync_video_status(&self) {
        self.shared
            .set_video_status(self.video.is_some(), self.playback.is_some());
        let last_frame = self.video_info.frame_count.saturating_sub(1);
        self.shared.set_video_media_observation(
            if self.video.is_some() { 4 } else { 0 },
            self.video.is_some() && self.playback.is_none() && self.frame_index >= last_frame,
        );
    }

    fn sync_source_observation(&self) {
        let Some(source) = self.video_source.as_ref() else {
            return;
        };
        match source {
            NativeVideoSource::Local(path) => {
                let name = path.file_name().map_or_else(
                    || path.display().to_string(),
                    |name| name.to_string_lossy().into(),
                );
                self.shared.set_video_source_observation(
                    VideoSourceKind::LocalFile,
                    name,
                    Some(self.video_info.byte_length),
                );
            }
            NativeVideoSource::Url(url) => {
                self.shared
                    .set_video_source_observation(VideoSourceKind::HttpUrl, url, None);
            }
        }
    }

    fn frame_duration(&self) -> Duration {
        Duration::from_secs_f64(
            f64::from(self.video_info.fps_den) / f64::from(self.video_info.fps_num),
        )
    }
}

impl eframe::App for NativeVideoEditingApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        self.update_playback();
        self.service_editor_requests();
        eframe::App::ui(&mut self.editor, ui, frame);
        self.service_editor_requests();
        if self.playback.is_some() {
            ui.ctx().request_repaint_after(self.frame_duration());
        }
    }
}

fn replay_start(current_frame: u32, last_frame: u32) -> u32 {
    if current_frame >= last_frame {
        0
    } else {
        current_frame
    }
}

pub(crate) fn resolve_arrow_scene(
    scene: &mut trd_gui::video_editing::ArrowScene,
) -> Result<(), String> {
    for (index, reference) in scene.unresolved_mesh_references() {
        let bytes = load_mesh_reference(&reference)?;
        scene.resolve_gltf(index, &bytes)?;
    }
    Ok(())
}

fn load_mesh_reference(reference: &trd_core::MeshReference) -> Result<Vec<u8>, String> {
    if let Some(path) = reference.path.as_ref() {
        match std::fs::read(path) {
            Ok(bytes) => return Ok(bytes),
            Err(error) if reference.url.is_none() => {
                return Err(format!("failed to read {path}: {error}"));
            }
            Err(_) => {}
        }
    }
    let url = reference
        .url
        .as_deref()
        .ok_or_else(|| "glTF reference has neither a readable path nor a URL".to_owned())?;
    fetch_document(url)
}

fn catalog_paths(asset: CatalogAsset) -> (&'static Path, Option<&'static Path>) {
    match asset {
        CatalogAsset::CocaColaCan => (
            Path::new("assets/meshes/can/coke.obj"),
            Some(Path::new("assets/meshes/can/can_around.jpg")),
        ),
        CatalogAsset::BeerCan => (
            Path::new("assets/meshes/qd_beer/source/3d66.com_JDH5455878326.obj"),
            Some(Path::new(
                "assets/meshes/qd_beer/textures/3d66-export-JDH5455878326-001.jpg",
            )),
        ),
        CatalogAsset::Dragon => (
            Path::new("assets/meshes/glb/Meshy_AI_Dragon_0804104424_texture.glb"),
            None,
        ),
    }
}

fn read_asset(root: &Path, relative: &Path) -> Result<Vec<u8>, String> {
    let path = root.join(relative);
    std::fs::read(&path).map_err(|error| format!("failed to read {}: {error}", path.display()))
}

fn write_arrow_export(path: &Path, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(path, bytes)
        .map_err(|error| format!("failed to write Arrow scene {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn playback_clock_advances_at_declared_rate_and_clamps() {
        let started = Instant::now();
        let clock = PlaybackClock {
            start_frame: 10,
            started,
        };
        assert_eq!(clock.target_frame(started, 24, 1, 287), 10);
        assert_eq!(
            clock.target_frame(started + Duration::from_millis(500), 24, 1, 287),
            22
        );
        assert_eq!(
            clock.target_frame(started + Duration::from_secs(30), 24, 1, 287),
            287
        );
    }

    #[test]
    fn replay_restarts_only_at_end() {
        assert_eq!(replay_start(100, 287), 100);
        assert_eq!(replay_start(287, 287), 0);
    }

    #[test]
    fn arrow_export_writer_persists_the_exact_bytes() {
        let path = std::env::temp_dir().join(format!(
            "trd-arrow-export-{}-{}.arrow",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let bytes = b"arrow scene bytes";

        write_arrow_export(&path, bytes).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        std::fs::remove_file(path).unwrap();
    }
}
