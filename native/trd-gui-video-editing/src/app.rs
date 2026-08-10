use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant};

use crate::error::NativeVideoEditingError;
use crate::media::{DecodedFrame, NativeVideo, NativeVideoSource};
use trd_gui::video_editing::{
    CatalogAsset, VideoEditingApp, VideoEditingCommand, VideoEditingShared,
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
    document: trd_core::VideoEditingDocument,
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
}

impl NativeVideoEditingApp {
    pub fn new(
        document: trd_core::VideoEditingDocument,
        video_source: Option<NativeVideoSource>,
        preview_width: u32,
    ) -> Result<Self, NativeVideoEditingError> {
        let mut video = video_source
            .clone()
            .map(|source| NativeVideo::open(source, &document.video, preview_width))
            .transpose()?;
        let initial_frame = video
            .as_ref()
            .map(|video| video.decode_one(0))
            .transpose()?;
        let render_size = video
            .as_ref()
            .map(|video| (video.width, video.height))
            .unwrap_or_else(|| {
                let width = preview_width.min(document.video.width).max(1);
                let height = ((u64::from(width) * u64::from(document.video.height))
                    .div_ceil(u64::from(document.video.width))) as u32;
                (width, height)
            });
        if let Some(video) = &mut video {
            video.stop();
        }

        let shared = Rc::new(VideoEditingShared::default());
        let renderer = pollster::block_on(VideoPlacementRenderer::new_empty(
            render_size.0,
            render_size.1,
        ))
        .map_err(NativeVideoEditingError::Renderer)?;
        shared.set_renderer(renderer);
        let editor = VideoEditingApp::new(document.clone(), shared.clone());
        let mut app = Self {
            document,
            shared,
            editor,
            video_source,
            video,
            preview_width,
            frame_index: 0,
            playback: None,
            pending_frame: None,
            assets_root: std::env::current_dir().map_err(|source| {
                NativeVideoEditingError::Read {
                    path: "current working directory".to_owned(),
                    source,
                }
            })?,
            env_bytes: None,
        };
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
        match self
            .shared
            .update_video_frame_rgba(frame.rgba, width, height, frame.index)
        {
            Ok(()) => self.shared.clear_error(),
            Err(error) => {
                self.shared.set_error(error);
                self.stop_playback();
            }
        }
    }

    fn update_playback(&mut self) {
        let Some(clock) = self.playback else {
            return;
        };
        let last_frame = self.document.video.frame_count.saturating_sub(1);
        let target = clock.target_frame(
            Instant::now(),
            self.document.video.fps_num,
            self.document.video.fps_den,
            last_frame,
        );

        loop {
            if self.pending_frame.is_none() {
                match self.video.as_mut().and_then(NativeVideo::try_frame) {
                    Some(Ok(frame)) => self.pending_frame = Some(frame),
                    Some(Err(error)) => {
                        self.shared.set_error(error);
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
            self.shared.set_error(format!(
                "decoder ended at frame {} before final frame {last_frame}",
                self.frame_index
            ));
            self.stop_playback();
        }
    }

    fn service_editor_requests(&mut self) {
        if let Some(command) = self.shared.take_command() {
            match command {
                VideoEditingCommand::OpenLocalVideo => self.open_local_video(),
                VideoEditingCommand::Play => self.play(),
                VideoEditingCommand::Pause => self.pause(),
            }
        }
        if let Some(url) = self.shared.take_video_url_request() {
            self.open_video_source(NativeVideoSource::Url(url));
        }
        if let Some(index) = self.shared.take_seek_frame() {
            self.seek(index);
        }
        if let Some(asset) = self.shared.take_asset_request() {
            if let Err(error) = self.load_catalog_asset(asset) {
                self.shared.set_error(error);
            }
        }
    }

    fn open_local_video(&mut self) {
        self.stop_playback();
        let mut dialog = rfd::FileDialog::new().add_filter("MP4 video", &["mp4"]);
        if let Some(NativeVideoSource::Local(path)) = &self.video_source {
            if let Some(parent) = path.parent() {
                dialog = dialog.set_directory(parent);
            }
        }
        if let Some(path) = dialog.pick_file() {
            self.open_video_source(NativeVideoSource::Local(path));
        }
    }

    fn open_video_source(&mut self, source: NativeVideoSource) {
        self.stop_playback();
        let video =
            match NativeVideo::open(source.clone(), &self.document.video, self.preview_width) {
                Ok(video) => video,
                Err(error) => {
                    self.shared.set_error(error.to_string());
                    return;
                }
            };
        let frame = match video.decode_one(0) {
            Ok(frame) => frame,
            Err(error) => {
                self.shared.set_error(error.to_string());
                return;
            }
        };
        self.video_source = Some(source);
        self.video = Some(video);
        self.frame_index = 0;
        self.pending_frame = None;
        self.submit_frame(frame);
        self.sync_video_status();
    }

    fn play(&mut self) {
        if self.video.is_none() {
            self.shared
                .set_error("native playback requires a source passed with --video");
            return;
        }
        let last_frame = self.document.video.frame_count.saturating_sub(1);
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
                self.shared.clear_error();
                self.sync_video_status();
            }
            Err(error) => self.shared.set_error(error.to_string()),
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
            self.frame_index = index.min(self.document.video.frame_count.saturating_sub(1));
            self.shared
                .set_error("native seeking requires a source passed with --video");
            return;
        };
        video.stop();
        self.playback = None;
        self.pending_frame = None;
        match video.decode_one(index) {
            Ok(frame) => self.submit_frame(frame),
            Err(error) => self.shared.set_error(error.to_string()),
        }
        self.sync_video_status();
    }

    fn load_catalog_asset(&mut self, asset: CatalogAsset) -> Result<(), String> {
        let (model_path, texture_path) = catalog_paths(asset);
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
            .unwrap_or((self.document.video.width, self.document.video.height));
        let renderer = pollster::block_on(VideoPlacementRenderer::new(
            asset,
            &model_bytes,
            &texture_bytes,
            self.env_bytes.as_deref().expect("loaded above"),
            width,
            height,
        ))?;
        self.shared.set_catalog_renderer(asset, renderer);
        Ok(())
    }

    fn sync_video_status(&self) {
        self.shared
            .set_video_status(self.video.is_some(), self.playback.is_some());
    }

    fn frame_duration(&self) -> Duration {
        Duration::from_secs_f64(
            f64::from(self.document.video.fps_den) / f64::from(self.document.video.fps_num),
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
}
