use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Duration;

use crate::error::NativeVideoEditingError;
use crate::media::{DecodedFrame, NativeVideo};
use trd_gui::video_editing::{
    CatalogAsset, VideoEditingApp, VideoEditingCommand, VideoEditingShared,
};
use trd_gui::video_editing_renderer::VideoPlacementRenderer;

pub struct NativeVideoEditingApp {
    document: trd_core::VideoEditingDocument,
    shared: Rc<VideoEditingShared>,
    editor: VideoEditingApp,
    video_path: Option<PathBuf>,
    video: Option<NativeVideo>,
    frame_index: u32,
    playing: bool,
    assets_root: PathBuf,
    env_bytes: Option<Vec<u8>>,
}

impl NativeVideoEditingApp {
    pub fn new(
        document: trd_core::VideoEditingDocument,
        video_path: Option<PathBuf>,
        preview_width: u32,
    ) -> Result<Self, NativeVideoEditingError> {
        let mut video = video_path
            .clone()
            .map(|path| NativeVideo::new(path, &document.video, preview_width))
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
        let editor = VideoEditingApp::new(document.clone(), shared.clone())
            .map_err(|error| NativeVideoEditingError::Renderer(error.to_string()))?;
        let mut app = Self {
            document,
            shared,
            editor,
            video_path,
            video,
            frame_index: 0,
            playing: false,
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
                self.playing = false;
                self.shared.set_error(error);
            }
        }
    }

    fn consume_video_frames(&mut self) {
        while let Some(frame) = self.video.as_mut().and_then(NativeVideo::try_frame) {
            match frame {
                Ok(frame) => self.submit_frame(frame),
                Err(error) => {
                    self.playing = false;
                    self.shared.set_error(error);
                }
            }
        }
        if self.playing
            && self
                .video
                .as_ref()
                .is_some_and(|video| !video.is_streaming())
        {
            self.playing = false;
            self.sync_video_status();
        }
    }

    fn service_editor_requests(&mut self) {
        if let Some(command) = self.shared.take_command() {
            match command {
                VideoEditingCommand::OpenLocalVideo => {
                    let source = self
                        .video_path
                        .as_ref()
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| "none".to_owned());
                    self.shared.set_error(format!(
                        "native source is configured with --video (current: {source})"
                    ));
                }
                VideoEditingCommand::Play => self.play(),
                VideoEditingCommand::Pause => self.pause(),
            }
        }
        if let Some(url) = self.shared.take_video_url_request() {
            self.shared.set_error(format!(
                "HTTP(S) video sources are browser-only; use --video for native playback ({url})"
            ));
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

    fn play(&mut self) {
        let Some(video) = &mut self.video else {
            self.shared
                .set_error("native playback requires a source passed with --video");
            return;
        };
        match video.play_from(self.frame_index) {
            Ok(()) => {
                self.playing = true;
                self.shared.clear_error();
                self.sync_video_status();
            }
            Err(error) => self.shared.set_error(error.to_string()),
        }
    }

    fn pause(&mut self) {
        if let Some(video) = &mut self.video {
            video.stop();
        }
        self.playing = false;
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
        self.playing = false;
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
            .set_video_status(self.video.is_some(), self.playing);
    }

    fn frame_duration(&self) -> Duration {
        Duration::from_secs_f64(
            f64::from(self.document.video.fps_den) / f64::from(self.document.video.fps_num),
        )
    }
}

impl eframe::App for NativeVideoEditingApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        self.consume_video_frames();
        self.service_editor_requests();
        eframe::App::ui(&mut self.editor, ui, frame);
        self.service_editor_requests();
        if self.playing {
            ui.ctx().request_repaint_after(self.frame_duration());
        }
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
