//! Browser video-editing example state and UI (#163).

#[cfg(target_arch = "wasm32")]
use std::cell::{Cell, RefCell};
#[cfg(target_arch = "wasm32")]
use std::rc::Rc;

#[cfg(target_arch = "wasm32")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum CatalogAsset {
    CocaColaCan,
    BeerCan,
    Dragon,
}

#[cfg(target_arch = "wasm32")]
impl CatalogAsset {
    const ALL: [Self; 3] = [Self::CocaColaCan, Self::BeerCan, Self::Dragon];

    const fn label(self) -> &'static str {
        match self {
            Self::CocaColaCan => "Coca-Cola can",
            Self::BeerCan => "Beer can",
            Self::Dragon => "Dragon",
        }
    }

    const fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::CocaColaCan),
            2 => Some(Self::BeerCan),
            3 => Some(Self::Dragon),
            _ => None,
        }
    }

    const fn code(self) -> u8 {
        self as u8 + 1
    }
}

#[cfg(target_arch = "wasm32")]
const COMMAND_NONE: u8 = 0;
#[cfg(target_arch = "wasm32")]
const COMMAND_PICK_VIDEO: u8 = 1;
#[cfg(target_arch = "wasm32")]
const COMMAND_PLAY: u8 = 2;
#[cfg(target_arch = "wasm32")]
const COMMAND_PAUSE: u8 = 3;

#[cfg(target_arch = "wasm32")]
#[derive(Clone)]
struct IncomingVideoFrame {
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    frame_index: u32,
}

#[cfg(target_arch = "wasm32")]
pub struct VideoEditingShared {
    frame: RefCell<Option<IncomingVideoFrame>>,
    latest_video_frame: RefCell<Option<IncomingVideoFrame>>,
    rendered_frame: RefCell<Option<IncomingVideoFrame>>,
    context: RefCell<Option<egui::Context>>,
    command: Cell<u8>,
    asset_request: Cell<u8>,
    video_url_request: RefCell<Option<String>>,
    seek_frame: Cell<i32>,
    video_loaded: Cell<bool>,
    video_playing: Cell<bool>,
    needs_overlay: Cell<bool>,
    render_in_flight: Cell<bool>,
    pending_pick: Cell<Option<(u32, u32)>>,
    pick_in_flight: Cell<bool>,
    pick_result: RefCell<Option<Option<u32>>>,
    pub(crate) renderer: RefCell<Option<crate::video_editing_renderer::VideoPlacementRenderer>>,
    asset_defaults: RefCell<Option<(CatalogAsset, trd_core::RenderMode, trd_core::DisneyMaterial)>>,
    error: RefCell<Option<String>>,
}

#[cfg(target_arch = "wasm32")]
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
            needs_overlay: Cell::new(false),
            render_in_flight: Cell::new(false),
            pending_pick: Cell::new(None),
            pick_in_flight: Cell::new(false),
            pick_result: RefCell::new(None),
            renderer: RefCell::new(None),
            asset_defaults: RefCell::new(None),
            error: RefCell::new(None),
        }
    }
}

/// Browser bridge for the dedicated editor. It transfers browser-decoded pixels
/// and services commands emitted by Rust UI; it never computes scene matrices.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub struct VideoEditingHandle {
    shared: Rc<VideoEditingShared>,
    source_name: String,
    byte_length: u64,
    fps_num: u32,
    fps_den: u32,
    frame_count: u32,
    width: u32,
    height: u32,
}

#[cfg(target_arch = "wasm32")]
impl VideoEditingHandle {
    pub(crate) fn new(
        document: &trd_core::VideoEditingDocument,
        shared: Rc<VideoEditingShared>,
    ) -> Self {
        Self {
            shared,
            source_name: document.video.source_name.clone(),
            byte_length: document.video.byte_length,
            fps_num: document.video.fps_num,
            fps_den: document.video.fps_den,
            frame_count: document.video.frame_count,
            width: document.video.width,
            height: document.video.height,
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
impl VideoEditingHandle {
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = validateVideoFile)]
    pub fn validate_video_file(
        &self,
        filename: &str,
        byte_length: f64,
    ) -> Result<(), wasm_bindgen::JsValue> {
        if filename != self.source_name {
            return Err(wasm_bindgen::JsValue::from_str(&format!(
                "expected {}, got {filename}",
                self.source_name
            )));
        }
        if byte_length != self.byte_length as f64 {
            return Err(wasm_bindgen::JsValue::from_str(&format!(
                "expected {} bytes, got {byte_length:.0}",
                self.byte_length
            )));
        }
        Ok(())
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = validateVideoMetadata)]
    pub fn validate_video_metadata(
        &self,
        width: u32,
        height: u32,
        duration_seconds: f64,
    ) -> Result<(), wasm_bindgen::JsValue> {
        if (width, height) != (self.width, self.height) {
            return Err(wasm_bindgen::JsValue::from_str(&format!(
                "expected {}x{} video, got {width}x{height}",
                self.width, self.height
            )));
        }
        let expected_duration =
            f64::from(self.frame_count) * f64::from(self.fps_den) / f64::from(self.fps_num);
        let frame_duration = f64::from(self.fps_den) / f64::from(self.fps_num);
        if !duration_seconds.is_finite()
            || (duration_seconds - expected_duration).abs() > frame_duration
        {
            return Err(wasm_bindgen::JsValue::from_str(&format!(
                "expected {expected_duration:.3}s video, got {duration_seconds:.3}s"
            )));
        }
        Ok(())
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = frameIndexAtMediaTime)]
    pub fn frame_index_at_media_time(&self, media_time_seconds: f64) -> u32 {
        let frame = (media_time_seconds * f64::from(self.fps_num) / f64::from(self.fps_den))
            .round()
            .max(0.0) as u32;
        frame.min(self.frame_count.saturating_sub(1))
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = mediaTimeAtFrame)]
    pub fn media_time_at_frame(&self, frame_index: u32) -> f64 {
        let frame = frame_index.min(self.frame_count.saturating_sub(1));
        f64::from(frame) * f64::from(self.fps_den) / f64::from(self.fps_num)
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = updateVideoFrameRgba)]
    pub fn update_video_frame_rgba(
        &self,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        frame_index: u32,
    ) -> Result<(), wasm_bindgen::JsValue> {
        let expected = width as usize * height as usize * 4;
        if rgba.len() != expected {
            return Err(wasm_bindgen::JsValue::from_str(&format!(
                "video RGBA length {} != {width}x{height}x4",
                rgba.len()
            )));
        }
        if frame_index >= self.frame_count {
            return Err(wasm_bindgen::JsValue::from_str(
                "video frame index out of range",
            ));
        }
        self.shared.frame.replace(Some(IncomingVideoFrame {
            rgba,
            width,
            height,
            frame_index,
        }));
        if let Some(context) = self.shared.context.borrow().as_ref() {
            context.request_repaint();
        }
        Ok(())
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = setVideoStatus)]
    pub fn set_video_status(&self, loaded: bool, playing: bool) {
        self.shared.video_loaded.set(loaded);
        self.shared.video_playing.set(playing);
        if !loaded {
            self.shared.error.replace(None);
        }
        if let Some(context) = self.shared.context.borrow().as_ref() {
            context.request_repaint();
        }
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = setVideoError)]
    pub fn set_video_error(&self, message: String) {
        self.shared.error.replace(Some(message));
        if let Some(context) = self.shared.context.borrow().as_ref() {
            context.request_repaint();
        }
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = takeCommand)]
    pub fn take_command(&self) -> u8 {
        self.shared.command.replace(COMMAND_NONE)
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = takeAssetRequest)]
    pub fn take_asset_request(&self) -> u8 {
        self.shared.asset_request.replace(0)
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = takeVideoUrlRequest)]
    pub fn take_video_url_request(&self) -> Option<String> {
        self.shared.video_url_request.borrow_mut().take()
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = takeSeekFrame)]
    pub fn take_seek_frame(&self) -> i32 {
        self.shared.seek_frame.replace(-1)
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = loadCatalogAsset)]
    pub async fn load_catalog_asset(
        &self,
        asset_code: u8,
        model_bytes: Vec<u8>,
        texture_bytes: Vec<u8>,
        env_bytes: Vec<u8>,
    ) -> Result<(), wasm_bindgen::JsValue> {
        let asset = CatalogAsset::from_code(asset_code)
            .ok_or_else(|| wasm_bindgen::JsValue::from_str("unknown catalog asset"))?;
        let renderer = crate::video_editing_renderer::VideoPlacementRenderer::new(
            asset,
            &model_bytes,
            &texture_bytes,
            &env_bytes,
            self.width,
            self.height,
        )
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error))?;
        let (mode, material) = renderer.defaults();
        self.shared
            .asset_defaults
            .replace(Some((asset, mode, material)));
        self.shared.renderer.replace(Some(renderer));
        self.shared.needs_overlay.set(true);
        if let Some(context) = self.shared.context.borrow().as_ref() {
            context.request_repaint();
        }
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
pub struct VideoEditingApp {
    document: trd_core::VideoEditingDocument,
    display_image: egui::ColorImage,
    display_texture: Option<egui::TextureHandle>,
    current_frame_index: u32,
    displayed_frame_index: u32,
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
}

#[cfg(target_arch = "wasm32")]
impl VideoEditingApp {
    pub fn new(
        document: trd_core::VideoEditingDocument,
        shared: Rc<VideoEditingShared>,
    ) -> Result<Self, image::ImageError> {
        let image = image::load_from_memory(&document.poster_bytes)?.to_rgba8();
        let size = [image.width() as usize, image.height() as usize];
        shared.latest_video_frame.replace(Some(IncomingVideoFrame {
            rgba: image.as_raw().to_vec(),
            width: image.width(),
            height: image.height(),
            frame_index: 0,
        }));
        let scene = crate::scene::SceneState::default();
        let mut controller = crate::interaction::InteractionController::new(scene);
        controller.target = crate::interaction::InteractionTarget::Object;
        controller.move_direction = crate::interaction::MoveDirection::Reference1;
        controller.move_reference_axes = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
        controller.state.camera.distance = 1.0;
        Ok(Self {
            document,
            display_image: egui::ColorImage::from_rgba_unmultiplied(size, image.as_raw()),
            display_texture: None,
            current_frame_index: 0,
            displayed_frame_index: 0,
            display_size: (image.width(), image.height()),
            shared,
            controller,
            selected_quad: false,
            show_quad_gizmo: false,
            was_playing: false,
            selected_asset: None,
            image_sizing: crate::ui::ImageSizing::FitCanvas,
            fitted_render_size: (image.width(), image.height()),
            show_video_source_dialog: false,
            video_url: String::new(),
        })
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

    fn set_display_frame(&mut self, frame: &IncomingVideoFrame) {
        self.display_size = (frame.width, frame.height);
        self.displayed_frame_index = frame.frame_index;
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
        self.shared.needs_overlay.set(true);
        self.schedule_overlay();
    }

    fn consume_rendered_frame(&mut self) {
        let frame = self.shared.rendered_frame.borrow_mut().take();
        let Some(frame) = frame else {
            return;
        };
        self.set_display_frame(&frame);
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
                },
                CatalogAsset::CocaColaCan | CatalogAsset::BeerCan => trd_core::Lighting::default(),
            };
            self.controller.rebase_reset();
            self.shared.needs_overlay.set(true);
        }
    }

    fn consume_pick_result(&mut self) {
        let Some(hit) = self.shared.pick_result.borrow_mut().take() else {
            return;
        };
        if hit != self.controller.state.selected {
            self.controller.state.selected = hit;
            self.shared.needs_overlay.set(true);
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
        let render_size = renderer.size();
        self.shared.needs_overlay.set(false);
        let mut state = self.controller.state.clone();
        if self.shared.video_playing.get() {
            state.selected = None;
            state.show_aabb = false;
            state.show_axes = false;
            state.show_local_axes = false;
            state.show_world_grid = false;
            state.show_local_grid = false;
        }
        self.shared.render_in_flight.set(true);
        let shared = self.shared.clone();
        let width = self.document.video.width;
        let height = self.document.video.height;
        let show_quad_gizmo = self.show_quad_gizmo;
        let background_frame_index = video.frame_index;
        wasm_bindgen_futures::spawn_local(async move {
            let result = renderer
                .render(
                    &video.rgba,
                    width,
                    height,
                    &background_frame,
                    quad_model,
                    quad_axes,
                    show_quad_gizmo,
                    placement_frame.as_ref(),
                    model,
                    &state,
                )
                .await;
            shared.renderer.replace(Some(renderer));
            match result {
                Ok(rgba) => {
                    shared.rendered_frame.replace(Some(IncomingVideoFrame {
                        rgba,
                        width: render_size.0,
                        height: render_size.1,
                        frame_index: background_frame_index,
                    }));
                }
                Err(error) => {
                    shared.error.replace(Some(error));
                }
            }
            shared.render_in_flight.set(false);
            if let Some(context) = shared.context.borrow().as_ref() {
                context.request_repaint();
            }
        });
    }

    fn schedule_pick(&self) {
        if self.shared.render_in_flight.get() || self.shared.pick_in_flight.get() {
            return;
        }
        let Some(point) = self.shared.pending_pick.take() else {
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
            self.shared.pending_pick.set(Some(point));
            return;
        };
        let source_size = (self.document.video.width, self.document.video.height);
        let render_size = renderer.size();
        let target_point = (
            point.0 * render_size.0 / self.display_size.0.max(1),
            point.1 * render_size.1 / self.display_size.1.max(1),
        );
        self.shared.pick_in_flight.set(true);
        let shared = self.shared.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let hit = renderer
                .pick(&frame, source_size, model, target_point)
                .await;
            shared.renderer.replace(Some(renderer));
            shared.pick_result.replace(Some(hit));
            shared.pick_in_flight.set(false);
            if let Some(context) = shared.context.borrow().as_ref() {
                context.request_repaint();
            }
        });
    }

    fn quad_frame_at(&self, frame_index: u32) -> Option<trd_placement::QuadFrame> {
        let frame = self.document.frames.get(frame_index as usize)?;
        let k = frame.k?;
        let placement_quad = frame.placement_quad?;
        trd_placement::quad_frame(
            trd_placement::CameraIntrinsics { row_major: k },
            trd_placement::PlacementQuad {
                points_px: placement_quad,
            },
        )
        .ok()
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
}

#[cfg(target_arch = "wasm32")]
impl eframe::App for VideoEditingApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.shared.context.replace(Some(ui.ctx().clone()));
        self.consume_video_frame();
        self.consume_rendered_frame();
        self.consume_asset_defaults();
        self.consume_pick_result();
        self.ensure_texture(ui.ctx());
        let playing = self.shared.video_playing.get();
        if playing && !self.was_playing {
            self.show_quad_gizmo = false;
            self.shared.needs_overlay.set(true);
        }
        self.was_playing = playing;
        self.schedule_pick();
        self.schedule_overlay();

        self.video_source_dialog(ui.ctx());

        let overlay_frame_index = self.displayed_frame_index;
        let timeline_frame = &self.document.frames[overlay_frame_index as usize];
        let quad = timeline_frame.placement_quad;
        let quad_frame = self.quad_frame_at(overlay_frame_index);
        let selected_quad = self.selected_quad;
        let show_quad_gizmo = self.show_quad_gizmo;
        let selected_asset = self.selected_asset;
        let video_loaded = self.shared.video_loaded.get();
        let video_playing = self.shared.video_playing.get();
        let video = &self.document.video;
        let error = self.shared.error.borrow().clone();
        let mut requested_asset = None;
        let mut open_video_requested = false;
        let mut top_controls = |ui: &mut egui::Ui| {
            ui.heading("Video");
            if ui.button("Open video...").clicked() {
                open_video_requested = true;
            }
            ui.weak("Display: fit right pane (16:9)");
            ui.collapsing("Source", |ui| {
                ui.label(format!("Source: {}", video.source_name));
                ui.label(format!(
                    "{}x{} · {}/{} fps · {} frames",
                    video.width, video.height, video.fps_num, video.fps_den, video.frame_count
                ));
                ui.label(if video_loaded {
                    if video_playing {
                        "Playing video"
                    } else {
                        "Video paused"
                    }
                } else {
                    "No video loaded"
                });
                if let Some(error) = error.as_deref() {
                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                }
            });
            false
        };
        let mut extra_controls = |ui: &mut egui::Ui| {
            let mut changed = false;
            ui.collapsing("Placement quad (standalone)", |ui| {
                ui.label(format!("Frame {}", timeline_frame.video_frame_index));
                ui.label(if timeline_frame.tracked {
                    if video_playing {
                        "Placement quad hidden during playback"
                    } else if selected_quad {
                        if show_quad_gizmo {
                            "Placement quad selected; gizmo visible"
                        } else {
                            "Placement quad selected; click it to show gizmo"
                        }
                    } else {
                        "Click the green quad to select it"
                    }
                } else {
                    "Background-only row: quad and object hidden"
                });
                if let Some(local) = quad_frame {
                    ui.label(format!("Local axis length: {:.4}", local.axis_length));
                    ui.weak("RGB axes: e1 / e2 / e3");
                    ui.weak("Quad overlay follows the displayed tracking row.");
                    ui.weak("Object edit state persists; quad basis updates per frame.");
                    ui.weak("Local X/Y/Z rotate with the placed object.");
                    ui.weak("Initial can placement matches the Olympic upper-can preset.");
                }
            });
            ui.add_enabled_ui(selected_quad, |ui| {
                ui.collapsing("Object catalog", |ui| {
                    for asset in CatalogAsset::ALL {
                        if ui
                            .selectable_label(selected_asset == Some(asset), asset.label())
                            .clicked()
                        {
                            requested_asset = Some(asset);
                            changed = true;
                        }
                    }
                });
            });
            changed
        };
        let mut requested_frame = self.current_frame_index;
        let mut playback_command = None;
        let mut central_bottom = |ui: &mut egui::Ui| {
            ui.add_space(4.0);
            let (row_rect, _) = ui
                .allocate_exact_size(egui::vec2(ui.available_width(), 28.0), egui::Sense::hover());
            let button_rect =
                egui::Rect::from_center_size(row_rect.center(), egui::vec2(64.0, 28.0));
            ui.scope_builder(
                egui::UiBuilder::new()
                    .max_rect(button_rect)
                    .layout(egui::Layout::top_down(egui::Align::Center)),
                |ui| {
                    if video_playing {
                        if ui.button("Pause").clicked() {
                            playback_command = Some(COMMAND_PAUSE);
                        }
                    } else if ui
                        .add_enabled(video_loaded, egui::Button::new("Play"))
                        .clicked()
                    {
                        playback_command = Some(COMMAND_PLAY);
                    }
                },
            );
            ui.scope_builder(
                egui::UiBuilder::new()
                    .max_rect(row_rect)
                    .layout(egui::Layout::right_to_left(egui::Align::Center)),
                |ui| {
                    if video_loaded {
                        let current =
                            media_time_label(requested_frame, video.fps_num, video.fps_den);
                        let total =
                            media_time_label(video.frame_count, video.fps_num, video.fps_den);
                        ui.monospace(format!(
                            "{current} / {total}  ·  frame {}/{}",
                            requested_frame.saturating_add(1),
                            video.frame_count
                        ));
                    } else {
                        ui.monospace("00:00 / 00:00  ·  frame 0/0");
                    }
                },
            );
            ui.add_space(6.0);
            let last = video.frame_count.saturating_sub(1);
            video_progress_bar(ui, &mut requested_frame, last, video_loaded);
        };
        let mut pick_request = None;
        let mut fitted_render_size = self.fitted_render_size;
        let mut view = crate::ui::View {
            controller: &mut self.controller,
            texture: self
                .shared
                .video_loaded
                .get()
                .then_some(self.display_texture.as_ref())
                .flatten(),
            render_size: self.display_size,
            last_render_ms: None,
            pick_request: &mut pick_request,
        };
        let mut extensions = crate::ui::UiExtensions {
            top_controls: Some(&mut top_controls),
            extra_controls: Some(&mut extra_controls),
            image_overlay: None,
            camera_locked: true,
            image_sizing: self.image_sizing,
            move_reference_labels: Some(["e1", "e2", "e3"]),
            hide_empty_image: true,
            fitted_render_size: Some(&mut fitted_render_size),
            central_bottom: Some(&mut central_bottom),
            central_bottom_height: Some(80.0),
        };
        let changed = crate::ui::show_with_extensions(ui, &mut view, &mut extensions);
        if open_video_requested {
            self.show_video_source_dialog = true;
            ui.ctx().request_repaint();
        }
        if let Some(command) = playback_command {
            self.shared.command.set(command);
        }
        if requested_frame != self.current_frame_index {
            self.current_frame_index = requested_frame;
            self.shared.seek_frame.set(requested_frame as i32);
        }
        let fitted_render_size = (
            fitted_render_size.0.min(video.width).max(1),
            fitted_render_size.1.min(video.height).max(1),
        );
        if self.image_sizing == crate::ui::ImageSizing::FitCanvas
            && fitted_render_size != self.fitted_render_size
        {
            self.fitted_render_size = fitted_render_size;
            if self.selected_asset.is_some() {
                self.shared.needs_overlay.set(true);
                ui.ctx().request_repaint();
            }
        }

        if let Some(asset) = requested_asset {
            self.selected_asset = Some(asset);
            self.controller.state.objects[0] = crate::scene::ObjectTransform::default();
            self.controller.state.selected = Some(0);
            self.controller.target = crate::interaction::InteractionTarget::Object;
            self.shared.renderer.borrow_mut().take();
            self.shared.asset_request.set(asset.code());
            self.shared.needs_overlay.set(true);
        }

        if let Some((x, y)) = pick_request {
            let clicked_quad = quad.is_some_and(|points| {
                let source = [
                    x as f32 * video.width as f32 / self.display_size.0 as f32,
                    y as f32 * video.height as f32 / self.display_size.1 as f32,
                ];
                point_in_quad(source, points)
            });
            if self.shared.video_playing.get() {
                if self.selected_asset.is_some() && self.selected_quad {
                    self.shared.pending_pick.set(Some((x, y)));
                }
            } else if self.selected_asset.is_some() && self.selected_quad {
                if clicked_quad && !self.show_quad_gizmo {
                    self.show_quad_gizmo = true;
                } else {
                    self.shared.pending_pick.set(Some((x, y)));
                }
            } else {
                self.selected_quad = clicked_quad;
                self.show_quad_gizmo = clicked_quad;
            }
            if self.selected_quad {
                self.controller.target = crate::interaction::InteractionTarget::Object;
            }
            self.shared.needs_overlay.set(true);
        }

        if changed {
            self.shared.needs_overlay.set(true);
            ui.ctx().request_repaint();
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn video_progress_bar(ui: &mut egui::Ui, frame: &mut u32, last_frame: u32, enabled: bool) -> bool {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 18.0),
        if enabled {
            egui::Sense::click_and_drag()
        } else {
            egui::Sense::hover()
        },
    );
    let mut changed = false;
    if enabled && (response.clicked() || response.dragged()) {
        if let Some(pointer) = response.interact_pointer_pos() {
            let fraction = ((pointer.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
            let next = (fraction * last_frame as f32).round() as u32;
            changed = next != *frame;
            *frame = next;
        }
    }

    let visuals = ui.visuals();
    let track = egui::Rect::from_center_size(rect.center(), egui::vec2(rect.width(), 6.0));
    let fraction = if last_frame == 0 {
        0.0
    } else {
        *frame as f32 / last_frame as f32
    };
    let knob_x = egui::lerp(track.left()..=track.right(), fraction);
    let played = egui::Rect::from_min_max(track.min, egui::pos2(knob_x, track.bottom()));
    let background = if enabled {
        visuals.widgets.inactive.bg_fill
    } else {
        visuals.widgets.noninteractive.bg_fill
    };
    let accent = visuals.selection.bg_fill;
    ui.painter().rect_filled(track, 3.0, background);
    ui.painter().rect_filled(played, 3.0, accent);
    ui.painter().circle_filled(
        egui::pos2(knob_x, track.center().y),
        if response.hovered() { 6.0 } else { 5.0 },
        if enabled {
            accent
        } else {
            visuals.weak_text_color()
        },
    );
    changed
}

#[cfg(target_arch = "wasm32")]
fn media_time_label(frame: u32, fps_num: u32, fps_den: u32) -> String {
    let seconds = u64::from(frame) * u64::from(fps_den) / u64::from(fps_num.max(1));
    format!("{:02}:{:02}", seconds / 60, seconds % 60)
}

#[cfg(target_arch = "wasm32")]
fn point_in_quad(point: [f32; 2], quad: [[f32; 2]; 4]) -> bool {
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
