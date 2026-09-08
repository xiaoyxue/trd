use std::rc::Rc;

use super::{ArrowScene, ErrorScope, VideoEditingApp, VideoEditingShared};
use crate::video_editing_renderer::VideoPlacementRenderer;

type Completion<T> = Box<dyn FnOnce(Result<T, String>)>;

pub(super) struct SeekCompletion {
    id: u64,
    complete: Completion<()>,
}

pub(super) enum SceneOperation {
    Replace {
        scene: Rc<ArrowScene>,
        renderer: VideoPlacementRenderer,
        complete: Completion<()>,
    },
    Reset {
        renderer: VideoPlacementRenderer,
        complete: Completion<()>,
    },
    Export {
        complete: Completion<Vec<u8>>,
    },
    Seek {
        seconds: f64,
        complete: Completion<()>,
    },
}

impl VideoEditingShared {
    /// Applies a prepared scene and its GPU assets together on the UI thread.
    pub fn replace_arrow_scene(
        &self,
        scene: Rc<ArrowScene>,
        renderer: VideoPlacementRenderer,
        complete: impl FnOnce(Result<(), String>) + 'static,
    ) {
        self.scene_operations
            .borrow_mut()
            .push_back(SceneOperation::Replace {
                scene,
                renderer,
                complete: Box::new(complete),
            });
        self.request_repaint();
    }

    /// Replaces scene assets with a video-only renderer without touching media.
    pub fn reset_arrow_scene(
        &self,
        renderer: VideoPlacementRenderer,
        complete: impl FnOnce(Result<(), String>) + 'static,
    ) {
        self.scene_operations
            .borrow_mut()
            .push_back(SceneOperation::Reset {
                renderer,
                complete: Box::new(complete),
            });
        self.request_repaint();
    }

    /// Returns a source snapshot, independently of the UI's download command.
    pub fn export_arrow_scene(&self, complete: impl FnOnce(Result<Vec<u8>, String>) + 'static) {
        self.scene_operations
            .borrow_mut()
            .push_back(SceneOperation::Export {
                complete: Box::new(complete),
            });
        self.request_repaint();
    }

    /// Uses the GUI's seek dispatch; completion requires a displayed answer, not just decoding.
    pub fn seek_scene_to_seconds(
        &self,
        seconds: f64,
        complete: impl FnOnce(Result<(), String>) + 'static,
    ) {
        self.scene_operations
            .borrow_mut()
            .push_back(SceneOperation::Seek {
                seconds,
                complete: Box::new(complete),
            });
        self.request_repaint();
    }

    pub(super) fn finish_scene_seeks(&self, answers_seek: u64) {
        let pending = std::mem::take(&mut *self.scene_seeks.borrow_mut());
        for seek in pending {
            if seek.id <= answers_seek {
                (seek.complete)(Ok(()));
            } else {
                self.scene_seeks.borrow_mut().push(seek);
            }
        }
    }

    pub(super) fn fail_scene_seeks(&self, message: &str) {
        let pending = std::mem::take(&mut *self.scene_seeks.borrow_mut());
        for seek in pending {
            (seek.complete)(Err(message.to_owned()));
        }
    }
}

impl VideoEditingApp {
    pub(super) fn process_scene_operations(&mut self) {
        // Old render/pick jobs own their renderer and source snapshot until completion.
        if self.shared.render_in_flight.get() || self.shared.pick_in_flight.get() {
            return;
        }
        loop {
            let operation = self.shared.scene_operations.borrow_mut().pop_front();
            let Some(operation) = operation else { break };
            match operation {
                SceneOperation::Replace {
                    scene,
                    renderer,
                    complete,
                } => {
                    if self.shared.video_loaded.get() {
                        if let Some(error) = self.arrow_scene_validation_error(&scene) {
                            self.shared.set_error(ErrorScope::Document, &error);
                            complete(Err(error));
                            continue;
                        }
                    }
                    self.clear_scene_state();
                    self.shared.set_renderer(renderer);
                    self.set_arrow_scene(Some(scene));
                    complete(Ok(()));
                }
                SceneOperation::Reset { renderer, complete } => {
                    self.clear_scene_state();
                    self.shared.set_renderer(renderer);
                    complete(Ok(()));
                }
                SceneOperation::Export { complete } => {
                    let result = self.export_source_snapshot();
                    if let Err(error) = &result {
                        self.shared.set_error(ErrorScope::Export, error);
                    } else {
                        self.shared.clear_error(ErrorScope::Export);
                    }
                    complete(result);
                }
                SceneOperation::Seek { seconds, complete } => {
                    if !seconds.is_finite() || seconds < 0.0 {
                        complete(Err("seek time must be finite and nonnegative".to_owned()));
                    } else if !self.shared.video_loaded.get() {
                        complete(Err("load a video before seeking".to_owned()));
                    } else {
                        let index = super::frame_index_at_media_time(
                            seconds,
                            self.video.fps_num,
                            self.video.fps_den,
                            self.presentable_frame_count(),
                        );
                        if index > i32::MAX as u32 {
                            complete(Err("seek frame exceeds the host command range".to_owned()));
                        } else {
                            let id = self.begin_seek(index);
                            self.shared
                                .scene_seeks
                                .borrow_mut()
                                .push(SeekCompletion { id, complete });
                        }
                    }
                }
            }
        }
    }

    pub(super) fn begin_seek(&mut self, frame_index: u32) -> u64 {
        self.current_frame_index = frame_index;
        let id = self.shared.request_seek(frame_index);
        self.pending_seek = Some(super::PendingSeek { frame_index, id });
        self.shared.request_repaint();
        id
    }

    fn clear_scene_state(&mut self) {
        self.set_arrow_scene(None);
        self.reset_all();
        self.displayed_diagnostics = None;
        self.last_rendered_frame_index = None;
        self.shared.rendered_frame.borrow_mut().take();
        self.shared.pending_pick.set(None);
        self.shared.pick_result.borrow_mut().take();
        self.shared.asset_defaults.borrow_mut().take();
        self.shared.incoming_document.borrow_mut().take();
        self.shared.incoming_scene.borrow_mut().take();
        self.shared.pending_document.borrow_mut().take();
        self.document_url.clear();
        if self.shared.command.get() == super::COMMAND_EXPORT_ARROW {
            self.shared.command.set(super::COMMAND_NONE);
        }
    }

    fn export_source_snapshot(&mut self) -> Result<Vec<u8>, String> {
        self.apply_source_transforms()?;
        let source = self
            .arrow_scene
            .as_ref()
            .and_then(|scene| scene.source.as_ref())
            .ok_or_else(|| "no current Arrow document to export".to_owned())?;
        source.borrow().write().map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    type Captured<T> = Rc<RefCell<Option<Result<T, String>>>>;

    #[test]
    fn api_seek_uses_gui_dispatch_and_waits_for_a_matching_displayed_answer() {
        let mut app = app();
        app.video.frame_count = 288;
        app.video.fps_num = 24;
        app.video.fps_den = 1;
        app.shared.set_video_status(true, true);
        let (result, complete) = completion();
        app.shared.seek_scene_to_seconds(4.5, complete);
        app.process_scene_operations();
        assert_eq!(app.current_frame_index, 108);
        assert_eq!(app.shared.take_seek_frame(), Some(108));
        let id = app.pending_seek.unwrap().id;
        app.shared.finish_scene_seeks(id - 1);
        assert!(
            result.borrow().is_none(),
            "an old displayed frame cannot settle a new seek"
        );
        app.shared.finish_scene_seeks(id);
        result.borrow_mut().take().unwrap().unwrap();
        assert!(app.shared.video_playing.get());
    }

    #[test]
    fn api_seek_rejects_invalid_time_missing_video_and_media_failures() {
        let mut app = app();
        for seconds in [f64::NAN, -1.0, f64::INFINITY, 1.0] {
            let (result, complete) = completion();
            app.shared.seek_scene_to_seconds(seconds, complete);
            app.process_scene_operations();
            assert!(result.borrow_mut().take().unwrap().is_err());
        }
        app.shared.set_video_status(true, false);
        for change_source in [false, true] {
            let (result, complete) = completion();
            app.shared.seek_scene_to_seconds(0.0, complete);
            app.process_scene_operations();
            assert_eq!(app.shared.take_seek_frame(), Some(0));
            if change_source {
                app.shared.set_video_status(false, false);
            } else {
                app.shared
                    .set_error(ErrorScope::Media, "reader seek failed");
            }
            assert!(result.borrow_mut().take().unwrap().is_err());
        }
        assert!(app.shared.scene_seeks.borrow().is_empty());
    }

    #[test]
    fn overlapping_api_seeks_coalesce_to_the_gui_winner() {
        let mut app = app();
        app.video.frame_count = 288;
        app.video.fps_num = 24;
        app.video.fps_den = 1;
        app.shared.set_video_status(true, false);
        let (first, a) = completion();
        let (second, b) = completion();
        app.shared.seek_scene_to_seconds(1.0, a);
        app.shared.seek_scene_to_seconds(100.0, b);
        app.process_scene_operations();
        assert_eq!(app.shared.take_seek_frame(), Some(287));
        app.shared.finish_scene_seeks(app.pending_seek.unwrap().id);
        first.borrow_mut().take().unwrap().unwrap();
        second.borrow_mut().take().unwrap().unwrap();
    }

    fn scene(name: &str) -> Rc<ArrowScene> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../trd-core/tests/golden")
            .join(name);
        Rc::new(
            ArrowScene::from_source(
                trd_core::SceneDocument::read(&std::fs::read(path).unwrap()).unwrap(),
            )
            .unwrap(),
        )
    }

    fn app() -> VideoEditingApp {
        let mut video = super::super::tests::document().video;
        video.frame_count = 3;
        VideoEditingApp::player(video, Rc::new(VideoEditingShared::default()))
    }

    fn completion<T: 'static>() -> (Captured<T>, Completion<T>) {
        let value = Rc::new(RefCell::new(None));
        let target = value.clone();
        (
            value,
            Box::new(move |result| {
                target.replace(Some(result));
            }),
        )
    }

    #[test]
    fn export_waits_for_inflight_jobs_and_never_queues_a_download() {
        let mut app = app();
        let source = scene("stage2.arrow");
        let expected = source.source.as_ref().unwrap().borrow().write().unwrap();
        app.set_arrow_scene(Some(source));
        let (result, complete) = completion();
        app.shared.export_arrow_scene(complete);
        app.shared.render_in_flight.set(true);
        app.process_scene_operations();
        assert!(result.borrow().is_none());
        app.shared.render_in_flight.set(false);
        app.shared.pick_in_flight.set(true);
        app.process_scene_operations();
        assert!(result.borrow().is_none());
        app.shared.pick_in_flight.set(false);
        app.process_scene_operations();
        assert_eq!(result.borrow_mut().take().unwrap().unwrap(), expected);
        assert!(app.shared.take_arrow_export().is_none());
        assert_eq!(app.shared.command.get(), super::super::COMMAND_NONE);
    }

    #[test]
    fn export_flushes_current_edits_across_all_source_rows() {
        let mut app = app();
        let source = scene("stage2.arrow");
        let original = source.source.as_ref().unwrap().borrow().clone();
        app.set_arrow_scene(Some(source));
        app.sync_source_controller().unwrap();
        app.controller.state.objects[0].translation = [0.2, 0.1, -0.15];
        app.controller.state.objects[0].yaw = 0.4;
        let adjustment = app.controller.state.objects[0].model_matrix();
        let (result, complete) = completion();
        app.shared.export_arrow_scene(complete);
        app.process_scene_operations();
        let bytes = result.borrow_mut().take().unwrap().unwrap();
        let reopened = trd_core::SceneDocument::read(&bytes).unwrap();
        let mut expected = original.clone();
        let edits = original
            .model_track(0, 0)
            .unwrap()
            .into_iter()
            .map(|base| trd_core::ModelEdit {
                model: adjustment * base.model,
                ..base
            })
            .collect::<Vec<_>>();
        expected.apply_model_edits(&edits).unwrap();
        assert_eq!(reopened.write().unwrap(), expected.write().unwrap());
        assert_ne!(bytes, original.write().unwrap());
    }

    #[test]
    fn export_without_a_current_document_fails_explicitly() {
        let mut app = app();
        let (result, complete) = completion();
        app.shared.export_arrow_scene(complete);
        app.process_scene_operations();
        assert!(result
            .borrow_mut()
            .take()
            .unwrap()
            .unwrap_err()
            .contains("no current Arrow"));
        assert!(app
            .shared
            .error_text()
            .unwrap()
            .contains("no current Arrow"));
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn reset_releases_old_source_and_assets_preserves_media_and_loads_a_new_scene() {
        let env = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../assets/envmap/uffizi-large.hdr"),
        )
        .unwrap();
        let instance = trd_core::create_instance();
        let gpu = pollster::block_on(trd_core::GpuContext::request(
            &instance,
            &trd_core::GpuRequest::default(),
        ))
        .unwrap();
        let mut app = app();
        app.shared.set_shared_gpu(gpu.clone());
        app.shared.video_loaded.set(true);
        app.shared.video_playing.set(true);
        app.shared.source_generation.set(42);
        app.shared.seek_generation.set(7);
        app.shared.dispatched_seek.set(7);
        app.current_frame_index = 1;
        app.displayed_frame_index = 1;
        app.shared
            .latest_video_frame
            .replace(Some(super::super::IncomingVideoFrame {
                rgba: vec![1, 2, 3, 255],
                width: 1,
                height: 1,
                frame_index: 1,
                media_time_seconds: 0.5,
                duration_seconds: 0.04,
                source_generation: 42,
                answers_seek: 7,
            }));
        let source = scene("stage2.arrow");
        let weak_source = Rc::downgrade(source.source.as_ref().unwrap());
        let renderer =
            VideoPlacementRenderer::new_arrow_scene_with_gpu(gpu.clone(), &source, &env, 96, 54)
                .unwrap();
        let (loaded, complete) = completion();
        app.shared.replace_arrow_scene(source, renderer, complete);
        assert!(loaded.borrow().is_none());
        app.process_scene_operations();
        loaded.borrow_mut().take().unwrap().unwrap();
        assert_eq!(
            app.arrow_scene
                .as_ref()
                .unwrap()
                .source
                .as_ref()
                .unwrap()
                .borrow()
                .meshes()
                .len(),
            2
        );
        let old_generation = app.shared.renderer_generation.get();

        let empty = VideoPlacementRenderer::new_empty_with_gpu(gpu.clone(), 96, 54).unwrap();
        let (reset, complete) = completion();
        app.shared.reset_arrow_scene(empty, complete);
        app.shared.render_in_flight.set(true);
        app.process_scene_operations();
        assert!(reset.borrow().is_none());
        app.shared.render_in_flight.set(false);
        app.process_scene_operations();
        reset.borrow_mut().take().unwrap().unwrap();
        assert!(weak_source.upgrade().is_none());
        assert!(app.arrow_scene.is_none());
        assert!(app
            .shared
            .renderer
            .borrow()
            .as_ref()
            .unwrap()
            .diagnostics()
            .asset
            .is_none());
        assert!(app.shared.renderer_generation.get() > old_generation);
        assert!(app.shared.video_loaded.get() && app.shared.video_playing.get());
        assert_eq!(app.shared.source_generation.get(), 42);
        assert_eq!(app.shared.seek_generation.get(), 7);
        assert_eq!((app.current_frame_index, app.displayed_frame_index), (1, 1));
        let video = app.shared.latest_video_frame.borrow();
        let frame = video.as_ref().unwrap();
        assert_eq!(
            (
                frame.frame_index,
                frame.media_time_seconds,
                frame.answers_seek
            ),
            (1, 0.5, 7)
        );
        assert_eq!(frame.rgba, [1, 2, 3, 255]);
        drop(video);
        assert!(app
            .export_source_snapshot()
            .unwrap_err()
            .contains("no current Arrow"));

        let next = scene("stage1.arrow");
        let expected = next.source.as_ref().unwrap().borrow().write().unwrap();
        let renderer =
            VideoPlacementRenderer::new_arrow_scene_with_gpu(gpu.clone(), &next, &env, 96, 54)
                .unwrap();
        let (loaded, complete) = completion();
        app.shared
            .replace_arrow_scene(next.clone(), renderer, complete);
        app.process_scene_operations();
        loaded.borrow_mut().take().unwrap().unwrap();
        assert_eq!(app.export_source_snapshot().unwrap(), expected);
        assert_eq!(next.source.as_ref().unwrap().borrow().meshes().len(), 1);

        let source = next.source.as_ref().unwrap().borrow();
        let short = trd_core::SceneDocument::from_batches(
            source.schema().clone(),
            vec![source.batches()[0].slice(0, 1)],
            source.meshes().to_vec(),
        )
        .unwrap();
        drop(source);
        let invalid = Rc::new(ArrowScene::from_source(short).unwrap());
        let renderer =
            VideoPlacementRenderer::new_arrow_scene_with_gpu(gpu, &invalid, &env, 96, 54).unwrap();
        let generation = app.shared.renderer_generation.get();
        let (rejected, complete) = completion();
        app.shared.replace_arrow_scene(invalid, renderer, complete);
        app.process_scene_operations();
        assert!(rejected
            .borrow_mut()
            .take()
            .unwrap()
            .unwrap_err()
            .contains("params rows"));
        assert_eq!(app.shared.renderer_generation.get(), generation);
        assert_eq!(app.export_source_snapshot().unwrap(), expected);
    }
}
