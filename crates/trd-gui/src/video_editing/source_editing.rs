use super::VideoEditingApp;

pub(super) fn initial_bound_instance(
    frame: &trd_core::DocumentFrame,
    mesh_count: usize,
) -> Option<u32> {
    (mesh_count == 1 && frame.objects.len() == 1 && frame.objects[0].quad.is_some()).then_some(0)
}

impl VideoEditingApp {
    pub(super) fn sync_source_controller(&mut self) -> Result<(), String> {
        let Some(scene) = self.arrow_scene.as_ref() else {
            return Ok(());
        };
        let Some(source) = scene.source.clone() else {
            return Ok(());
        };
        let Some(row) = scene.source_row(self.displayed_frame_index) else {
            self.source_controller_row = None;
            self.controller.state.selected = None;
            return Ok(());
        };
        let source = source.borrow();
        if source.meshes().is_empty() || self.source_controller_row == Some(row) {
            return Ok(());
        }

        let frame = source.frame(row).map_err(|error| error.to_string())?;
        let count = frame.objects.len();
        if count == 0 {
            self.source_controller_row = None;
            self.controller.state.selected = None;
            return Ok(());
        }

        let mut materials = Vec::with_capacity(count);
        let mut mesh_ids = Vec::with_capacity(count);
        for index in 0..count {
            let slot = source
                .object_mesh_slot(&frame, index)
                .map_err(|error| error.to_string())?;
            materials.push(
                source.meshes()[slot]
                    .material()
                    .map_err(|error| error.to_string())?
                    .clone(),
            );
            mesh_ids.push(u32::try_from(slot).map_err(|_| "too many meshes".to_owned())?);
        }
        self.source_model_baselines = vec![None; count];
        self.source_applied_adjustments = vec![trd_core::Matrix4::IDENTITY; count];
        let state = &mut self.controller.state;
        state.objects = vec![crate::scene::ObjectTransform::default(); count];
        state.mesh_ids = mesh_ids;
        state.materials = materials;
        state.modes = vec![trd_core::RenderMode::Shaded; count];
        state.image_based_lighting.resize(count, Default::default());
        state.tone_mappings.resize(count, Default::default());
        state.pbr_debug_views.resize(count, Default::default());
        state.selected = (self.selected_quad && self.source_selected_instance < count)
            .then_some(self.source_selected_instance as u32);
        self.controller.target = crate::interaction::InteractionTarget::Object;
        self.controller.move_reference_axes = super::QUAD_MOVE_AXES;
        self.controller.rebase_reset();
        self.source_controller_row = Some(row);
        Ok(())
    }

    /// Apply the adjustment to each original local model, never to the previous
    /// drag result. Every tracked frame keeps its own camera, quad and base.
    pub(super) fn apply_source_transforms(&mut self) -> Result<bool, String> {
        let Some(row) = self.source_controller_row else {
            return Ok(false);
        };
        let Some(source) = self
            .arrow_scene
            .as_ref()
            .and_then(|scene| scene.source.clone())
        else {
            return Ok(false);
        };
        if self.shared.video_playing.get() {
            return Ok(false);
        }
        let mut edits = Vec::new();
        let mut changes = Vec::new();
        for (index, transform) in self.controller.state.objects.iter().enumerate() {
            let adjustment = transform.model_matrix();
            let Some(previous) = self.source_applied_adjustments.get(index) else {
                return Err("source editor transform state is out of sync".to_owned());
            };
            if adjustment != *previous {
                let baseline = self
                    .source_model_baselines
                    .get_mut(index)
                    .ok_or_else(|| "source editor matrix baseline is missing".to_owned())?;
                if baseline.is_none() {
                    *baseline = Some(
                        source
                            .borrow()
                            .model_track(row, index)
                            .map_err(|error| error.to_string())?,
                    );
                }
                edits.extend(
                    baseline
                        .as_ref()
                        .expect("captured above")
                        .iter()
                        .map(|base| trd_core::ModelEdit {
                            model: adjustment * base.model,
                            ..*base
                        }),
                );
                changes.push((index, adjustment));
            }
        }
        if edits.is_empty() {
            return Ok(false);
        }
        source
            .borrow_mut()
            .apply_model_edits(&edits)
            .map_err(|error| error.to_string())?;
        for (index, adjustment) in changes {
            self.source_applied_adjustments[index] = adjustment;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction::{InteractionEvent, MoveDirection, TransformMode};
    use std::rc::Rc;

    fn app() -> VideoEditingApp {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../trd-core/tests/golden/stage1.arrow");
        let source = trd_core::SceneDocument::read(&std::fs::read(fixture).unwrap()).unwrap();
        app_from_source(source)
    }

    fn app_from_source(source: trd_core::SceneDocument) -> VideoEditingApp {
        let mut app = VideoEditingApp::player(
            super::super::tests::document().video,
            Rc::new(super::super::VideoEditingShared::default()),
        );
        app.set_arrow_scene(Some(Rc::new(
            super::super::ArrowScene::from_source(source).unwrap(),
        )));
        app.selected_quad = true;
        app.sync_source_controller().unwrap();
        app.controller.state.selected = Some(0);
        app
    }

    #[test]
    #[ignore = "requires the matching FIBA/Dragon params fixture"]
    fn dragon_edit_replay_persists_the_model_in_all_222_sparse_frames() {
        let path = std::env::var_os("TRD_DRAGON_PARAMS").expect("set TRD_DRAGON_PARAMS");
        let original = trd_core::SceneDocument::read(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(original.row_count(), 222);
        for omit_model in [false, true] {
            let source = if omit_model {
                let indices = original
                    .schema()
                    .fields()
                    .iter()
                    .enumerate()
                    .filter_map(|(index, field)| (field.name() != "model").then_some(index))
                    .collect::<Vec<_>>();
                let batches = original
                    .batches()
                    .iter()
                    .map(|batch| batch.project(&indices).unwrap())
                    .collect::<Vec<_>>();
                trd_core::SceneDocument::from_batches(
                    batches[0].schema(),
                    batches,
                    original.meshes().to_vec(),
                )
                .unwrap()
            } else {
                original.clone()
            };
            let before = source.clone();
            let mut author = app_from_source(source);
            author.controller.state.objects[0].translation = [0.2, 0.1, -0.15];
            author.controller.state.objects[0].yaw = 0.4;
            author.controller.state.objects[0].scale = [0.8; 3];
            let adjustment = author.controller.state.objects[0].model_matrix();
            assert!(author.apply_source_transforms().unwrap());
            let document = author
                .arrow_scene
                .as_ref()
                .unwrap()
                .source
                .as_ref()
                .unwrap()
                .borrow();
            let exported = document.write().unwrap();
            let reopened = trd_core::SceneDocument::read(&exported).unwrap();
            assert_eq!(
                reopened.row_count(),
                222,
                "no invented rows for the video-only tail"
            );
            assert_eq!(reopened.meshes(), original.meshes());
            assert_eq!(reopened.schema().metadata(), before.schema().metadata());
            for (expected_batch, actual_batch) in before.batches().iter().zip(reopened.batches()) {
                assert_eq!(expected_batch.num_rows(), actual_batch.num_rows());
                for field in before
                    .schema()
                    .fields()
                    .iter()
                    .filter(|field| field.name() != "model")
                {
                    assert_eq!(
                        expected_batch.column_by_name(field.name()),
                        actual_batch.column_by_name(field.name())
                    );
                }
            }
            for row in 0..222 {
                let mut expected = before.frame(row).unwrap();
                expected.objects[0].model = adjustment * expected.objects[0].model;
                assert_eq!(
                    reopened.frame(row).unwrap(),
                    expected,
                    "row {row}, absent={omit_model}"
                );
            }
            let mut replay = app_from_source(reopened);
            for index in [110, 221, 222, 287, 0] {
                replay.displayed_frame_index = index;
                replay.sync_source_controller().unwrap();
                assert!(!replay.apply_source_transforms().unwrap());
            }
            assert_eq!(
                replay
                    .arrow_scene
                    .as_ref()
                    .unwrap()
                    .source
                    .as_ref()
                    .unwrap()
                    .borrow()
                    .write()
                    .unwrap(),
                exported,
                "play/seek in a fresh replay must leave every saved matrix unchanged",
            );
        }
    }

    #[test]
    fn existing_move_scale_rotate_controls_write_source_model_without_accumulating_base() {
        let mut app = app();
        let original = app
            .arrow_scene
            .as_ref()
            .unwrap()
            .source
            .as_ref()
            .unwrap()
            .borrow()
            .clone();
        app.controller.mode = TransformMode::Move;
        app.controller.move_direction = MoveDirection::Reference1;
        assert!(app
            .controller
            .apply(InteractionEvent::Primary { dx: 0.2, dy: 0.0 }));
        app.controller.mode = TransformMode::Rotate;
        assert!(app
            .controller
            .apply(InteractionEvent::Primary { dx: 0.1, dy: 0.0 }));
        app.controller.mode = TransformMode::Scale;
        assert!(app.controller.apply(InteractionEvent::Scale { delta: 1.0 }));
        let adjustment = app.controller.state.objects[0].model_matrix();
        assert!(app.apply_source_transforms().unwrap());
        assert!(!app.apply_source_transforms().unwrap());
        let source = app
            .arrow_scene
            .as_ref()
            .unwrap()
            .source
            .as_ref()
            .unwrap()
            .borrow();
        let reopened = trd_core::SceneDocument::read(&source.write().unwrap()).unwrap();
        assert_eq!(reopened.row_count(), original.row_count());
        assert_eq!(reopened.meshes(), original.meshes());
        assert_eq!(reopened.schema(), original.schema());
        assert_eq!(
            reopened
                .batches()
                .iter()
                .map(|batch| batch.num_rows())
                .collect::<Vec<_>>(),
            original
                .batches()
                .iter()
                .map(|batch| batch.num_rows())
                .collect::<Vec<_>>(),
        );
        for row in 0..original.row_count() {
            let mut expected = original.frame(row).unwrap();
            expected.objects[0].model = adjustment * expected.objects[0].model;
            assert_eq!(source.frame(row).unwrap(), expected, "edited row {row}");
            assert_eq!(reopened.frame(row).unwrap(), expected, "reopened row {row}");
        }
        drop(source);

        app.displayed_frame_index = 1;
        app.sync_source_controller().unwrap();
        assert!(
            !app.apply_source_transforms().unwrap(),
            "seeking must not apply the adjustment twice"
        );
        let mut replay = VideoEditingApp::player(
            super::super::tests::document().video,
            Rc::new(super::super::VideoEditingShared::default()),
        );
        let exported = reopened.write().unwrap();
        replay.set_arrow_scene(Some(Rc::new(
            super::super::ArrowScene::from_source(reopened).unwrap(),
        )));
        for row in [0, 1, 2, 0] {
            replay.displayed_frame_index = row;
            replay.sync_source_controller().unwrap();
            assert!(!replay.apply_source_transforms().unwrap());
        }
        assert_eq!(
            replay
                .arrow_scene
                .as_ref()
                .unwrap()
                .source
                .as_ref()
                .unwrap()
                .borrow()
                .write()
                .unwrap(),
            exported,
            "fresh replay/playhead changes do not rewrite saved matrices",
        );
    }

    #[test]
    fn consecutive_drag_updates_replace_the_track_adjustment_instead_of_accumulating_it() {
        let mut app = app();
        let original = app
            .arrow_scene
            .as_ref()
            .unwrap()
            .source
            .as_ref()
            .unwrap()
            .borrow()
            .frames()
            .unwrap();
        for translation in [0.1, 0.2, -0.15, 0.0] {
            app.controller.state.objects[0].translation[0] = translation;
            let adjustment = app.controller.state.objects[0].model_matrix();
            assert!(app.apply_source_transforms().unwrap());
            let source = app
                .arrow_scene
                .as_ref()
                .unwrap()
                .source
                .as_ref()
                .unwrap()
                .borrow();
            for (row, original) in original.iter().enumerate() {
                let actual = source.frame(row).unwrap();
                assert_eq!(
                    actual.objects[0].model,
                    adjustment * original.objects[0].model
                );
                assert_eq!(&actual.objects[1..], &original.objects[1..]);
            }
        }
    }

    #[test]
    fn source_move_basis_keeps_e1_e2_e3_order_and_signed_directions() {
        for (direction, axis) in [
            (MoveDirection::Reference1, [1.0, 0.0, 0.0]),
            (MoveDirection::Reference2, [0.0, 0.0, -1.0]),
            (MoveDirection::Reference3, [0.0, 1.0, 0.0]),
        ] {
            let mut app = app();
            let original = app
                .arrow_scene
                .as_ref()
                .unwrap()
                .source
                .as_ref()
                .unwrap()
                .borrow()
                .frame(0)
                .unwrap()
                .objects[0]
                .model;
            app.controller.mode = TransformMode::Move;
            app.controller.move_direction = direction;
            assert!(app
                .controller
                .apply(InteractionEvent::Primary { dx: 0.2, dy: 0.0 }));
            let translation = app.controller.state.objects[0].translation;
            let amount: f32 = translation.iter().map(|component| component.abs()).sum();
            assert!(amount > 0.0);
            for (actual, expected) in translation.into_iter().zip(axis) {
                assert!((actual - expected * amount).abs() < 1e-6);
            }
            assert!(app.apply_source_transforms().unwrap());
            let source = app
                .arrow_scene
                .as_ref()
                .unwrap()
                .source
                .as_ref()
                .unwrap()
                .borrow();
            let columns = source.frame(0).unwrap().objects[0].model.to_cols_array();
            for index in 0..3 {
                assert!(
                    (columns[12 + index]
                        - original.to_cols_array()[12 + index]
                        - translation[index])
                        .abs()
                        < 1e-6
                );
            }
        }
    }

    #[test]
    fn changing_displayed_row_rebases_controls_without_modifying_source_matrices() {
        let mut app = app();
        let original = app
            .arrow_scene
            .as_ref()
            .unwrap()
            .source
            .as_ref()
            .unwrap()
            .borrow()
            .write()
            .unwrap();
        app.displayed_frame_index = 1;
        app.sync_source_controller().unwrap();
        assert_eq!(app.source_controller_row, Some(1));
        assert!(!app.apply_source_transforms().unwrap());
        let current = app
            .arrow_scene
            .as_ref()
            .unwrap()
            .source
            .as_ref()
            .unwrap()
            .borrow()
            .write()
            .unwrap();
        assert_eq!(current, original);
    }

    #[test]
    fn single_quad_and_mesh_have_an_automatic_initial_binding() {
        let object = trd_core::DocumentObject {
            mesh: Some(trd_core::DocumentMesh::Index(0)),
            model: trd_core::Matrix4::IDENTITY,
            quad: Some([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]),
            track_id: None,
            selection: trd_core::DrawSelection::INHERIT,
        };
        let mut frame = trd_core::DocumentFrame {
            params: trd_core::FrameParams::IDENTITY,
            source_size: None,
            present_index: Some(0),
            pts: None,
            objects: vec![object.clone()],
        };
        assert_eq!(initial_bound_instance(&frame, 1), Some(0));
        assert_eq!(initial_bound_instance(&frame, 0), None);
        frame.objects.push(object);
        assert_eq!(initial_bound_instance(&frame, 1), None);
    }

    #[test]
    fn picking_a_source_mesh_selects_its_bound_quad_and_enables_gizmos() {
        let mut app = app();
        app.selected_quad = false;
        app.show_gizmos = false;
        app.show_plane_grid = false;
        app.controller.state.selected = None;
        let shared = &app.shared;
        shared.pick_result.replace(Some(super::super::PickResult {
            id: shared.pick_revision.get(),
            source_generation: shared.source_generation.get(),
            render_revision: shared.render_revision.get(),
            hit: Some(0),
        }));
        app.consume_pick_result();
        assert_eq!(app.controller.state.selected, Some(0));
        assert!(app.selected_quad);
        assert!(app.show_gizmos && app.show_plane_grid);
        assert_eq!(app.source_selected_instance, 0);
    }
}
