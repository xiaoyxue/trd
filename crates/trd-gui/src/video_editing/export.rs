use super::{ErrorScope, VideoEditingApp, VideoEditingShared, COMMAND_EXPORT_ARROW};

#[derive(Debug, Clone)]
pub struct ArrowScene {
    pub mesh_resources: Vec<trd_core::MeshResource>,
    pub frames: Vec<trd_core::DecodedFrame>,
    pub frame_rate: f64,
}

impl ArrowScene {
    pub fn unresolved_mesh_references(&self) -> Vec<(u32, trd_core::MeshReference)> {
        self.mesh_resources
            .iter()
            .enumerate()
            .filter_map(|(index, resource)| match resource {
                trd_core::MeshResource::Gltf(reference) => Some((index as u32, reference.clone())),
                trd_core::MeshResource::Resolved(_) => None,
            })
            .collect()
    }

    pub fn resolve_gltf(&mut self, index: u32, bytes: &[u8]) -> Result<(), String> {
        let count = self.mesh_resources.len();
        let resource = self
            .mesh_resources
            .get_mut(index as usize)
            .ok_or_else(|| format!("mesh reference index {index} is out of range ({count})"))?;
        if !matches!(resource, trd_core::MeshResource::Gltf(_)) {
            return Err(format!("mesh row {index} is already resolved"));
        }
        *resource = trd_core::MeshResource::Resolved(Box::new(
            trd_core::import_glb(bytes)
                .map(trd_core::MeshAsset::from)
                .map_err(|error| error.to_string())?,
        ));
        Ok(())
    }

    pub fn mesh_assets(&self) -> Result<Vec<trd_core::MeshAsset>, String> {
        self.mesh_resources
            .iter()
            .enumerate()
            .map(|(index, resource)| match resource {
                trd_core::MeshResource::Resolved(asset) => Ok(asset.as_ref().clone()),
                trd_core::MeshResource::Gltf(reference) => Err(format!(
                    "mesh row {index} reference `{}` is unresolved",
                    reference.display()
                )),
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub enum VideoEditingInput {
    Annotation(trd_core::VideoEditingDocument),
    Scene(ArrowScene),
}

#[derive(Debug)]
pub struct ArrowExport {
    pub filename: String,
    pub bytes: Vec<u8>,
    pub frame_count: u32,
    pub placed_frame_count: u32,
}

pub fn decode_video_editing_input(bytes: &[u8]) -> Result<VideoEditingInput, String> {
    match trd_core::decode_video_editing_document(bytes) {
        Ok(document) => Ok(VideoEditingInput::Annotation(document)),
        Err(document_error) => decode_arrow_scene(bytes)
            .map(VideoEditingInput::Scene)
            .map_err(|scene_error| {
                format!(
                    "input is neither a video-editing document ({document_error}) nor a \
                     protocol {} scene ({scene_error})",
                    trd_core::PROTOCOL_VERSION
                )
            }),
    }
}

fn decode_arrow_scene(bytes: &[u8]) -> Result<ArrowScene, String> {
    let mut session = trd_core::InputSession::new();
    let batches = session.push(bytes).map_err(|error| error.to_string())?;
    session.finish().map_err(|error| error.to_string())?;
    if session.mesh_resource_count() == 0 {
        return Err("scene has no mesh table".to_owned());
    }
    let frames: Vec<_> = batches.into_iter().flatten().collect();
    if frames.is_empty() {
        return Err("scene has no params rows".to_owned());
    }
    let mesh_count = session.mesh_resource_count();
    if let Some(draw) = frames
        .iter()
        .filter_map(|frame| frame.draws.as_ref())
        .flatten()
        .find(|draw| draw.mesh_id as usize >= mesh_count)
    {
        return Err(format!(
            "draw references mesh {}, but the scene has {mesh_count} mesh row(s)",
            draw.mesh_id
        ));
    }
    Ok(ArrowScene {
        mesh_resources: session.mesh_resources().to_vec(),
        frames,
        frame_rate: session.frame_rate().unwrap_or(trd_core::DEFAULT_FRAME_RATE),
    })
}

#[derive(Debug, thiserror::Error)]
enum ArrowExportError {
    #[error("load a video before exporting")]
    NoVideo,
    #[error("load an annotation document before exporting")]
    NoDocument,
    #[error("select a catalog object before exporting")]
    NoSelectedAsset,
    #[error("wait for the selected catalog object to finish loading")]
    NoLoadedAsset,
    #[error("the video has no presentable frames")]
    NoPresentableFrames,
    #[error("the video frame rate is invalid")]
    InvalidFrameRate,
    #[error("tracked frame {0} has no camera intrinsics")]
    MissingIntrinsics(u32),
    #[error("tracked frame {0} has no valid placement")]
    InvalidPlacement(u32),
    #[error("the document has no tracked frame with a valid placement")]
    NoPlacedFrames,
    #[error(transparent)]
    Encode(#[from] trd_core::SceneEncodeError),
}

impl VideoEditingShared {
    pub fn pending_arrow_export_filename(&self) -> Option<String> {
        self.pending_export
            .borrow()
            .as_ref()
            .map(|export| export.filename.clone())
    }

    pub fn take_arrow_export(&self) -> Option<ArrowExport> {
        self.pending_export.borrow_mut().take()
    }

    pub fn complete_arrow_export(&self, result: Result<String, String>) {
        self.pending_export.borrow_mut().take();
        match &result {
            Ok(_) => self.clear_error(ErrorScope::Export),
            Err(error) => self.set_error(ErrorScope::Export, error),
        }
        self.export_status.replace(Some(result));
        self.request_repaint();
    }

    pub fn cancel_arrow_export(&self) {
        let cancelled = self.pending_export.borrow_mut().take().is_some();
        self.clear_error(ErrorScope::Export);
        if cancelled {
            self.export_status
                .replace(Some(Ok("Arrow export cancelled".to_owned())));
        } else {
            self.export_status.borrow_mut().take();
        }
        self.request_repaint();
    }

    pub(super) fn clear_export_asset(&self) {
        self.export_asset.borrow_mut().take();
    }

    fn queue_arrow_export(&self, export: ArrowExport) {
        let message = format!(
            "{} placed frame(s) across {} params row(s) ready as {}",
            export.placed_frame_count, export.frame_count, export.filename
        );
        self.pending_export.replace(Some(export));
        self.export_status.replace(Some(Ok(message)));
        self.clear_error(ErrorScope::Export);
        self.command.set(COMMAND_EXPORT_ARROW);
        self.request_repaint();
    }
}

impl VideoEditingApp {
    pub(super) fn arrow_export_disabled_reason(&self) -> Option<String> {
        if self.shared.pending_export.borrow().is_some() {
            return Some("An Arrow export is already waiting to be saved".to_owned());
        }
        if !self.shared.video_loaded.get() {
            return Some(ArrowExportError::NoVideo.to_string());
        }
        let Some(document) = self.document.as_ref() else {
            return Some(ArrowExportError::NoDocument.to_string());
        };
        if self.selected_asset.is_none() {
            return Some(ArrowExportError::NoSelectedAsset.to_string());
        }
        if self.shared.export_asset.borrow().is_none() {
            return Some(ArrowExportError::NoLoadedAsset.to_string());
        }
        let presentable = self.presentable_frame_count();
        let mut has_placement = false;
        for frame in document
            .frames
            .iter()
            .filter(|frame| frame.tracked && frame.video_frame_index < presentable)
        {
            if frame.k.is_none() {
                return Some(
                    ArrowExportError::MissingIntrinsics(frame.video_frame_index).to_string(),
                );
            }
            if self.placement_model_at(frame.video_frame_index).is_none() {
                return Some(
                    ArrowExportError::InvalidPlacement(frame.video_frame_index).to_string(),
                );
            }
            has_placement = true;
        }
        (!has_placement).then(|| ArrowExportError::NoPlacedFrames.to_string())
    }

    pub(super) fn request_arrow_export(&self) {
        match self.build_arrow_export() {
            Ok(export) => self.shared.queue_arrow_export(export),
            Err(error) => self.shared.complete_arrow_export(Err(error.to_string())),
        }
    }

    pub(super) fn arrow_export_status(&self) -> Option<Result<String, String>> {
        self.shared.export_status.borrow().clone()
    }

    fn build_arrow_export(&self) -> Result<ArrowExport, ArrowExportError> {
        if !self.shared.video_loaded.get() {
            return Err(ArrowExportError::NoVideo);
        }
        let document = self.document.as_ref().ok_or(ArrowExportError::NoDocument)?;
        self.selected_asset
            .ok_or(ArrowExportError::NoSelectedAsset)?;
        let asset = self
            .shared
            .export_asset
            .borrow()
            .clone()
            .ok_or(ArrowExportError::NoLoadedAsset)?;
        let frame_count = self.presentable_frame_count();
        if frame_count == 0 {
            return Err(ArrowExportError::NoPresentableFrames);
        }
        if self.video.fps_num == 0 || self.video.fps_den == 0 {
            return Err(ArrowExportError::InvalidFrameRate);
        }

        let identity_k = trd_core::Matrix3::IDENTITY.to_cols_array();
        let mut params = Vec::with_capacity(frame_count as usize);
        let mut draws = Vec::with_capacity(frame_count as usize);
        let mode = self.controller.state.mode_of(0);
        let mut placed_frame_count = 0_u32;

        for frame_index in 0..frame_count {
            let mut frame_params = trd_core::FrameParams {
                k: Some(identity_k),
                ..trd_core::FrameParams::IDENTITY
            };
            let frame_draws = match document.frame(frame_index) {
                Some(frame) if frame.tracked => {
                    let k = frame
                        .k
                        .ok_or(ArrowExportError::MissingIntrinsics(frame_index))?;
                    let model = self
                        .placement_model_at(frame_index)
                        .ok_or(ArrowExportError::InvalidPlacement(frame_index))?;
                    frame_params.k = Some(super::protocol_k_from_row_major(k));
                    placed_frame_count += 1;
                    vec![trd_core::Draw {
                        mesh_id: 0,
                        model,
                        selection: trd_core::DrawSelection::Mesh(Some(mode)),
                    }]
                }
                _ => Vec::new(),
            };
            params.push(frame_params);
            draws.push(frame_draws);
        }

        if placed_frame_count == 0 {
            return Err(ArrowExportError::NoPlacedFrames);
        }

        let material = self
            .controller
            .state
            .materials
            .first()
            .ok_or(ArrowExportError::NoLoadedAsset)?;
        let (scene_mesh, texture): (trd_core::SceneMesh<'_>, Option<&dyn trd_core::Texture>) =
            match asset.as_ref() {
                crate::video_editing_renderer::VideoExportAsset::Embedded { mesh, texture } => (
                    trd_core::SceneMesh::Embedded { mesh, material },
                    Some(texture),
                ),
                crate::video_editing_renderer::VideoExportAsset::Gltf(reference) => {
                    (trd_core::SceneMesh::Gltf(reference), None)
                }
            };
        let frame_rate = f64::from(self.video.fps_num) / f64::from(self.video.fps_den);
        let bytes = trd_core::encode_scene_resources(
            std::slice::from_ref(&scene_mesh),
            texture,
            &params,
            Some(&draws),
            Some(frame_rate),
        )?;

        Ok(ArrowExport {
            filename: export_filename(&self.video.source_name),
            bytes,
            frame_count,
            placed_frame_count,
        })
    }

    pub(super) fn presentable_frame_count(&self) -> u32 {
        let tail = self
            .video
            .unpresented_tail
            .map_or(0, |tail| tail.samples.min(self.video.frame_count));
        self.video.frame_count.saturating_sub(tail)
    }
}

fn export_filename(source_name: &str) -> String {
    let name = source_name
        .split(['/', '\\'])
        .next_back()
        .unwrap_or(source_name);
    let stem = name.rsplit_once('.').map_or(name, |(stem, _)| stem).trim();
    let stem = if stem.is_empty() { "scene" } else { stem };
    format!("{stem}.scene.arrow")
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;
    use crate::video_editing::CatalogAsset;
    use crate::video_editing_renderer::VideoExportAsset;

    fn tracked_document() -> trd_core::VideoEditingDocument {
        let k = [1000.0, 0.0, 960.0, 0.0, 1000.0, 540.0, 0.0, 0.0, 1.0];
        let quad = [
            [960.0, 540.0],
            [1160.0, 540.0],
            [1160.0, 740.0],
            [960.0, 740.0],
        ];
        trd_core::VideoEditingDocument {
            video: trd_core::VideoInfo {
                source_name: "C:\\video\\shot.mp4".to_owned(),
                mime: "video/mp4".to_owned(),
                codec: "h264".to_owned(),
                sha256: String::new(),
                byte_length: 1,
                width: 1920,
                height: 1080,
                fps_num: 24,
                fps_den: 1,
                frame_count: 4,
                duration_us: 166_667,
                unpresented_tail: Some(trd_core::UnpresentedTail {
                    samples: 1,
                    evidence: trd_core::UnpresentedTailEvidence::SampleTable,
                }),
            },
            poster_bytes: Vec::new(),
            frames: vec![
                trd_core::VideoEditingFrame {
                    video_frame_index: 0,
                    present_index: 0,
                    timestamp_us: 0,
                    k: Some(k),
                    placement_quad: Some(quad),
                    tracked: true,
                },
                trd_core::VideoEditingFrame {
                    video_frame_index: 2,
                    present_index: 2,
                    timestamp_us: 83_333,
                    k: Some(k),
                    placement_quad: Some(quad),
                    tracked: true,
                },
            ],
        }
    }

    fn export_ready_app() -> VideoEditingApp {
        let shared = Rc::new(VideoEditingShared::default());
        let document = tracked_document();
        let mut app = VideoEditingApp::new(document, shared.clone());
        app.selected_asset = Some(CatalogAsset::CocaColaCan);
        app.selected_quad = true;
        app.controller.state.modes[0] = trd_core::RenderMode::Wireframe;
        shared.video_loaded.set(true);
        shared
            .export_asset
            .replace(Some(Rc::new(VideoExportAsset::Embedded {
                mesh: trd_core::Mesh::from_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap(),
                texture: trd_core::ImageTexture::from_rgba(1, 1, vec![255; 4]).unwrap(),
            })));
        app
    }

    #[test]
    fn sparse_document_exports_a_dense_presentable_timeline() {
        let app = export_ready_app();
        let expected_first = app.placement_model_at(0).unwrap();
        let expected_last = app.placement_model_at(2).unwrap();
        let export = app.build_arrow_export().unwrap();

        assert_eq!(export.filename, "shot.scene.arrow");
        assert_eq!(export.frame_count, 3);
        assert_eq!(export.placed_frame_count, 2);

        let mut session = trd_core::InputSession::new();
        let batches = session.push(&export.bytes).unwrap();
        session.finish().unwrap();
        let decoded: Vec<_> = batches.into_iter().flatten().collect();

        assert_eq!(decoded.len(), 3);
        assert_eq!(session.frame_rate(), Some(24.0));
        assert_eq!(decoded[0].draws.as_ref().unwrap()[0].model, expected_first);
        assert!(decoded[1].draws.as_ref().unwrap().is_empty());
        assert_eq!(decoded[2].draws.as_ref().unwrap()[0].model, expected_last);
        assert_eq!(
            decoded[0].draws.as_ref().unwrap()[0].selection,
            trd_core::DrawSelection::Mesh(Some(trd_core::RenderMode::Wireframe))
        );
        assert_eq!(
            decoded[0].params.k,
            Some([1000.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 960.0, 540.0, 1.0])
        );
    }

    #[test]
    fn export_request_queues_one_immutable_file() {
        let mut app = export_ready_app();
        app.request_arrow_export();

        assert_eq!(
            app.shared.take_command(),
            Some(super::super::VideoEditingCommand::ExportArrow)
        );
        assert_eq!(
            app.shared.pending_arrow_export_filename().as_deref(),
            Some("shot.scene.arrow")
        );
        let export = app.shared.take_arrow_export().unwrap();
        assert!(!export.bytes.is_empty());
        assert!(app.shared.take_arrow_export().is_none());
        let VideoEditingInput::Scene(scene) = decode_video_editing_input(&export.bytes).unwrap()
        else {
            panic!("export must decode as a protocol scene");
        };
        app.set_arrow_scene(Some(Rc::new(scene)));
        assert!(app.document.is_none());
        assert_eq!(app.arrow_scene.as_ref().unwrap().frames.len(), 3);
    }

    #[test]
    fn filename_uses_the_source_basename() {
        assert_eq!(
            export_filename("https://example.com/media/clip.mp4"),
            "clip.scene.arrow"
        );
        assert_eq!(export_filename(""), "scene.scene.arrow");
    }

    #[test]
    fn coca_obj_exports_edited_material_and_albedo() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mesh = trd_core::Mesh::from_obj(
            &std::fs::read_to_string(root.join("assets/meshes/can/coke.obj")).unwrap(),
        )
        .unwrap();
        let texture = crate::assets::decode_texture(
            &std::fs::read(root.join("assets/meshes/can/can_around.jpg")).unwrap(),
        )
        .unwrap();
        let mut app = export_ready_app();
        let material = trd_core::DisneyMaterial {
            metallic: 0.33,
            roughness: 0.61,
            clearcoat: 0.27,
            specular: 0.72,
            auxiliary: trd_core::Auxiliary {
                textures: trd_core::MaterialTextures {
                    base_color: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..trd_core::DisneyMaterial::default()
        };
        app.controller.state.materials[0] = material.clone();
        app.controller.state.modes[0] = trd_core::RenderMode::Shaded;
        app.shared
            .export_asset
            .replace(Some(Rc::new(VideoExportAsset::Embedded {
                mesh,
                texture: texture.clone(),
            })));

        let export = app.build_arrow_export().unwrap();
        let VideoEditingInput::Scene(scene) = decode_video_editing_input(&export.bytes).unwrap()
        else {
            panic!("OBJ export must decode as a scene");
        };
        let assets = scene.mesh_assets().unwrap();

        assert_eq!(assets.len(), 1);
        assert_eq!(assets[0].material, material);
        assert_eq!(assets[0].base_color_texture.as_ref(), Some(&texture));
    }

    #[test]
    fn dragon_exports_only_a_reference_then_imports_all_glb_maps() {
        let relative = "assets/meshes/glb/Meshy_AI_Dragon_0804104424_texture.glb";
        let reference = trd_core::MeshReference::new(Some(relative.to_owned()), None).unwrap();
        let app = export_ready_app();
        app.shared
            .export_asset
            .replace(Some(Rc::new(VideoExportAsset::Gltf(reference.clone()))));

        let export = app.build_arrow_export().unwrap();
        let VideoEditingInput::Scene(mut scene) =
            decode_video_editing_input(&export.bytes).unwrap()
        else {
            panic!("Dragon export must decode as a scene");
        };
        assert_eq!(
            scene.mesh_resources,
            vec![trd_core::MeshResource::Gltf(reference)]
        );

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        scene
            .resolve_gltf(0, &std::fs::read(root.join(relative)).unwrap())
            .unwrap();
        let assets = scene.mesh_assets().unwrap();
        assert!(assets[0].base_color_texture.is_some());
        assert!(assets[0].metallic_roughness_texture.is_some());
        assert!(assets[0].normal_texture.is_some());
    }

    #[test]
    fn real_annotation_input_is_not_misclassified_as_a_scene() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/videos/fiba/fiba-shot1.arrow");
        let bytes = std::fs::read(path).unwrap();

        assert!(matches!(
            decode_video_editing_input(&bytes).unwrap(),
            VideoEditingInput::Annotation(_)
        ));
    }

    #[test]
    fn replay_rejects_a_scene_for_a_different_timeline() {
        let mut app = export_ready_app();
        app.set_arrow_scene(Some(Rc::new(ArrowScene {
            mesh_resources: vec![trd_core::MeshResource::Resolved(Box::new(
                trd_core::MeshAsset::embedded(
                    trd_core::Mesh::from_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap(),
                    trd_core::DisneyMaterial::default(),
                ),
            ))],
            frames: vec![trd_core::DecodedFrame {
                params: trd_core::FrameParams::IDENTITY,
                draws: Some(Vec::new()),
                frame_ref: None,
                frame_id: None,
            }],
            frame_rate: 24.0,
        })));

        assert!(app.arrow_scene.is_none());
        assert!(app.shared.error_text().unwrap().contains("params rows"));
    }

    #[test]
    fn queued_input_only_publishes_its_own_format() {
        let shared = VideoEditingShared::default();
        shared.queue_annotation_document(tracked_document());
        assert!(shared.take_incoming_document().unwrap().is_some());
        assert!(shared.take_incoming_scene().is_none());

        shared.queue_arrow_scene(Rc::new(ArrowScene {
            mesh_resources: vec![trd_core::MeshResource::Resolved(Box::new(
                trd_core::MeshAsset::embedded(
                    trd_core::Mesh::from_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap(),
                    trd_core::DisneyMaterial::default(),
                ),
            ))],
            frames: vec![trd_core::DecodedFrame {
                params: trd_core::FrameParams::IDENTITY,
                draws: Some(Vec::new()),
                frame_ref: None,
                frame_id: None,
            }],
            frame_rate: 24.0,
        }));
        assert!(shared.take_incoming_scene().unwrap().is_some());
        assert!(shared.take_incoming_document().is_none());
    }
}
