use trd_core::{
    Camera, DocumentFrame, Draw, DrawableObject, FrameFit, Matrix4, Mesh, MeshAsset, RenderOptions,
    Scene, SceneDocument, Vertex, Viewport,
};

use crate::{
    quad_frame, quad_origin_model, quad_outline_model, CameraIntrinsics, PlacementError,
    PlacementQuad,
};

#[derive(Debug, thiserror::Error)]
pub enum DocumentSceneError {
    #[error(transparent)]
    Protocol(#[from] trd_core::ProtocolError),
    #[error(transparent)]
    Camera(#[from] trd_core::CameraFormError),
    #[error(transparent)]
    Placement(#[from] PlacementError),
    #[error("params row {0} is out of range")]
    Row(usize),
    #[error("the camera pose must be finite and invertible")]
    Pose,
}

/// The sole meshless input geometry is an explicit reference cube, never a
/// replacement for a failed GLB import.
pub fn document_assets(document: &SceneDocument) -> Result<Vec<MeshAsset>, DocumentSceneError> {
    if document.meshes().is_empty() {
        let vertices = [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]
        .into_iter()
        .map(|position| Vertex {
            position,
            color: [1.0; 3],
            uv: [0.0; 2],
        })
        .collect();
        return Ok(vec![MeshAsset::embedded(
            Mesh {
                vertices,
                indices: vec![
                    0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4,
                    7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
                ],
                shading: None,
            },
            trd_core::DisneyMaterial::default(),
        )]);
    }
    document
        .meshes()
        .iter()
        .enumerate()
        .map(|(slot, mesh)| {
            let slot = u32::try_from(slot).map_err(|_| DocumentSceneError::Row(slot))?;
            Ok(mesh.decode(slot)?)
        })
        .collect()
}

/// Assembles both source adapters through the same domain camera and scene.
pub fn document_scene(
    document: &SceneDocument,
    frame: &DocumentFrame,
    viewport: Viewport,
    options: &RenderOptions,
    frame_fit: Option<FrameFit>,
) -> Result<(Camera, Scene), DocumentSceneError> {
    let source_camera = frame
        .params
        .to_camera(frame.source_size.unwrap_or(viewport))?;
    let pose = source_camera.to_pose().matrix();
    if !pose.to_cols_array().iter().all(|value| value.is_finite())
        || !pose.determinant().is_finite()
        || pose.determinant() == 0.0
    {
        return Err(DocumentSceneError::Pose);
    }
    let mut params = frame.params;
    if let (Some(source), Some(mut k)) = (frame.source_size, params.k) {
        let sx = viewport.width as f32 / source.width as f32;
        let sy = viewport.height as f32 / source.height as f32;
        for i in [0, 3, 6] {
            k[i] *= sx;
        }
        for i in [1, 4, 7] {
            k[i] *= sy;
        }
        params.k = Some(k);
    }
    let camera = params.to_camera(viewport)?;
    let k = source_camera.to_intrinsics();
    let intrinsics = CameraIntrinsics {
        row_major: [k[0], k[3], k[6], k[1], k[4], k[7], k[2], k[5], k[8]],
    };
    let reference_only = document.meshes().is_empty();
    let mut draws = Vec::new();
    let mut reference = Vec::new();
    for (index, object) in frame.objects.iter().enumerate() {
        let origin = if let Some(points_px) = object.quad {
            let quad = quad_frame(intrinsics, PlacementQuad { points_px })?;
            if reference_only {
                reference.push(DrawableObject::quad_outline(
                    pose * quad_outline_model(quad),
                    false,
                ));
            }
            pose * quad_origin_model(quad)
        } else {
            Matrix4::IDENTITY
        };
        if reference_only {
            reference.push(DrawableObject::aabb_box(0, origin));
            reference.push(DrawableObject::coordinate_axes(origin));
        } else {
            let slot = document.object_mesh_slot(frame, index)?;
            draws.push(Draw {
                mesh_id: u32::try_from(slot).map_err(|_| DocumentSceneError::Row(slot))?,
                model: origin * object.model,
                selection: object.selection,
            });
        }
    }
    let mut scene = Scene::from_draws(&draws, options, frame_fit);
    scene.extend(reference);
    Ok((camera, scene))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn cg_source() -> SceneDocument {
        let vector = |values: [f32; 3]| -> ArrayRef {
            Arc::new(FixedSizeListArray::new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                3,
                Arc::new(Float32Array::from(values.to_vec())),
                None,
            ))
        };
        let arrays: Vec<ArrayRef> = vec![
            vector([2.0, 2.0, 4.0]),
            vector([0.0, 0.0, 0.0]),
            Arc::new(Float32Array::from(vec![0.7])),
        ];
        let schema = Arc::new(
            Schema::new(
                ["eye", "target", "fovy"]
                    .into_iter()
                    .zip(&arrays)
                    .map(|(name, array)| Field::new(name, array.data_type().clone(), false))
                    .collect::<Vec<_>>(),
            )
            .with_metadata(
                [
                    (
                        trd_core::PROTOCOL_VERSION_KEY.to_owned(),
                        trd_core::PROTOCOL_VERSION.to_owned(),
                    ),
                    (trd_core::TABLE_KIND_KEY.to_owned(), "params".to_owned()),
                ]
                .into_iter()
                .collect(),
            ),
        );
        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).unwrap();
        SceneDocument::from_batches(schema, vec![batch], vec![]).unwrap()
    }

    fn frame() -> DocumentFrame {
        DocumentFrame {
            params: trd_core::FrameParams {
                k: Some([1000.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 960.0, 540.0, 1.0]),
                ..trd_core::FrameParams::IDENTITY
            },
            source_size: Some(Viewport {
                width: 1920,
                height: 1080,
            }),
            present_index: Some(0),
            pts: None,
            objects: vec![trd_core::DocumentObject {
                mesh: None,
                model: Matrix4::IDENTITY,
                quad: Some([
                    [860.0, 440.0],
                    [1060.0, 440.0],
                    [1060.0, 640.0],
                    [860.0, 640.0],
                ]),
                track_id: None,
                selection: trd_core::DrawSelection::INHERIT,
            }],
        }
    }

    #[test]
    fn origin_basis_has_no_preview_scale_or_lift() {
        let frame = frame();
        let quad = quad_frame(
            CameraIntrinsics {
                row_major: [1000.0, 0.0, 960.0, 0.0, 1000.0, 540.0, 0.0, 0.0, 1.0],
            },
            PlacementQuad {
                points_px: frame.objects[0].quad.unwrap(),
            },
        )
        .unwrap();
        let matrix = quad_origin_model(quad);
        let columns = matrix.to_cols_array();
        assert!(columns[12].abs() < 1e-5);
        assert!(columns[13].abs() < 1e-5);
        assert!((columns[14] + 5.0).abs() < 1e-5);
        assert!((matrix.determinant() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn params_only_scene_has_a_centered_cube_and_axes_not_a_model() {
        let document = cg_source();
        let assets = document_assets(&document).unwrap();
        assert_eq!(assets.len(), 1);
        assert_eq!(assets[0].mesh.aabb().center().to_array(), [0.0; 3]);
        let (_, scene) = document_scene(
            &document,
            &document.frame(0).unwrap(),
            Viewport {
                width: 128,
                height: 128,
            },
            &RenderOptions::default(),
            None,
        )
        .unwrap();
        let primitives = scene
            .objects()
            .iter()
            .map(|object| object.primitive())
            .collect::<Vec<_>>();
        assert_eq!(
            primitives,
            [
                trd_core::Primitive::AabbBox { mesh_id: 0 },
                trd_core::Primitive::CoordinateAxes,
            ]
        );
    }

    #[test]
    fn placement_reference_includes_quad_axes_and_cube() {
        let document = cg_source();
        let (_, scene) = document_scene(
            &document,
            &frame(),
            Viewport {
                width: 1920,
                height: 1080,
            },
            &RenderOptions::default(),
            None,
        )
        .unwrap();
        let objects = scene.objects();
        assert_eq!(objects.len(), 3);
        assert_eq!(
            objects[0].primitive(),
            trd_core::Primitive::QuadOutline { selected: false }
        );
        assert_eq!(
            objects[1].primitive(),
            trd_core::Primitive::AabbBox { mesh_id: 0 }
        );
        assert_eq!(objects[2].primitive(), trd_core::Primitive::CoordinateAxes);
        assert_eq!(objects[1].model(), objects[2].model());
        assert_eq!(
            &objects[0].model().to_cols_array()[12..15],
            &objects[1].model().to_cols_array()[12..15],
        );
    }

    #[test]
    #[ignore = "requires a real GPU"]
    fn params_only_reference_renders_on_a_real_gpu() {
        pollster::block_on(async {
            let document = cg_source();
            let assets = document_assets(&document).unwrap();
            let instance = trd_core::create_instance();
            let gpu = trd_core::GpuContext::request(&instance, &trd_core::GpuRequest::default())
                .await
                .unwrap();
            eprintln!("document reference adapter: {:?}", gpu.adapter_facts());
            let mut renderer =
                trd_core::Renderer::with_assets(gpu, trd_core::TEXTURE_TARGET_FORMAT, &assets)
                    .unwrap();
            let target = renderer.create_texture_target(128, 128).unwrap();
            let (camera, scene) = document_scene(
                &document,
                &document.frame(0).unwrap(),
                Viewport {
                    width: 128,
                    height: 128,
                },
                &RenderOptions::default(),
                None,
            )
            .unwrap();
            renderer.draw_layers(&[trd_core::SceneLayer::new(camera, &scene)], &target);
            let pixels = renderer.read_pixels(&target).await.unwrap();
            assert_eq!(pixels.len(), 128 * 128 * 4);
            assert!(pixels
                .chunks_exact(4)
                .any(|pixel| pixel[..3].iter().any(|channel| *channel > 20)));
        });
    }
}
