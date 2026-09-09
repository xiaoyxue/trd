use trd_core::{
    Camera, DocumentFrame, Draw, DrawableObject, FrameFit, Matrix4, MeshAsset, RenderOptions,
    Scene, SceneDocument, Viewport,
};

use crate::{
    grounded_asset_model, quad_axes_model, quad_frame, quad_origin_model, quad_outline_model,
    reference_cube_model, CameraIntrinsics, PlacementError, PlacementQuad,
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlacementOverlays {
    pub quad: bool,
    pub axes: bool,
    pub cube: bool,
    pub grid: bool,
    pub selected: Option<usize>,
    pub hovered: Option<usize>,
}

impl PlacementOverlays {
    pub const ALL: Self = Self {
        quad: true,
        axes: true,
        cube: true,
        grid: false,
        selected: None,
        hovered: None,
    };
}

/// The sole meshless input geometry is an explicit reference cube, never a
/// replacement for a failed GLB import.
pub fn document_assets(document: &SceneDocument) -> Result<Vec<MeshAsset>, DocumentSceneError> {
    Ok(document.decoded_assets()?)
}

/// Assembles both source adapters through the same domain camera and scene.
pub fn document_scene(
    document: &SceneDocument,
    frame: &DocumentFrame,
    viewport: Viewport,
    options: &RenderOptions,
    frame_fit: Option<FrameFit>,
) -> Result<(Camera, Scene), DocumentSceneError> {
    let overlays = if document.meshes().is_empty() {
        PlacementOverlays::ALL
    } else {
        PlacementOverlays::default()
    };
    let (camera, [mut background, foreground]) =
        document_scene_with_overlays(document, frame, viewport, options, frame_fit, overlays)?;
    background.extend(foreground.objects().iter().copied());
    Ok((camera, background))
}

/// Back-to-front scenes: video/quad guides, then meshes and their own gizmos.
/// Submit both with `Renderer::draw_layers`; primitive sorting within one scene
/// cannot keep depth-disabled quad axes below an opaque mesh.
pub fn document_scene_with_overlays(
    document: &SceneDocument,
    frame: &DocumentFrame,
    viewport: Viewport,
    options: &RenderOptions,
    frame_fit: Option<FrameFit>,
    overlays: PlacementOverlays,
) -> Result<(Camera, [Scene; 2]), DocumentSceneError> {
    let (camera, scenes, _) =
        assemble_document_scene(document, frame, viewport, options, frame_fit, overlays)?;
    Ok((camera, scenes))
}

pub fn document_pick_draws(
    document: &SceneDocument,
    frame: &DocumentFrame,
    viewport: Viewport,
) -> Result<(Camera, Vec<Draw>), DocumentSceneError> {
    let (camera, _, draws) = assemble_document_scene(
        document,
        frame,
        viewport,
        &RenderOptions::default(),
        None,
        PlacementOverlays::default(),
    )?;
    Ok((camera, draws))
}

fn assemble_document_scene(
    document: &SceneDocument,
    frame: &DocumentFrame,
    viewport: Viewport,
    options: &RenderOptions,
    frame_fit: Option<FrameFit>,
    overlays: PlacementOverlays,
) -> Result<(Camera, [Scene; 2], Vec<Draw>), DocumentSceneError> {
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
    let mut cubes = Vec::new();
    let config = document.render_config();
    let automatic_shadows = config.shadow.enable
        && !frame
            .objects
            .iter()
            .any(|object| !object.selection.is_mesh());
    let mut shadows = Vec::new();
    for (index, object) in frame.objects.iter().enumerate() {
        let mut reference_frame = Matrix4::IDENTITY;
        let mut cube_model = Matrix4::IDENTITY;
        let selected = overlays.selected == Some(index);
        let show_local = overlays.selected.is_none() || selected;
        let origin = if let Some(points_px) = object.quad {
            let quad = quad_frame(intrinsics, PlacementQuad { points_px })?;
            reference_frame = pose * quad_axes_model(quad);
            cube_model = pose * reference_cube_model(quad)?;
            if overlays.quad {
                if selected || overlays.hovered == Some(index) {
                    reference.push(DrawableObject::quad_fill(pose * quad_outline_model(quad)));
                }
                reference.push(DrawableObject::quad_outline(
                    pose * quad_outline_model(quad),
                    selected,
                ));
            }
            if overlays.grid && show_local {
                reference.push(DrawableObject::plane_grid(
                    trd_core::GridPlane::Xy,
                    pose * quad_outline_model(quad),
                ));
            }
            pose * quad_origin_model(quad)?
        } else {
            Matrix4::IDENTITY
        };
        if overlays.cube && reference_only && show_local {
            cubes.push(DrawableObject::aabb_box(0, cube_model));
        }
        if overlays.axes && show_local {
            reference.push(DrawableObject::coordinate_axes(reference_frame));
        }
        if !reference_only {
            let slot = document.object_mesh_slot(frame, index)?;
            let bounds = document.meshes()[slot].bounds()?;
            let asset_model = if object.quad.is_some() {
                grounded_asset_model(bounds)?
            } else {
                Matrix4::IDENTITY
            };
            if automatic_shadows {
                let bounds = trd_core::Transform::from_matrix(object.model * asset_model)
                    .transform_aabb(bounds);
                let ground_y = if object.quad.is_some() {
                    0.0
                } else {
                    bounds.min().y()
                };
                let shadow = DrawableObject::blob_shadow_for_bounds(bounds, ground_y);
                shadows.push(DrawableObject::blob_shadow(origin * shadow.model()));
            }
            draws.push(Draw {
                mesh_id: u32::try_from(slot).map_err(|_| DocumentSceneError::Row(slot))?,
                model: origin * object.model * asset_model,
                selection: object.selection,
            });
        }
    }
    let mut foreground =
        Scene::from_draws_with_config(&draws, options, None, config).without_automatic_shadows();
    foreground.extend(shadows);
    foreground.extend(cubes);
    let background = Scene::from(reference)
        .with_background(trd_core::Background {
            frame: frame_fit,
            environment: foreground.background_mut().environment.take(),
        })
        .with_lighting(foreground.lighting());
    Ok((camera, [background, foreground], draws))
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
    fn blob_shadows_stay_on_support_planes_and_respect_document_config() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../trd-core/tests/golden/stage1.arrow");
        let meshes = SceneDocument::read(&std::fs::read(fixture).unwrap()).unwrap();
        let source = cg_source();
        let mut document = SceneDocument::from_batches(
            source.schema().clone(),
            source.batches().to_vec(),
            vec![meshes.meshes()[0].clone()],
        )
        .unwrap();
        let mut frame = frame();
        frame.objects[0].model = trd_core::Transform::from_scale_rotation_translation(
            trd_core::Vector3::new(0.8, 1.2, 0.6),
            trd_core::Rotation::from_rotation_x(0.4),
            trd_core::Vector3::new(0.3, 2.0, -0.2),
        )
        .matrix();
        let viewport = Viewport {
            width: 1920,
            height: 1080,
        };
        let render = |document: &SceneDocument, frame: &DocumentFrame| {
            document_scene(document, frame, viewport, &RenderOptions::default(), None)
                .unwrap()
                .1
        };
        let enabled = render(&document, &frame);
        let shadow = enabled
            .objects()
            .iter()
            .find(|object| object.primitive() == trd_core::Primitive::BlobShadow)
            .unwrap();
        let quad = quad_frame(
            CameraIntrinsics {
                row_major: [1000.0, 0.0, 960.0, 0.0, 1000.0, 540.0, 0.0, 0.0, 1.0],
            },
            PlacementQuad {
                points_px: frame.objects[0].quad.unwrap(),
            },
        )
        .unwrap();
        let local = (quad_origin_model(quad).unwrap().inverse() * shadow.model()).to_cols_array();
        for index in [1, 5, 13] {
            assert!(
                local[index].abs() < 1e-5,
                "blob left the support plane: {local:?}"
            );
        }
        let mut config = document.render_config();
        config.shadow.enable = false;
        document.set_render_config(config).unwrap();
        let disabled = render(&document, &frame);
        assert_eq!(
            disabled.objects(),
            enabled
                .objects()
                .iter()
                .copied()
                .filter(|object| object.primitive() != trd_core::Primitive::BlobShadow)
                .collect::<Vec<_>>()
        );

        config.shadow.enable = true;
        document.set_render_config(config).unwrap();
        frame.objects[0].quad = None;
        let non_quad = render(&document, &frame);
        let shadow = non_quad
            .objects()
            .iter()
            .find(|object| object.primitive() == trd_core::Primitive::BlobShadow)
            .unwrap();
        let bounds = trd_core::Transform::from_matrix(frame.objects[0].model)
            .transform_aabb(document.meshes()[0].bounds().unwrap());
        assert_eq!(shadow.model().to_cols_array()[13], bounds.min().y());
        let mut authored = frame.objects[0].clone();
        authored.selection = trd_core::DrawSelection::Shadow;
        authored.model = Matrix4::IDENTITY;
        frame.objects.push(authored);
        let scene = render(&document, &frame);
        let shadows = scene
            .objects()
            .iter()
            .filter(|object| object.primitive() == trd_core::Primitive::BlobShadow)
            .collect::<Vec<_>>();
        assert_eq!(shadows.len(), 1, "authored shadows must not be duplicated");
        assert_eq!(shadows[0].model(), Matrix4::IDENTITY);
    }

    #[test]
    fn origin_basis_uses_equal_half_edge_lengths_without_offset_or_lift() {
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
        let matrix = quad_origin_model(quad).unwrap();
        let columns = matrix.to_cols_array();
        assert!(columns[12].abs() < 1e-5);
        assert!(columns[13].abs() < 1e-5);
        assert!((columns[14] + 5.0).abs() < 1e-5);
        let unit_length = quad.axis_length;
        for start in [0, 4, 8] {
            let length = columns[start..start + 3]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            assert!((length - unit_length).abs() < 1e-5);
        }
        assert!((matrix.determinant() - unit_length.powi(3)).abs() < 1e-5);
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
                trd_core::Primitive::CoordinateAxes,
                trd_core::Primitive::AabbBox { mesh_id: 0 },
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
        assert_eq!(objects[1].primitive(), trd_core::Primitive::CoordinateAxes);
        assert_eq!(
            objects[2].primitive(),
            trd_core::Primitive::AabbBox { mesh_id: 0 }
        );
        let cube = trd_core::Transform::from_matrix(objects[2].model());
        let bottom_center = cube
            .transform_point(trd_core::Point3::new(0.0, -0.5, 0.0))
            .to_array();
        let origin = objects[1].model().to_cols_array();
        for (actual, expected) in bottom_center.into_iter().zip(&origin[12..15]) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn selected_quad_adds_highlight_and_plane_grid() {
        let document = cg_source();
        let (_, [background, foreground]) = document_scene_with_overlays(
            &document,
            &frame(),
            Viewport {
                width: 1920,
                height: 1080,
            },
            &RenderOptions::default(),
            None,
            PlacementOverlays {
                grid: true,
                selected: Some(0),
                ..PlacementOverlays::ALL
            },
        )
        .unwrap();
        let objects = background.objects();
        assert_eq!(objects.len(), 4);
        assert_eq!(objects[0].primitive(), trd_core::Primitive::QuadFill);
        assert_eq!(
            objects[1].primitive(),
            trd_core::Primitive::QuadOutline { selected: true }
        );
        assert_eq!(
            objects[2].primitive(),
            trd_core::Primitive::PlaneGrid {
                plane: trd_core::GridPlane::Xy
            }
        );
        assert_eq!(objects[3].primitive(), trd_core::Primitive::CoordinateAxes);
        assert_eq!(foreground.objects().len(), 1);
        assert_eq!(
            foreground.objects()[0].primitive(),
            trd_core::Primitive::AabbBox { mesh_id: 0 }
        );
    }

    #[test]
    fn coordinate_and_plane_visibility_are_independent() {
        let document = cg_source();
        for (axes, grid) in [(true, false), (false, true), (true, true), (false, false)] {
            let (_, [scene, _]) = document_scene_with_overlays(
                &document,
                &frame(),
                Viewport {
                    width: 1920,
                    height: 1080,
                },
                &RenderOptions::default(),
                None,
                PlacementOverlays {
                    quad: true,
                    axes,
                    grid,
                    selected: Some(0),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(
                scene
                    .objects()
                    .iter()
                    .any(|object| object.primitive() == trd_core::Primitive::CoordinateAxes),
                axes,
            );
            assert_eq!(
                scene.objects().iter().any(|object| matches!(
                    object.primitive(),
                    trd_core::Primitive::PlaneGrid { .. }
                )),
                grid,
            );
        }
    }

    #[test]
    fn hovering_a_quad_adds_fill_without_selecting_it() {
        let document = cg_source();
        let (_, [scene, _]) = document_scene_with_overlays(
            &document,
            &frame(),
            Viewport {
                width: 1920,
                height: 1080,
            },
            &RenderOptions::default(),
            None,
            PlacementOverlays {
                quad: true,
                hovered: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(scene.objects().len(), 2);
        assert_eq!(
            scene.objects()[0].primitive(),
            trd_core::Primitive::QuadFill
        );
        assert_eq!(
            scene.objects()[1].primitive(),
            trd_core::Primitive::QuadOutline { selected: false }
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
