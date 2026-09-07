//! Real-source placement geometry, not a substitute for the headed roundtrip.

use trd_core::{Matrix4, Point3, SceneDocument, Transform, Viewport};

#[test]
#[ignore = "requires the matching FIBA params plus Dragon GLB bundle"]
fn dragon_identity_has_default_placement_size_and_aabb_bottom_on_the_quad_plane() {
    let path = std::env::var_os("TRD_DRAGON_PARAMS").expect("set TRD_DRAGON_PARAMS");
    let document = SceneDocument::read(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(document.meshes().len(), 1);
    assert_eq!(document.row_count(), 222);
    let bounds = document.meshes()[0].bounds().unwrap();
    let min = bounds.min().to_array();
    let max = bounds.max().to_array();
    let bottom = Point3::new((min[0] + max[0]) / 2.0, min[1], (min[2] + max[2]) / 2.0);
    let mut maximum_anchor_error = 0.0f32;
    for index in 0..document.row_count() {
        let frame = document.frame(index).unwrap();
        assert_eq!(frame.objects.len(), 1);
        assert_eq!(frame.objects[0].model, Matrix4::IDENTITY);
        let viewport = frame.source_size.unwrap_or(Viewport {
            width: 1920,
            height: 1080,
        });
        let (camera, draws) =
            trd_placement::document_pick_draws(&document, &frame, viewport).unwrap();
        let k = camera.to_intrinsics();
        let quad = trd_placement::quad_frame(
            trd_placement::CameraIntrinsics {
                row_major: [k[0], k[3], k[6], k[1], k[4], k[7], k[2], k[5], k[8]],
            },
            trd_placement::PlacementQuad {
                points_px: frame.objects[0].quad.unwrap(),
            },
        )
        .unwrap();
        let expected_origin = [
            quad.origin_camera[0],
            -quad.origin_camera[1],
            -quad.origin_camera[2],
        ];
        let model = Transform::from_matrix(draws[0].model);
        let actual = model.transform_point(bottom).to_array();
        for (actual, expected) in actual.into_iter().zip(expected_origin) {
            maximum_anchor_error = maximum_anchor_error.max((actual - expected).abs());
        }
        let unplaced = trd_placement::quad_origin_model(quad)
            .unwrap()
            .try_inverse()
            .unwrap();
        let local = Transform::from_matrix(unplaced * draws[0].model).transform_aabb(bounds);
        assert!(
            local.min().y().abs() < 1e-5,
            "row {index}: AABB penetrates the plane"
        );
        let size = local.size().to_array();
        assert!(
            (size[0].max(size[1]).max(size[2]) - trd_placement::DEFAULT_PLACEMENT_EXTENT).abs()
                < 1e-5
        );
    }
    eprintln!(
        "222 Dragon placements: max AABB bottom-center anchor error {maximum_anchor_error:.8}"
    );
    assert!(maximum_anchor_error < 1e-5);
}

#[test]
#[ignore = "requires a real GPU and the matching FIBA/Dragon input"]
fn dragon_gpu_pick_uses_the_same_grounded_model_as_rendering() {
    let path = std::env::var_os("TRD_DRAGON_PARAMS").expect("set TRD_DRAGON_PARAMS");
    let document = SceneDocument::read(&std::fs::read(path).unwrap()).unwrap();
    pollster::block_on(async {
        let viewport = Viewport {
            width: 1920,
            height: 1080,
        };
        let instance = trd_core::create_instance();
        let gpu = trd_core::GpuContext::request(&instance, &trd_core::GpuRequest::default())
            .await
            .unwrap();
        eprintln!("Dragon picking adapter: {:?}", gpu.adapter_facts());
        let assets = trd_placement::document_assets(&document).unwrap();
        let mut renderer =
            trd_core::Renderer::with_assets(gpu, trd_core::TEXTURE_TARGET_FORMAT, &assets).unwrap();
        let frame = document.frame(0).unwrap();
        let (camera, draws) =
            trd_placement::document_pick_draws(&document, &frame, viewport).unwrap();
        let (_, scene) = trd_placement::document_scene(
            &document,
            &frame,
            viewport,
            &trd_core::RenderOptions::default(),
            None,
        )
        .unwrap();
        let rendered = scene
            .objects()
            .iter()
            .find(|object| matches!(object.primitive(), trd_core::Primitive::Mesh { .. }))
            .unwrap();
        assert_eq!(rendered.model(), draws[0].model);
        let clip = (camera.view_projection().matrix() * draws[0].model).to_cols_array();
        let mesh = &assets[0].mesh;
        let step = (mesh.indices.len() / 3 / 16).max(1);
        let mut hit = None;
        for triangle in mesh.indices.chunks_exact(3).step_by(step).take(16) {
            let mut center = [0.0; 4];
            for &index in triangle {
                let vertex = mesh.vertices[index as usize].position;
                for axis in 0..3 {
                    center[axis] += vertex[axis] / 3.0;
                }
            }
            center[3] = 1.0;
            let projected: [f32; 4] = std::array::from_fn(|row| {
                (0..4)
                    .map(|column| clip[column * 4 + row] * center[column])
                    .sum()
            });
            if projected[3] <= 0.0 {
                continue;
            }
            let x = (projected[0] / projected[3] + 1.0) * viewport.width as f32 / 2.0;
            let y = (1.0 - projected[1] / projected[3]) * viewport.height as f32 / 2.0;
            if !(0.0..viewport.width as f32).contains(&x)
                || !(0.0..viewport.height as f32).contains(&y)
            {
                continue;
            }
            hit = renderer
                .pick(camera, &draws, x as u32, y as u32, viewport)
                .await;
            if hit.is_some() {
                break;
            }
        }
        assert_eq!(hit, Some(0), "visible Dragon triangles were not pickable");
        assert_eq!(renderer.pick(camera, &draws, 0, 0, viewport).await, None);
    });
}
