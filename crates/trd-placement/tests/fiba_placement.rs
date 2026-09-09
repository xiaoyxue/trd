//! Cross-language placement parity on real FIBA rows, before accepting UI snapshots.

use std::path::PathBuf;

use trd_core::{Matrix4, Primitive, RenderOptions, SceneDocument, Viewport};

fn input(variable: &str) -> PathBuf {
    std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {variable} to the generated FIBA input"))
}

fn project(matrix: Matrix4, point: [f32; 3], width: u32, height: u32) -> [f32; 2] {
    let m = matrix.to_cols_array();
    let p = [point[0], point[1], point[2], 1.0];
    let clip: [f32; 4] =
        std::array::from_fn(|row| (0..4).map(|column| m[column * 4 + row] * p[column]).sum());
    assert!(clip[3] > 0.0);
    [
        (clip[0] / clip[3] + 1.0) * width as f32 / 2.0,
        (1.0 - clip[1] / clip[3]) * height as f32 / 2.0,
    ]
}

#[test]
#[ignore = "requires FIBA params and support/direct_basis_reference.py output"]
fn equal_length_fiba_quad_origin_axes_and_cube_match_python() {
    let bytes = std::fs::read(input("TRD_FIBA_PARAMS")).unwrap();
    let document = SceneDocument::read(&bytes).unwrap();
    assert!(
        document.meshes().is_empty(),
        "Case 1 must not contain GLB resources"
    );
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(input("TRD_FIBA_REFERENCE")).unwrap()).unwrap();
    let rows = expected["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 222);
    assert_eq!(document.row_count(), rows.len());
    let viewport = Viewport {
        width: rows[0]["width"].as_u64().unwrap() as u32,
        height: rows[0]["height"].as_u64().unwrap() as u32,
    };
    let mut maximum_corner_error = 0.0f32;
    let mut maximum_origin_error = 0.0f32;
    let mut maximum_cube_anchor_error = 0.0f32;
    let mut maximum_vertex_error = 0.0f32;
    let mut maximum_axis_length_error = 0.0f32;
    let cube = trd_core::Mesh::reference_cube().unwrap();
    assert_eq!(cube.aabb().min().to_array(), [-0.5; 3]);
    assert_eq!(cube.aabb().max().to_array(), [0.5; 3]);
    for (index, reference) in rows.iter().enumerate() {
        let frame = document.frame(index).unwrap();
        assert_eq!(
            frame.present_index,
            Some(reference["video_frame_index"].as_i64().unwrap())
        );
        assert_eq!(frame.present_index, reference["present_index"].as_i64());
        assert_eq!(frame.objects[0].model, Matrix4::IDENTITY);
        assert_eq!(reference["width"].as_u64(), Some(u64::from(viewport.width)));
        assert_eq!(
            reference["height"].as_u64(),
            Some(u64::from(viewport.height))
        );
        let source_quad: Vec<[f32; 2]> = serde_json::from_value(reference["quad"].clone()).unwrap();
        let input_quad = frame.objects[0].quad.unwrap();
        for (actual, original) in input_quad
            .iter()
            .flatten()
            .zip(source_quad.iter().flatten())
        {
            assert!(
                (actual - original).abs() < 0.0005,
                "frame {index}: converter changed a corner"
            );
        }
        let (camera, scene) = trd_placement::document_scene(
            &document,
            &frame,
            viewport,
            &RenderOptions::default(),
            None,
        )
        .unwrap();
        let objects = scene.objects();
        assert_eq!(objects.len(), 3);
        assert_eq!(
            objects[0].primitive(),
            Primitive::QuadOutline { selected: false }
        );
        assert_eq!(objects[1].primitive(), Primitive::CoordinateAxes);
        assert_eq!(objects[2].primitive(), Primitive::AabbBox { mesh_id: 0 });
        let expected_axes: [f32; 16] =
            serde_json::from_value(reference["axis_model"].clone()).unwrap();
        for (actual, expected) in objects[1]
            .model()
            .to_cols_array()
            .into_iter()
            .zip(expected_axes)
        {
            assert!(
                (actual - expected).abs() < 0.002,
                "frame {index}: axes differ from Python"
            );
        }
        let expected_cube: [f32; 16] =
            serde_json::from_value(reference["cube_model"].clone()).unwrap();
        for (actual, expected) in objects[2]
            .model()
            .to_cols_array()
            .into_iter()
            .zip(expected_cube)
        {
            assert!(
                (actual - expected).abs() < 0.002,
                "frame {index}: cube differs from Python placement"
            );
        }
        let vp = camera.view_projection().matrix();
        let cube_columns = objects[2].model().to_cols_array();
        let target_length = reference["placement_axis_length"].as_f64().unwrap() as f32;
        for start in [0, 4, 8] {
            let length = cube_columns[start..start + 3]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            maximum_axis_length_error =
                maximum_axis_length_error.max((length - target_length).abs());
        }
        let handedness = reference["handedness"].as_f64().unwrap() as f32;
        for (coefficients_key, pixels_key) in [
            ("unit_quad_coefficients", "unit_quad_pixels"),
            ("unit_axes_coefficients", "unit_axes_pixels"),
        ] {
            let coefficients: Vec<[f32; 3]> =
                serde_json::from_value(reference[coefficients_key].clone()).unwrap();
            let pixels: Vec<[f32; 2]> =
                serde_json::from_value(reference[pixels_key].clone()).unwrap();
            assert_eq!(coefficients.len(), pixels.len());
            for ([u, v, w], expected) in coefficients.into_iter().zip(pixels) {
                let actual = project(
                    vp * objects[2].model(),
                    [u, w - 0.5, -handedness * v],
                    viewport.width,
                    viewport.height,
                );
                for axis in 0..2 {
                    maximum_vertex_error =
                        maximum_vertex_error.max((actual[axis] - expected[axis]).abs());
                }
            }
        }
        for (corner, expected) in [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ]
        .into_iter()
        .zip(&source_quad)
        {
            let actual = project(
                vp * objects[0].model(),
                corner,
                viewport.width,
                viewport.height,
            );
            for axis in 0..2 {
                maximum_corner_error =
                    maximum_corner_error.max((actual[axis] - expected[axis]).abs());
            }
        }
        let actual = project(
            vp * objects[1].model(),
            [0.0; 3],
            viewport.width,
            viewport.height,
        );
        let expected: [f32; 2] = serde_json::from_value(reference["origin_px"].clone()).unwrap();
        for axis in 0..2 {
            maximum_origin_error = maximum_origin_error.max((actual[axis] - expected[axis]).abs());
        }
        let anchor = project(
            vp * objects[2].model(),
            [0.0, -0.5, 0.0],
            viewport.width,
            viewport.height,
        );
        for axis in 0..2 {
            maximum_cube_anchor_error =
                maximum_cube_anchor_error.max((anchor[axis] - expected[axis]).abs());
        }
    }
    eprintln!(
        "222 FIBA rows: max corner reprojection error {maximum_corner_error:.6}px; \
         max origin error {maximum_origin_error:.6}px; \
         max cube bottom-center error {maximum_cube_anchor_error:.6}px; \
         max independently expanded vertex error {maximum_vertex_error:.6}px; \
         max equal-axis length error {maximum_axis_length_error:.8}"
    );
    assert!(maximum_corner_error < 0.05);
    assert!(maximum_origin_error < 0.05);
    assert!(maximum_cube_anchor_error < 0.05);
    assert!(maximum_vertex_error < 0.05);
    assert!(maximum_axis_length_error < 1e-5);
}
