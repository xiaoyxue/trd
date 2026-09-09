//! Pixel coverage for the three params/GLB input forms, through real placement.
//! These headless snapshots complement, rather than replace, native/Chrome UI E2E.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use arrow::array::{
    Array, ArrayRef, FixedSizeBinaryArray, FixedSizeListArray, Float32Array, ListArray,
    RecordBatch, StringArray, StructArray,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use trd_core::{
    GlbMesh, GpuContext, Matrix4, ModelEdit, RenderMode, RenderOptions, Renderer, SceneDocument,
    SceneLayer, Viewport,
};

#[path = "../../trd-core/tests/support/golden_image.rs"]
mod golden_image;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 180;
static GPU_SERIAL: Mutex<()> = Mutex::new(());

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn golden(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.png"))
}

fn fixture_meshes() -> Vec<GlbMesh> {
    let bytes = std::fs::read(root().join("crates/trd-core/tests/golden/stage2.arrow")).unwrap();
    SceneDocument::read(&bytes).unwrap().meshes().to_vec()
}

fn quad(center: f32, width: f32) -> [[f32; 2]; 4] {
    [
        [center - width * 0.35, 106.0],
        [center + width * 0.35, 106.0],
        [center + width * 0.5, 146.0],
        [center - width * 0.5, 146.0],
    ]
}

fn source(quads: &[[[f32; 2]; 4]], meshes: &[GlbMesh], include_model: bool) -> SceneDocument {
    let floats = |values: Vec<f32>| -> ArrayRef { Arc::new(Float32Array::from(values)) };
    let points = StructArray::new(
        ["x", "y", "w", "h"]
            .map(|name| Arc::new(Field::new(name, DataType::Float32, false)))
            .to_vec()
            .into(),
        vec![
            floats(
                quads
                    .iter()
                    .flatten()
                    .map(|point| 2.0 * point[0] - WIDTH as f32)
                    .collect(),
            ),
            floats(
                quads
                    .iter()
                    .flatten()
                    .map(|point| 2.0 * point[1] - HEIGHT as f32)
                    .collect(),
            ),
            floats(vec![WIDTH as f32; quads.len() * 4]),
            floats(vec![HEIGHT as f32; quads.len() * 4]),
        ],
        None,
    );
    let quads_array = FixedSizeListArray::new(
        Arc::new(Field::new("item", points.data_type().clone(), false)),
        4,
        Arc::new(points),
        None,
    );
    let placements = ListArray::new(
        Arc::new(Field::new("item", quads_array.data_type().clone(), false)),
        OffsetBuffer::from_lengths([quads.len()]),
        Arc::new(quads_array),
        None,
    );
    let k = StructArray::new(
        ["fx", "fy", "cx", "cy", "skew", "w", "h"]
            .map(|name| Arc::new(Field::new(name, DataType::Float32, false)))
            .to_vec()
            .into(),
        [480.0, 480.0, 0.0, 0.0, 0.0, WIDTH as f32, HEIGHT as f32]
            .map(|value| floats(vec![value]))
            .to_vec(),
        None,
    );
    let mut columns: Vec<ArrayRef> = vec![Arc::new(placements), Arc::new(k)];
    let mut fields = ["bottom_quads", "k"]
        .into_iter()
        .zip(&columns)
        .map(|(name, array)| Field::new(name, array.data_type().clone(), false))
        .collect::<Vec<_>>();
    if include_model {
        let rows = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            4,
            floats(
                (0..quads.len())
                    .flat_map(|_| Matrix4::IDENTITY.to_cols_array())
                    .collect(),
            ),
            None,
        );
        let matrices = FixedSizeListArray::new(
            Arc::new(Field::new("item", rows.data_type().clone(), false)),
            4,
            Arc::new(rows),
            None,
        );
        let models = ListArray::new(
            Arc::new(Field::new("item", matrices.data_type().clone(), false)),
            OffsetBuffer::from_lengths([quads.len()]),
            Arc::new(matrices),
            None,
        );
        fields.push(Field::new("model", models.data_type().clone(), false));
        columns.push(Arc::new(models));
    }
    if !meshes.is_empty() {
        assert_eq!(quads.len(), meshes.len());
        let ids =
            FixedSizeBinaryArray::try_from_iter(meshes.iter().map(|mesh| *mesh.id().as_bytes()))
                .unwrap();
        let item = Field::new("item", DataType::FixedSizeBinary(16), false).with_metadata(
            [("ARROW:extension:name".to_owned(), "arrow.uuid".to_owned())]
                .into_iter()
                .collect(),
        );
        let bindings = ListArray::new(
            Arc::new(item),
            OffsetBuffer::from_lengths([meshes.len()]),
            Arc::new(ids),
            None,
        );
        fields.push(Field::new("mesh_id", bindings.data_type().clone(), false));
        columns.push(Arc::new(bindings));
    }
    fields.push(Field::new("upstream_note", DataType::Utf8, true));
    columns.push(Arc::new(StringArray::from(vec![Some("keep this field")])));
    let schema = Arc::new(
        Schema::new(fields).with_metadata(
            [("upstream.owner".to_owned(), "golden fixture".to_owned())]
                .into_iter()
                .collect(),
        ),
    );
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    SceneDocument::from_batches(schema, vec![batch], meshes.to_vec()).unwrap()
}

fn gpu() -> Arc<GpuContext> {
    let instance = trd_core::create_instance();
    let gpu = pollster::block_on(GpuContext::request(
        &instance,
        &trd_core::GpuRequest::default(),
    ))
    .unwrap();
    let facts = gpu.adapter_facts();
    eprintln!(
        "placement golden adapter: {} ({}, {})",
        facts.name, facts.backend, facts.device_type
    );
    assert_eq!(
        facts.device_type, "DiscreteGpu",
        "goldens require a real discrete GPU"
    );
    gpu
}

fn render(document: &SceneDocument, gpu: Arc<GpuContext>) -> Vec<u8> {
    render_with_background(document, gpu, None)
}

fn render_with_background(
    document: &SceneDocument,
    gpu: Arc<GpuContext>,
    background: Option<[u8; 4]>,
) -> Vec<u8> {
    let assets = trd_placement::document_assets(document).unwrap();
    let mut renderer =
        Renderer::with_assets(gpu, trd_core::TEXTURE_TARGET_FORMAT, &assets).unwrap();
    if document.meshes().is_empty() {
        renderer
            .set_mesh_aabb_color(0, trd_core::Mesh::REFERENCE_CUBE_COLOR)
            .unwrap();
    }
    let target = renderer.create_texture_target(WIDTH, HEIGHT).unwrap();
    let env = image::open(root().join("assets/envmap/uffizi-large.hdr"))
        .unwrap()
        .to_rgba32f();
    renderer.set_env_map(trd_core::EnvMapData::from_rgba32f(
        env.width(),
        env.height(),
        env.into_raw(),
        2048,
    ));
    renderer.set_tonemap_operator(trd_core::MeshTarget::All, trd_core::Tonemap::Aces);
    if let Some(color) = background {
        renderer.update_frame_texture_rgba(&color.repeat((WIDTH * HEIGHT) as usize), WIDTH, HEIGHT);
    }
    let options = RenderOptions {
        mode: RenderMode::Shaded,
        ..Default::default()
    };
    let (camera, scene) = trd_placement::document_scene(
        document,
        &document.frame(0).unwrap(),
        Viewport {
            width: WIDTH,
            height: HEIGHT,
        },
        &options,
        background.map(|_| trd_core::FrameFit::Stretch),
    )
    .unwrap();
    renderer.draw_layers(&[SceneLayer::new(camera, &scene)], &target);
    pollster::block_on(renderer.read_pixels(&target)).unwrap()
}

fn check(name: &str, image: &[u8]) {
    let path = golden(name);
    let update = std::env::var_os("TRD_UPDATE_GOLDENS").is_some();
    if update {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    }
    golden_image::compare_or_update(image, WIDTH, HEIGHT, &path, update).unwrap_or_else(|error| {
        let output = root().join("output/actual");
        std::fs::create_dir_all(&output).unwrap();
        image::RgbaImage::from_raw(WIDTH, HEIGHT, image.to_vec())
            .unwrap()
            .save(output.join(format!("{name}.actual.png")))
            .unwrap();
        panic!("{error}");
    });
}

fn edit(scale: f32, x: f32, yaw: f32) -> Matrix4 {
    let (sin, cos) = yaw.sin_cos();
    Matrix4::from_cols_array(&[
        scale * cos,
        0.0,
        -scale * sin,
        0.0,
        0.0,
        scale,
        0.0,
        0.0,
        scale * sin,
        0.0,
        scale * cos,
        0.0,
        x,
        0.15,
        0.0,
        1.0,
    ])
}

#[test]
#[ignore = "requires a real GPU"]
fn golden_params_reference_quad_axes_cube() {
    let _serial = GPU_SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let document = source(&[quad(160.0, 72.0)], &[], false);
    check("params_reference", &render(&document, gpu()));
}

#[test]
#[ignore = "requires a real GPU"]
fn golden_params_single_glb_edit_roundtrip() {
    let _serial = GPU_SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let gpu = gpu();
    let mesh = fixture_meshes().remove(0);
    let expected = edit(0.45, 0.2, 0.3);
    let mut final_image = None;
    for include_model in [false, true] {
        let mut document = source(
            &[quad(160.0, 144.0)],
            std::slice::from_ref(&mesh),
            include_model,
        );
        let mut config = document.render_config();
        config.shadow.enable = false;
        document.set_render_config(config).unwrap();
        let before = render(&document, gpu.clone());
        document
            .apply_model_edits(&[ModelEdit {
                row: 0,
                object: 0,
                model: expected,
            }])
            .unwrap();
        let edited = render(&document, gpu.clone());
        assert_ne!(before, edited, "the edit must visibly change the model");
        let reopened = SceneDocument::read(&document.write().unwrap()).unwrap();
        assert_eq!(reopened.frame(0).unwrap().objects[0].model, expected);
        assert_eq!(reopened.schema().metadata(), document.schema().metadata());
        assert_eq!(
            reopened.batches()[0]
                .column_by_name("upstream_note")
                .unwrap()
                .to_data(),
            document.batches()[0]
                .column_by_name("upstream_note")
                .unwrap()
                .to_data(),
        );
        assert_eq!(reopened.meshes()[0].bytes(), mesh.bytes());
        assert_eq!(
            edited,
            render(&reopened, gpu.clone()),
            "round-trip pixels changed"
        );
        final_image = Some(edited);
    }
    check("params_single_roundtrip", &final_image.unwrap());
}

#[test]
#[ignore = "requires a real GPU"]
fn golden_params_multiple_glb_bindings() {
    let _serial = GPU_SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let gpu = gpu();
    let meshes = fixture_meshes();
    let mut document = source(&[quad(92.0, 100.0), quad(228.0, 100.0)], &meshes, true);
    let mut config = document.render_config();
    config.shadow.enable = false;
    document.set_render_config(config).unwrap();
    document
        .apply_model_edits(&[
            ModelEdit {
                row: 0,
                object: 0,
                model: edit(0.35, -0.1, 0.3),
            },
            ModelEdit {
                row: 0,
                object: 1,
                model: edit(0.3, 0.1, -0.5),
            },
        ])
        .unwrap();
    let image = render(&document, gpu.clone());
    let reversed = SceneDocument::from_batches(
        document.schema().clone(),
        document.batches().to_vec(),
        meshes.into_iter().rev().collect(),
    )
    .unwrap();
    assert_eq!(
        image,
        render(&reversed, gpu),
        "mesh row order changed UUID bindings"
    );
    check("params_multiple", &image);
}

#[test]
#[ignore = "requires a real GPU"]
fn golden_params_blob_shadows_toggle_and_replay() {
    let _serial = GPU_SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let gpu = gpu();
    let meshes = fixture_meshes();
    let mut document = source(&[quad(92.0, 100.0), quad(228.0, 100.0)], &meshes, true);
    assert!(!document
        .schema()
        .metadata()
        .contains_key(trd_core::RENDER_CONFIG_KEY));
    assert_eq!(document.render_config(), trd_core::RenderConfig::default());
    document
        .apply_model_edits(&[
            ModelEdit {
                row: 0,
                object: 0,
                model: edit(0.5, -0.1, 0.3),
            },
            ModelEdit {
                row: 0,
                object: 1,
                model: edit(0.4, 0.1, -0.5),
            },
        ])
        .unwrap();
    let enabled = render_with_background(&document, gpu.clone(), Some([190, 190, 190, 255]));
    let original_frames = document.frames().unwrap();
    let mut config = document.render_config();
    config.shadow.enable = false;
    document.set_render_config(config).unwrap();
    let disabled = render_with_background(&document, gpu.clone(), Some([190, 190, 190, 255]));
    let darker = enabled
        .chunks_exact(4)
        .zip(disabled.chunks_exact(4))
        .filter(|(on, off)| on[0].saturating_add(5) < off[0])
        .count();
    assert!(
        darker > 100,
        "enabled shadows must visibly darken the background: {darker} pixels"
    );
    let reopened = SceneDocument::read(&document.write().unwrap()).unwrap();
    assert_eq!(reopened.frames().unwrap(), original_frames);
    assert_eq!(reopened.meshes(), document.meshes());
    assert_eq!(
        disabled,
        render_with_background(&reopened, gpu.clone(), Some([190, 190, 190, 255]))
    );
    config.shadow.enable = true;
    document.set_render_config(config).unwrap();
    let reopened = SceneDocument::read(&document.write().unwrap()).unwrap();
    assert_eq!(
        enabled,
        render_with_background(&reopened, gpu.clone(), Some([190, 190, 190, 255]))
    );
    check("params_blob_shadows", &enabled);

    let mut reference = source(&[quad(160.0, 100.0)], &[], false);
    let before = render_with_background(&reference, gpu.clone(), Some([190, 190, 190, 255]));
    config.shadow.enable = false;
    reference.set_render_config(config).unwrap();
    assert_eq!(
        before,
        render_with_background(&reference, gpu, Some([190, 190, 190, 255]))
    );
}

#[test]
#[ignore = "requires a real GPU"]
fn quad_guides_stay_below_meshes_and_their_gizmos() {
    let _serial = GPU_SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let gpu = gpu();
    let meshes = fixture_meshes();
    let document = source(&[quad(110.0, 160.0), quad(215.0, 160.0)], &meshes, true);
    let assets = trd_placement::document_assets(&document).unwrap();
    for samples in [1, 4] {
        let mut renderer = Renderer::with_assets_sample_count(
            gpu.clone(),
            trd_core::TEXTURE_TARGET_FORMAT,
            &assets,
            samples,
        )
        .unwrap();
        let target = renderer.create_texture_target(WIDTH, HEIGHT).unwrap();
        renderer.update_frame_texture_rgba(&[255; 4], 1, 1);
        let white = trd_core::Scene::new().with_background(trd_core::Background {
            frame: Some(trd_core::FrameFit::Stretch),
            ..Default::default()
        });
        for mode in [
            RenderMode::Filled,
            RenderMode::Shaded,
            RenderMode::Wireframe,
        ] {
            let (camera, [background, foreground]) = trd_placement::document_scene_with_overlays(
                &document,
                &document.frame(0).unwrap(),
                target.viewport(),
                &RenderOptions {
                    mode,
                    show_aabb: true,
                    show_local_axes: true,
                    selected: Some(1),
                    ..Default::default()
                },
                None,
                trd_placement::PlacementOverlays {
                    quad: true,
                    axes: true,
                    grid: true,
                    selected: Some(0),
                    hovered: Some(1),
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(background.objects().iter().all(|object| !matches!(
                object.primitive(),
                trd_core::Primitive::Mesh { .. } | trd_core::Primitive::AabbBox { .. }
            )));
            assert!(foreground.objects().iter().all(|object| !matches!(
                object.primitive(),
                trd_core::Primitive::QuadFill | trd_core::Primitive::QuadOutline { .. }
            )));
            let mut draw = |scenes: &[&trd_core::Scene]| {
                let layers = scenes
                    .iter()
                    .map(|scene| SceneLayer::new(camera, scene))
                    .collect::<Vec<_>>();
                pollster::block_on(renderer.render_layers(&layers, &target)).unwrap()
            };
            let black_base = draw(&[&foreground]);
            let white_base = draw(&[&white, &foreground]);
            let composited = draw(&[&background, &foreground]);
            let mut merged = background.clone();
            merged.extend(foreground.objects().iter().copied());
            let old_order = draw(&[&merged]);

            // Agreement over black/white identifies opaque foreground samples,
            // including fully covered AABB/axis pixels, without a geometry mask.
            let mut opaque = 0;
            let mut exposed_regression = 0;
            for (((black, white), actual), old) in black_base
                .chunks_exact(4)
                .zip(white_base.chunks_exact(4))
                .zip(composited.chunks_exact(4))
                .zip(old_order.chunks_exact(4))
            {
                if black == white {
                    opaque += 1;
                    assert_eq!(actual, black, "{mode:?}, {samples}x: guide covers content");
                    exposed_regression += usize::from(old != black);
                }
            }
            assert!(opaque > 50, "{mode:?}, {samples}x: missing opaque content");
            assert!(
                exposed_regression > 0,
                "{mode:?}, {samples}x: fixture must expose the old single-pass ordering"
            );
        }
    }
}
