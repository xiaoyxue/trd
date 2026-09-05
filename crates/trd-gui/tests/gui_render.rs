//! GPU-gated end-to-end test for the trd-gui in-process render backend (#97).
//!
//! Renders the built-in scene through [`GuiRenderer`] — the exact path the
//! interactive window uses — and asserts the backend produces a correctly sized,
//! non-blank frame that responds to interaction (orbiting the camera changes the
//! image). It needs a GPU adapter, so it is `#[ignore]`d and run locally (nixGL
//! on Linux, the MSVC toolchain on Windows) per the repo's dual-platform policy;
//! CI skips it.

use trd_gui::renderer::GuiRenderer;
use trd_gui::scene::{SceneSeed, SceneState};

/// A small origin-centered cube (matches the shape of the CLI's built-in mesh).
const CUBE_OBJ: &str = "\
v -0.5 -0.5 -0.5 0.1 0.1 0.9
v  0.5 -0.5 -0.5 0.9 0.1 0.1
v  0.5  0.5 -0.5 0.9 0.9 0.1
v -0.5  0.5 -0.5 0.1 0.9 0.1
v -0.5 -0.5  0.5 0.1 0.9 0.9
v  0.5 -0.5  0.5 0.9 0.1 0.9
v  0.5  0.5  0.5 0.9 0.9 0.9
v -0.5  0.5  0.5 0.2 0.2 0.2
f 1 2 3 4
f 5 6 7 8
f 1 5 8 4
f 2 6 7 3
f 4 8 7 3
f 1 5 6 2
";

fn backend(width: u32, height: u32) -> GuiRenderer {
    let mesh = trd_core::Mesh::from_obj(CUBE_OBJ).expect("cube parses");
    pollster::block_on(GuiRenderer::new(&[mesh], &[], &[], None, width, height))
        .expect("GPU backend builds")
}

fn scene(renderer: &GuiRenderer) -> SceneState {
    SceneState::seeded(
        renderer.mesh_table(),
        SceneSeed {
            materials: vec![trd_core::DisneyMaterial::default(); renderer.mesh_table().len()],
            mode: trd_core::RenderMode::Filled,
            image_based_lighting: trd_core::ImageBasedLighting::default(),
            tone_mapping: trd_core::ToneMapping::default(),
            lighting: trd_core::Lighting::default(),
            environment_available: false,
            show_environment_background: false,
        },
    )
    .expect("scene uses the renderer's registrations")
}

#[test]
#[ignore = "requires a GPU adapter; run locally"]
fn renders_a_nonblank_frame_of_expected_size() {
    let (w, h) = (128, 128);
    let mut renderer = backend(w, h);
    let state = scene(&renderer);
    let image = pollster::block_on(renderer.render(&state)).expect("render succeeds");

    assert_eq!(image.width, w);
    assert_eq!(image.height, h);
    assert_eq!(image.rgba.len(), (w * h * 4) as usize);

    // The colored cube must cover some pixels: not every pixel is the clear
    // color, so the frame carries actual geometry.
    let first = &image.rgba[0..4];
    assert!(
        image.rgba.chunks_exact(4).any(|px| px != first),
        "frame is a flat color — nothing was drawn"
    );
}

#[test]
#[ignore = "requires a GPU adapter; run locally"]
fn textured_mode_samples_the_bound_texture() {
    let (w, h) = (128, 128);
    let mesh = trd_core::Mesh::from_obj(CUBE_OBJ).expect("cube parses");
    // A uniform 2×2 red albedo; the untextured cube (uv = 0,0) samples red.
    let red = trd_core::ImageTexture::from_rgba(
        2,
        2,
        vec![
            255, 0, 0, 255, 255, 0, 0, 255, //
            255, 0, 0, 255, 255, 0, 0, 255,
        ],
    )
    .expect("texture builds");
    let mut renderer = pollster::block_on(GuiRenderer::new(
        &[mesh],
        &[Some(&red as &dyn trd_core::Texture)],
        &[],
        None,
        w,
        h,
    ))
    .expect("GPU backend builds");

    let mut state = scene(&renderer);
    state.objects[0].mode = trd_core::RenderMode::Textured;
    let image = pollster::block_on(renderer.render(&state)).expect("render succeeds");

    let red_px = image
        .rgba
        .chunks_exact(4)
        .any(|p| p[0] > 150 && p[1] < 100 && p[2] < 100);
    assert!(red_px, "textured cube shows no red from the bound texture");
}

#[test]
#[ignore = "requires a GPU adapter; run locally"]
fn orbiting_the_camera_changes_the_image() {
    let (w, h) = (128, 128);
    let mut renderer = backend(w, h);
    let state = scene(&renderer);

    let base = pollster::block_on(renderer.render(&state))
        .expect("render succeeds")
        .rgba;

    let mut orbited = state;
    orbited.camera.orbit(1.0, 0.2);
    let moved = pollster::block_on(renderer.render(&orbited))
        .expect("render succeeds")
        .rgba;

    assert_ne!(base, moved, "orbiting the camera did not change the frame");
}

/// Overlay toggles must reach the rendered image.
///
/// The deleted `arrow_backend` test compared two GUI backends pixel-for-pixel;
/// with one backend left there is nothing to compare against, so what is worth
/// pinning instead is that the scene the GUI hands to `trd-core` still honours
/// `SceneState`'s overlay flags — the path that changed when the renderer's nine
/// retained overlay fields were replaced by `scene_with_overlays` (#180).
#[test]
#[ignore = "requires a GPU adapter"]
fn overlay_toggles_change_the_rendered_image() {
    let (w, h) = (96, 96);
    let mut renderer = pollster::block_on(GuiRenderer::new(
        &[trd_core::Mesh::from_obj(CUBE_OBJ).expect("cube parses")],
        &[],
        &[],
        None,
        w,
        h,
    ))
    .expect("the GUI renderer builds");

    let plain = scene(&renderer);
    let base = pollster::block_on(renderer.render(&plain)).expect("render without overlays");

    let with_axes = SceneState {
        show_axes: true,
        ..plain.clone()
    };
    let axes = pollster::block_on(renderer.render(&with_axes)).expect("render with world axes");

    assert_eq!(base.rgba.len(), axes.rgba.len());
    assert_ne!(
        base.rgba, axes.rgba,
        "enabling the world-axes overlay must change the image"
    );

    let with_aabb = SceneState {
        show_aabb: true,
        ..plain
    };
    let aabb = pollster::block_on(renderer.render(&with_aabb)).expect("render with AABB");
    assert_ne!(
        base.rgba, aabb.rgba,
        "enabling the bounding-box overlay must change the image"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn deleting_adding_and_picking_keep_gui_rows_bound_to_live_meshes() {
    let mesh = trd_core::Mesh::from_obj(CUBE_OBJ).expect("cube");
    let mut renderer = pollster::block_on(GuiRenderer::new(
        &[mesh.clone(), mesh.clone(), mesh.clone()],
        &[],
        &[],
        None,
        256,
        128,
    ))
    .expect("renderer");
    let mut state = scene(&renderer);
    state.camera.distance = 10.0;
    let initial: Vec<_> = state.objects.iter().map(|object| object.mesh).collect();
    state.selected = Some(1);
    let removed = state.remove_selected_object().expect("selected row");
    assert_eq!(removed, initial[1]);
    assert!(!state.uses_mesh(removed));
    renderer.remove_mesh(removed).expect("resident mesh");

    let asset = trd_core::GltfAsset {
        mesh,
        material: trd_core::DisneyMaterial::default(),
        base_color_texture: None,
        metallic_roughness_texture: None,
        normal_texture: None,
    };
    let added = renderer.add_model(&asset).expect("upload replacement");
    assert!(
        !initial.contains(&added),
        "a recycled GPU slot cannot recycle identity"
    );
    state.add_object(
        added,
        asset.material,
        trd_core::RenderMode::Filled,
        trd_core::ToneMapping::default(),
    );
    assert_eq!(state.objects[1].mesh, initial[2]);
    state.objects[1].mode = trd_core::RenderMode::Shaded;
    state.objects[1].appearance.material.base_color = [0.0, 0.0, 1.0];
    state.selected = Some(1);
    let image = pollster::block_on(renderer.render(&state)).expect("survivors render");
    let center = &image.rgba[((64 * 256 + 128) * 4)..][..4];
    assert!(
        center[2] > center[0] && center[2] > center[1],
        "the survivor keeps its blue appearance"
    );
    assert_eq!(
        pollster::block_on(renderer.pick(&state, 128, 64)).expect("pick survives deletion"),
        Some(1),
        "the centered survivor is row 1, not its opaque resource identity"
    );
    assert!(
        renderer.remove_mesh(removed).is_err(),
        "stale removal is surfaced"
    );
    let mut stale = state.clone();
    stale.objects[1].mesh = removed;
    assert!(pollster::block_on(renderer.render(&stale)).is_err());
    assert!(pollster::block_on(renderer.pick(&stale, 128, 64)).is_err());
}
