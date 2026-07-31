//! GPU-gated end-to-end test for the trd-gui in-process render backend (#97).
//!
//! Renders the built-in scene through [`InProcRenderer`] — the exact path the
//! interactive window uses — and asserts the backend produces a correctly sized,
//! non-blank frame that responds to interaction (orbiting the camera changes the
//! image). It needs a GPU adapter, so it is `#[ignore]`d and run locally (nixGL
//! on Linux, the MSVC toolchain on Windows) per the repo's dual-platform policy;
//! CI skips it.

use trd_gui::render_backend::{InProcRenderer, SceneRenderer};
use trd_gui::scene::SceneState;

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

fn backend(width: u32, height: u32) -> InProcRenderer {
    let mesh = trd_core::Mesh::from_obj(CUBE_OBJ).expect("cube parses");
    InProcRenderer::new(&[mesh], None, None, width, height).expect("GPU backend builds")
}

#[test]
#[ignore = "requires a GPU adapter; run locally"]
fn renders_a_nonblank_frame_of_expected_size() {
    let (w, h) = (128, 128);
    let mut renderer = backend(w, h);
    let image = renderer
        .render(&SceneState::default())
        .expect("render succeeds");

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
    let mut renderer =
        InProcRenderer::new(&[mesh], Some(&red), None, w, h).expect("GPU backend builds");

    let state = SceneState {
        mode: trd_core::RenderMode::Textured,
        ..Default::default()
    };
    let image = renderer.render(&state).expect("render succeeds");

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

    let base = renderer
        .render(&SceneState::default())
        .expect("render succeeds")
        .rgba;

    let mut orbited = SceneState::default();
    orbited.camera.orbit(1.0, 0.2);
    let moved = renderer.render(&orbited).expect("render succeeds").rgba;

    assert_ne!(base, moved, "orbiting the camera did not change the frame");
}
