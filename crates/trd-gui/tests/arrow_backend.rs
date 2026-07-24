//! GPU-gated test for the Arrow round-trip backend (#97, Slice 3b).
//!
//! The Arrow backend authors the scene as a `[mesh][params]` Arrow stream, runs
//! it through `trd-core`'s `run_stream`, and decodes the image back — the same
//! pipeline the headless CLI uses. Because both backends ultimately drive the
//! same `BatchRenderer` with the same decoded mesh / camera / draws, their output
//! must be **pixel-identical**. Needs a GPU adapter, so it is `#[ignore]`d and run
//! locally (MSVC on Windows, nixGL on Linux); CI skips it.

use trd_gui::render_backend::{ArrowRoundTripRenderer, InProcRenderer, SceneRenderer};
use trd_gui::scene::SceneState;

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

fn cube() -> trd_core::Mesh {
    trd_core::Mesh::from_obj(CUBE_OBJ).expect("cube parses")
}

#[test]
#[ignore = "requires a GPU adapter; run locally"]
fn arrow_backend_defers_and_renders_a_nonblank_frame() {
    let (w, h) = (128, 128);
    let mut renderer = ArrowRoundTripRenderer::new(&[cube()], w, h).expect("arrow backend builds");
    assert!(renderer.defer_expensive());

    let image = renderer
        .render(&SceneState::default())
        .expect("round-trip render succeeds");
    assert_eq!((image.width, image.height), (w, h));
    assert_eq!(image.rgba.len(), (w * h * 4) as usize);

    let first = &image.rgba[0..4];
    assert!(
        image.rgba.chunks_exact(4).any(|px| px != first),
        "arrow round-trip frame is a flat color — nothing was drawn"
    );
}

#[test]
#[ignore = "requires a GPU adapter; run locally"]
fn arrow_backend_matches_inproc_pixel_for_pixel() {
    let (w, h) = (128, 128);
    let state = SceneState::default();

    let mut inproc = InProcRenderer::new(&[cube()], None, w, h).expect("inproc builds");
    let mut arrow = ArrowRoundTripRenderer::new(&[cube()], w, h).expect("arrow builds");

    let a = inproc.render(&state).expect("inproc render").rgba;
    let b = arrow.render(&state).expect("arrow render").rgba;

    assert_eq!(
        a, b,
        "the Arrow round-trip must be pixel-identical to the in-process backend"
    );
}
