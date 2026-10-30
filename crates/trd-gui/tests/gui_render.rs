//! GPU-gated end-to-end test for the trd-gui in-process render backend (#97).
//!
//! Renders the built-in scene through [`GuiRenderer`] — the exact path the
//! interactive window uses — and asserts the backend produces a correctly sized,
//! non-blank frame that responds to interaction (orbiting the camera changes the
//! image). It needs a GPU adapter, so it is `#[ignore]`d and run locally (nixGL
//! on Linux, the MSVC toolchain on Windows) per the repo's dual-platform policy;
//! CI skips it.

use trd_core::Overlays;
use trd_gui::renderer::GuiRenderer;
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

fn backend(width: u32, height: u32) -> GuiRenderer {
    let mesh = trd_core::Mesh::from_obj(CUBE_OBJ).expect("cube parses");
    pollster::block_on(GuiRenderer::new(&[mesh], &[], &[], None, width, height))
        .expect("GPU backend builds")
}

#[test]
#[ignore = "requires a GPU adapter; run locally"]
fn renders_a_nonblank_frame_of_expected_size() {
    let (w, h) = (128, 128);
    let mut renderer = backend(w, h);
    let image =
        pollster::block_on(renderer.render(&SceneState::default())).expect("render succeeds");

    assert_eq!(image.width, w);
    assert_eq!(image.height, h);
    assert_eq!(image.rgba.len(), (w * h * 4) as usize);

    // The colored cube must cover some pixels: not every pixel is the clear
    // color, so the frame carries actual geometry.
    let first = &image.rgba[0..4];
    assert!(
        image.rgba.as_chunks::<4>().0.iter().any(|px| px != first),
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

    let state = SceneState {
        modes: vec![trd_core::RenderMode::Textured],
        ..Default::default()
    };
    let image = pollster::block_on(renderer.render(&state)).expect("render succeeds");

    let red_px = image
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .any(|p| p[0] > 150 && p[1] < 100 && p[2] < 100);
    assert!(red_px, "textured cube shows no red from the bound texture");
}

#[test]
#[ignore = "requires a GPU adapter; run locally"]
fn orbiting_the_camera_changes_the_image() {
    let (w, h) = (128, 128);
    let mut renderer = backend(w, h);

    let base = pollster::block_on(renderer.render(&SceneState::default()))
        .expect("render succeeds")
        .rgba;

    let mut orbited = SceneState::default();
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

    let plain = SceneState::default();
    let base = pollster::block_on(renderer.render(&plain)).expect("render without overlays");

    let with_axes = SceneState {
        overlays: Overlays {
            axes: true,
            ..Default::default()
        },
        ..SceneState::default()
    };
    let axes = pollster::block_on(renderer.render(&with_axes)).expect("render with world axes");

    assert_eq!(base.rgba.len(), axes.rgba.len());
    assert_ne!(
        base.rgba, axes.rgba,
        "enabling the world-axes overlay must change the image"
    );

    let with_aabb = SceneState {
        overlays: Overlays {
            aabb: true,
            ..Default::default()
        },
        ..SceneState::default()
    };
    let aabb = pollster::block_on(renderer.render(&with_aabb)).expect("render with AABB");
    assert_ne!(
        base.rgba, aabb.rgba,
        "enabling the bounding-box overlay must change the image"
    );
}

/// Two unit cubes side by side (`-x` and `+x`) as one mesh: 36 indices each.
fn two_cube_mesh() -> trd_core::Mesh {
    let shifted = |dx: f32, base: usize| {
        let mut obj = String::new();
        for line in CUBE_OBJ.lines() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            match fields.first() {
                Some(&"v") => {
                    let x: f32 = fields[1].parse().unwrap();
                    obj += &format!("v {} {} {}\n", x + dx, fields[2], fields[3]);
                }
                Some(&"f") => {
                    let ids: Vec<String> = fields[1..]
                        .iter()
                        .map(|id| (id.parse::<usize>().unwrap() + base).to_string())
                        .collect();
                    obj += &format!("f {}\n", ids.join(" "));
                }
                _ => {}
            }
        }
        obj
    };
    let mesh = trd_core::Mesh::from_obj(&(shifted(-0.5, 0) + &shifted(0.5, 8))).unwrap();
    assert_eq!(mesh.indices.len(), 72);
    mesh
}

fn flat(base_color: [f32; 3]) -> trd_core::DisneyMaterial {
    trd_core::DisneyMaterial {
        base_color,
        metallic: 0.0,
        roughness: 0.6,
        ..Default::default()
    }
}

/// Centroid x of the pixels `matches` accepts, and how many there were.
fn centroid_x(
    image: &trd_gui::renderer::ImageRgba,
    matches: impl Fn(&[u8; 4]) -> bool,
) -> (f32, usize) {
    let (mut sum, mut count) = (0.0, 0);
    for (index, pixel) in image.rgba.as_chunks::<4>().0.iter().enumerate() {
        if matches(pixel) {
            sum += (index as u32 % image.width) as f32;
            count += 1;
        }
    }
    (sum / count.max(1) as f32, count)
}

#[test]
#[ignore = "requires a GPU adapter; run locally"]
fn a_multi_material_object_draws_both_parts_and_picks_and_moves_as_one() {
    let (w, h) = (192, 128);
    let mut renderer =
        pollster::block_on(GuiRenderer::new(&[two_cube_mesh()], &[], &[], None, w, h))
            .expect("GPU backend builds");
    let whole = |range: std::ops::Range<u32>, color| trd_core::MaterialPart {
        indices: range,
        material: flat(color),
        base_color_texture: None,
        metallic_roughness_texture: None,
        normal_texture: None,
    };
    renderer
        .set_mesh_parts(
            0,
            &[
                whole(0..36, [0.9, 0.05, 0.05]),
                whole(36..72, [0.05, 0.05, 0.9]),
            ],
        )
        .unwrap();
    assert_eq!(renderer.mesh_part_count(0), 2);

    let mut state = SceneState::default();
    state.modes[0] = trd_core::RenderMode::Shaded;
    // A multi-part object ignores the mesh-wide material: each part keeps its own.
    state.materials[0] = flat([0.0, 1.0, 0.0]);
    let channels = |p: &[u8; 4]| [p[0], p[1], p[2]].map(u16::from);
    let red = |p: &[u8; 4]| {
        let [r, g, b] = channels(p);
        r > g + 40 && r > b + 40
    };
    let blue = |p: &[u8; 4]| {
        let [r, g, b] = channels(p);
        b > r + 40 && b > g + 40
    };
    let green = |p: &[u8; 4]| {
        let [r, g, b] = channels(p);
        g > r + 40 && g > b + 40
    };

    let before = pollster::block_on(renderer.render(&state)).unwrap();
    let (red_x, red_count) = centroid_x(&before, red);
    let (blue_x, blue_count) = centroid_x(&before, blue);
    assert!(
        red_count > 100 && blue_count > 100,
        "red {red_count} px, blue {blue_count} px"
    );
    assert!(red_x < blue_x, "the red part is the -x cube");
    assert_eq!(
        centroid_x(&before, green).1,
        0,
        "the mesh-wide material leaked into a part"
    );

    for x in [red_x, blue_x] {
        let picked = pollster::block_on(renderer.pick(&state, x as u32, h / 2));
        assert_eq!(picked, Some(0), "either part picks the one object");
    }

    state.objects[0].translation[0] += 0.3;
    let after = pollster::block_on(renderer.render(&state)).unwrap();
    let shift_red = centroid_x(&after, red).0 - red_x;
    let shift_blue = centroid_x(&after, blue).0 - blue_x;
    assert!(shift_red > 5.0, "the object did not move ({shift_red} px)");
    assert!(
        (shift_red - shift_blue).abs() < 2.0,
        "parts moved apart: {shift_red} vs {shift_blue}"
    );
}
