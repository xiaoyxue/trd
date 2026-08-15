//! End-to-end golden / snapshot render test (issue #88).
//!
//! Feeds a *fixed, committed* Arrow IPC input stream through the **entire real
//! pipeline** — [`trd_core::run_stream`] (Arrow decode → `build_scene` → GPU
//! render → readback → RGBA output stream) — and asserts the rendered frames
//! match committed golden PNGs within a small tolerance. It is the regression
//! safety net for the #82 refactor: unlike the property-style GPU tests
//! ("center pixel lit", "rotation changes pixels"), a pixel-diff golden catches
//! decoder divergence, GPU `Uniform` byte drift, and camera-math regressions.
//!
//! The fixtures are the two-stage cornellbox *placement* demo (#77), reduced to
//! a few frames at a small resolution by `scripts/golden_fixtures.py`. The mesh,
//! texture, inline background frames, and params are embedded in one Arrow byte
//! stream. Params `frame_id` values select the background resource, so the
//! fixture is self-contained while still compositing the scene over the
//! cornellbox background:
//!
//! * `stage1.arrow` — the reconstructed placement quad (wireframe) + local axes;
//! * `stage2.arrow` — a textured bunny anchored on that quad, with AABB + local
//!   axes + the wireframe quad overlay.
//!
//! The `stage1`/`stage2` fixtures are each rendered **twice** — at 4× MSAA (the
//! default anti-aliased mesh pass) and with MSAA disabled ([`trd_core::Msaa::Off`],
//! single-sample) — each pinned to its own goldens (`stageN_*` vs `stageN_noaa_*`),
//! so both the multisampled + resolve path and the raw single-sample path are
//! covered.
//!
//! The **`stage2` mesh is additionally rendered through the Disney PBR path**
//! ([`trd_core::RenderMode::Shaded`]) with a deterministic synthetic HDR environment
//! probe (no external `.hdr` decode), once per tone-map operator — pinning both
//! the historical [`trd_core::Tonemap::Reinhard`] curve and the [`Tonemap::Aces`]
//! filmic curve (#116). These `stage2_pbr_{reinhard,aces}_*` goldens are the
//! regression net for the physically-based shading + envmap-reflection + tone-map
//! stages of `disney.wgsl`.
//!
//! GPU-gated (`#[ignore]`, like the other render tests): run on a GPU box with
//! ```text
//! cargo test -p trd-core --test golden_render -- --ignored
//! ```
//! On non-NixOS Linux wrap it in nixGL (see AGENTS.md / README "Tests").
//!
//! Regenerate the golden PNGs (after changing the fixtures or an *intended*
//! visual change) by setting `TRD_UPDATE_GOLDENS=1`:
//! ```text
//! TRD_UPDATE_GOLDENS=1 cargo test -p trd-core --test golden_render -- --ignored
//! ```
//!
//! On a mismatch (or when `TRD_DUMP_ACTUAL=1`) the *actual* rendered frames and
//! an 8x-amplified per-pixel diff are written to the git-ignored `output/actual/`
//! so the render can be eyeballed without re-running — handy for cross-GPU
//! validation (e.g. on Windows):
//! ```text
//! TRD_DUMP_ACTUAL=1 cargo test -p trd-core --test golden_render -- --ignored
//! ```

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use arrow::array::{Array, FixedSizeListArray, UInt8Array};
use arrow::ipc::reader::StreamReader;
use trd_core::{
    run_stream, Camera, DisneyMaterial, Draw, DrawSelection, EnvMapData, EnvironmentBackground,
    EnvironmentLight, ImageBasedLighting, Lighting, Matrix4, Mesh, Msaa, PbrConfig, Point3,
    RenderMode, RenderOptions, Renderer, Scene, SceneLayer, ToneMapping, Tonemap, Vector3, Vertex,
    Viewport,
};

/// Golden render resolution (16:9; the fixtures' CV `k` is rescaled to match).
const WIDTH: u32 = 320;
const HEIGHT: u32 = 180;

/// A channel absolute difference `<= CHANNEL_EPS` is treated as equal, absorbing
/// minor cross-driver rasterization variance.
const CHANNEL_EPS: u8 = 16;
/// At most this fraction of pixels may differ beyond `CHANNEL_EPS`.
const MAX_DIFF_FRACTION: f64 = 0.02;

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn update_goldens() -> bool {
    std::env::var_os("TRD_UPDATE_GOLDENS").is_some()
}

/// Git-ignored `output/actual/` (relative to the repo root) where the *actual*
/// rendered frames — plus an 8x-amplified per-pixel diff vs the golden — are
/// written on a mismatch, or whenever `TRD_DUMP_ACTUAL` is set. Lets a failing
/// render be eyeballed without re-running (e.g. cross-GPU validation on Windows).
fn dump_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../output/actual")
}

fn should_dump() -> bool {
    std::env::var_os("TRD_DUMP_ACTUAL").is_some()
}

/// Best-effort dump of one actual frame + its amplified diff into `output/actual/`
/// (failures are logged, never fatal — this is a debugging aid).
fn dump_actual(name: &str, index: usize, actual: &[u8], golden: &Path) {
    let dir = dump_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("dump: create {}: {e}", dir.display());
        return;
    }
    if let Some(img) = image::RgbaImage::from_raw(WIDTH, HEIGHT, actual.to_vec()) {
        let _ = img.save(dir.join(format!("{name}_frame_{index}.actual.png")));
    }
    if let Ok(golden_img) = image::open(golden) {
        let golden_img = golden_img.to_rgba8().into_raw();
        let mut diff = vec![0u8; actual.len()];
        for p in 0..(WIDTH * HEIGHT) as usize {
            for c in 0..3 {
                diff[p * 4 + c] = actual[p * 4 + c]
                    .abs_diff(golden_img[p * 4 + c])
                    .saturating_mul(8);
            }
            diff[p * 4 + 3] = 255;
        }
        if let Some(img) = image::RgbaImage::from_raw(WIDTH, HEIGHT, diff) {
            let _ = img.save(dir.join(format!("{name}_frame_{index}.diff8x.png")));
        }
    }
}

/// A small, deterministic synthetic HDR environment probe (equirectangular,
/// linear-RGB `height`×`width`×4 f32) for the PBR golden cases. Building it in
/// code keeps the fixture self-contained — no `.hdr` file decode, no extra
/// binary asset — while still exercising the real envmap-reflection path: a
/// vertical sky→ground gradient with a bright "sun" lobe, so metallic surfaces
/// pick up a non-trivial, reproducible reflection (and HDR highlights that the
/// tone-map curve must roll off).
fn synthetic_env() -> EnvMapData {
    const W: u32 = 64;
    const H: u32 = 32;
    let sky = [0.30f32, 0.45, 0.85]; // linear-RGB zenith
    let ground = [0.35f32, 0.28, 0.20]; // linear-RGB nadir
    let mut rgba = vec![0.0f32; (W * H * 4) as usize];
    for y in 0..H {
        let v = y as f32 / (H - 1) as f32; // 0 top .. 1 bottom
        for x in 0..W {
            let u = x as f32 / (W - 1) as f32;
            let mut c = [
                sky[0] * (1.0 - v) + ground[0] * v,
                sky[1] * (1.0 - v) + ground[1] * v,
                sky[2] * (1.0 - v) + ground[2] * v,
            ];
            // A bright warm sun lobe in the upper hemisphere (HDR: > 1.0).
            let du = u - 0.5;
            let dv = v - 0.18;
            let sun = (-(du * du + dv * dv) * 60.0).exp() * 6.0;
            c[0] += sun;
            c[1] += sun * 0.95;
            c[2] += sun * 0.82;
            let i = ((y * W + x) * 4) as usize;
            rgba[i] = c[0];
            rgba[i + 1] = c[1];
            rgba[i + 2] = c[2];
            rgba[i + 3] = 1.0;
        }
    }
    EnvMapData::from_rgba32f(W, H, rgba, 2048)
}

/// The shared Disney material for the PBR golden cases: a saturated, strongly
/// metallic green under a bright rig — the "deep-green metallic can" scenario
/// (#116) where per-channel Reinhard desaturates the highlights toward grey and
/// ACES retains the hue. Only the tone-map operator differs between the two
/// PBR goldens, isolating the tone-map stage.
fn pbr_material() -> DisneyMaterial {
    DisneyMaterial {
        base_color: [0.20, 0.85, 0.35],
        metallic: 0.9,
        roughness: 0.30,
        specular: 0.6,
        ..DisneyMaterial::default()
    }
}

/// [`RenderOptions`] for a PBR golden case: PBR mesh mode + the synthetic env
/// probe and the shared material (with `tonemap`), AABB + local-axes overlays
/// like `stage2`, 4× MSAA.
fn pbr_options(tonemap: Tonemap) -> RenderOptions {
    RenderOptions {
        mode: RenderMode::Shaded,
        show_aabb: true,
        show_axes: false,
        show_local_axes: true,
        show_local_grid: None,
        show_local_grid_mesh: None,
        show_world_grid: None,

        show_object_grid: None,

        selected: None,

        env_background: None,

        pbr: Some(PbrConfig {
            material: pbr_material(),
            lighting: Lighting {
                ambient: 0.05,
                ..Lighting::default()
            },
            ibl: ImageBasedLighting { intensity: 1.0 },
            tone_mapping: ToneMapping {
                operator: tonemap,
                exposure: 1.4,
            },
            env_map: Some(synthetic_env()),
        }),
        msaa: Msaa::X4,
    }
}

/// Process-wide lock serializing the GPU render across the parallel golden
/// `#[test]` threads (see [`render_fixture`]).
static GPU_SERIAL: Mutex<()> = Mutex::new(());

fn render_fixture(fixture: &str, options: RenderOptions) -> Vec<Vec<u8>> {
    let bytes = std::fs::read(golden_dir().join(fixture))
        .unwrap_or_else(|e| panic!("read fixture {fixture}: {e}"));

    let mut out = Vec::new();
    {
        // Serialize the GPU work across the (otherwise parallel) `#[test]` threads.
        // Each test builds its own wgpu `Instance`/`Device` and submits an
        // MSAA render + resolve + readback; concurrent multi-device MSAA
        // submissions intermittently deadlock the NVIDIA driver (the process
        // stays alive at 0% GPU). A process-wide lock makes this mandatory gate
        // reliable under the default `cargo test` runner, with no need for
        // `-- --test-threads=1`. Poison is ignored: a panicking test still
        // releases the GPU for the rest.
        let _serial = GPU_SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        run_stream(&bytes[..], &mut out, WIDTH, HEIGHT, options, None)
            .unwrap_or_else(|e| panic!("run_stream on {fixture}: {e:?}"));
    }

    let pixels = (WIDTH * HEIGHT) as usize;
    let reader = StreamReader::try_new(&out[..], None).expect("output arrow reader");
    let mut frames = Vec::new();
    for batch in reader {
        let batch = batch.expect("output arrow batch");
        let channels: Vec<&FixedSizeListArray> = ["r", "g", "b", "a"]
            .iter()
            .map(|name| {
                batch
                    .column_by_name(name)
                    .unwrap_or_else(|| panic!("output missing column {name}"))
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .expect("channel is FixedSizeList<u8>")
            })
            .collect();
        for row in 0..batch.num_rows() {
            let cells: Vec<UInt8Array> = channels
                .iter()
                .map(|c| {
                    c.value(row)
                        .as_any()
                        .downcast_ref::<UInt8Array>()
                        .expect("channel cell is UInt8")
                        .clone()
                })
                .collect();
            let mut rgba = vec![0u8; pixels * 4];
            for p in 0..pixels {
                rgba[p * 4] = cells[0].value(p);
                rgba[p * 4 + 1] = cells[1].value(p);
                rgba[p * 4 + 2] = cells[2].value(p);
                rgba[p * 4 + 3] = cells[3].value(p);
            }
            frames.push(rgba);
        }
    }
    frames
}

/// Compare `actual` RGBA against the golden PNG, or (re)write it under
/// `TRD_UPDATE_GOLDENS`. Returns `Err(reason)` on mismatch.
fn compare_or_update(actual: &[u8], golden: &Path) -> Result<(), String> {
    assert_eq!(
        actual.len(),
        (WIDTH * HEIGHT * 4) as usize,
        "frame RGBA length mismatch"
    );

    if update_goldens() {
        let img = image::RgbaImage::from_raw(WIDTH, HEIGHT, actual.to_vec())
            .expect("RGBA buffer -> image");
        img.save(golden)
            .map_err(|e| format!("write golden {}: {e}", golden.display()))?;
        return Ok(());
    }

    let expected = image::open(golden)
        .map_err(|e| {
            format!(
                "open golden {} ({e}); regenerate with TRD_UPDATE_GOLDENS=1",
                golden.display()
            )
        })?
        .to_rgba8();
    if expected.dimensions() != (WIDTH, HEIGHT) {
        return Err(format!(
            "golden {} is {:?}, expected {WIDTH}x{HEIGHT}",
            golden.display(),
            expected.dimensions()
        ));
    }

    let expected = expected.into_raw();
    let total = (WIDTH * HEIGHT) as usize;
    let mut differing = 0usize;
    let mut max_diff = 0u8;
    for p in 0..total {
        let mut pixel_diff = 0u8;
        for c in 0..4 {
            let a = actual[p * 4 + c];
            let e = expected[p * 4 + c];
            let d = a.abs_diff(e);
            pixel_diff = pixel_diff.max(d);
        }
        max_diff = max_diff.max(pixel_diff);
        if pixel_diff > CHANNEL_EPS {
            differing += 1;
        }
    }

    let fraction = differing as f64 / total as f64;
    if fraction > MAX_DIFF_FRACTION {
        return Err(format!(
            "{}: {differing}/{total} pixels differ beyond eps={CHANNEL_EPS} \
             ({:.3}% > {:.3}%; max channel diff {max_diff}). \
             If this change is intentional, regenerate with TRD_UPDATE_GOLDENS=1",
            golden.display(),
            fraction * 100.0,
            MAX_DIFF_FRACTION * 100.0,
        ));
    }
    Ok(())
}

/// Render a fixture and compare every frame to its golden PNG.
fn check_fixture(name: &str, fixture: &str, options: RenderOptions) {
    let frames = render_fixture(fixture, options);
    assert!(!frames.is_empty(), "{fixture} produced no frames");

    let dir = golden_dir();
    let mut failures = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        let golden = dir.join(format!("{name}_frame_{i}.png"));
        match compare_or_update(frame, &golden) {
            Err(reason) => {
                dump_actual(name, i, frame, &golden);
                failures.push(reason);
            }
            Ok(()) if !update_goldens() && should_dump() => dump_actual(name, i, frame, &golden),
            Ok(()) => {}
        }
    }

    if update_goldens() {
        eprintln!("updated {} golden frame(s) for {name}", frames.len());
    }
    assert!(
        failures.is_empty(),
        "golden mismatch (actual frames + diffs dumped under {}):\n{}",
        dump_dir().display(),
        failures.join("\n")
    );
}

/// Stage 1: the reconstructed placement quad only — a cyan wireframe quad placed
/// by the authored CV camera, with each draw's local axes. Exercises Arrow
/// mesh+params decode, CV `k` normalization, per-draw model + `wireframe` mode,
/// and the local-axes gizmo. Rendered at 4× MSAA (the default anti-aliased mesh
/// pass); the [`golden_stage1_placement_quad_no_msaa`] counterpart covers the
/// single-sample path.
#[test]
#[ignore = "requires a GPU adapter"]
fn golden_stage1_placement_quad() {
    check_fixture(
        "stage1",
        "stage1.arrow",
        RenderOptions {
            mode: RenderMode::Filled,
            show_aabb: false,
            show_axes: false,
            show_local_axes: true,
            show_local_grid: None,
            show_local_grid_mesh: None,
            show_world_grid: None,

            show_object_grid: None,

            selected: None,

            env_background: None,

            pbr: None,
            msaa: Msaa::X4,
        },
    );
}

/// Stage 1 with **MSAA disabled** ([`Msaa::Off`], single-sample): the same
/// placement-quad scene rendered without multisampling, so the wireframe / axes
/// edges are the raw rasterized coverage. Guards the non-MSAA color-attachment
/// path (no MSAA target, no resolve) against regression, and pins its own golden.
#[test]
#[ignore = "requires a GPU adapter"]
fn golden_stage1_placement_quad_no_msaa() {
    check_fixture(
        "stage1_noaa",
        "stage1.arrow",
        RenderOptions {
            mode: RenderMode::Filled,
            show_aabb: false,
            show_axes: false,
            show_local_axes: true,
            show_local_grid: None,
            show_local_grid_mesh: None,
            show_world_grid: None,

            show_object_grid: None,

            selected: None,

            env_background: None,

            pbr: None,
            msaa: Msaa::Off,
        },
    );
}

/// Stage 2: the textured bunny anchored on the placement quad, with its AABB,
/// local axes, and the wireframe quad overlay. Exercises the full textured
/// pipeline (0.0.4 texture table), multi-mesh draw lists, AABB + local-axes
/// gizmos, and per-draw mode inheritance. Rendered at 4× MSAA; the
/// [`golden_stage2_textured_bunny_no_msaa`] counterpart covers the single-sample
/// path.
#[test]
#[ignore = "requires a GPU adapter"]
fn golden_stage2_textured_bunny() {
    check_fixture(
        "stage2",
        "stage2.arrow",
        RenderOptions {
            mode: RenderMode::Textured,
            show_aabb: true,
            show_axes: false,
            show_local_axes: true,
            show_local_grid: None,
            show_local_grid_mesh: None,
            show_world_grid: None,

            show_object_grid: None,

            selected: None,

            env_background: None,

            pbr: None,
            msaa: Msaa::X4,
        },
    );
}

/// Stage 2 with **MSAA disabled** ([`Msaa::Off`], single-sample): the textured
/// bunny + AABB + local axes rendered without multisampling. The mesh silhouette
/// and gizmo edges are aliased, so its golden differs from the 4× one — together
/// they pin both the anti-aliased and the raw single-sample mesh passes.
#[test]
#[ignore = "requires a GPU adapter"]
fn golden_stage2_textured_bunny_no_msaa() {
    check_fixture(
        "stage2_noaa",
        "stage2.arrow",
        RenderOptions {
            mode: RenderMode::Textured,
            show_aabb: true,
            show_axes: false,
            show_local_axes: true,
            show_local_grid: None,
            show_local_grid_mesh: None,
            show_world_grid: None,

            show_object_grid: None,

            selected: None,

            env_background: None,

            pbr: None,
            msaa: Msaa::Off,
        },
    );
}

/// Stage 2 (**Disney PBR**, ACES tone-map): the same bunny + wireframe-quad
/// scene as [`golden_stage2_textured_bunny`], but the bunny is shaded through the
/// physically-based `disney.wgsl` path — a saturated strongly-metallic green
/// under the synthetic HDR env probe — and tone-mapped with the ACES filmic
/// curve ([`Tonemap::Aces`], #116). Exercises PBR material upload, the
/// environment-map reflection, and the ACES tone-map stage end-to-end; the
/// [`golden_stage2_pbr_reinhard`] counterpart pins the historical Reinhard curve
/// on the identical scene, so a regression in *either* operator is caught.
#[test]
#[ignore = "requires a GPU adapter"]
fn golden_stage2_pbr_aces() {
    check_fixture(
        "stage2_pbr_aces",
        "stage2.arrow",
        pbr_options(Tonemap::Aces),
    );
}

/// Stage 2 (**Disney PBR**, Reinhard tone-map): the PBR bunny scene tone-mapped
/// with the historical per-channel Reinhard curve ([`Tonemap::Reinhard`], the
/// default). Together with [`golden_stage2_pbr_aces`] it pins both tone-map
/// operators on an identical physically-based scene, isolating the tone-map
/// stage of `disney.wgsl`.
#[test]
#[ignore = "requires a GPU adapter"]
fn golden_stage2_pbr_reinhard() {
    check_fixture(
        "stage2_pbr_reinhard",
        "stage2.arrow",
        pbr_options(Tonemap::Reinhard),
    );
}

// ---------------------------------------------------------------------------
// #182 P9 — the probe yaw as the single source of truth.
// ---------------------------------------------------------------------------

/// A deliberately **yaw-asymmetric** HDR probe: four saturated quadrants around
/// the horizon plus one bright lobe in the +X quadrant.
///
/// The existing [`synthetic_env`] gradient is nearly rotation-invariant away
/// from its sun, so it cannot catch a yaw disagreement. This one can: rotating
/// it swaps which color a given view direction sees, so if the sky and the
/// reflections read different yaws the golden shows two different colors meeting
/// at the silhouette.
fn quadrant_env() -> EnvMapData {
    const W: u32 = 128;
    const H: u32 = 64;
    // +X, +Z, -X, -Z quadrant colors (linear RGB), walking u = 0…1.
    let quadrants = [
        [0.90f32, 0.10, 0.10], // red
        [0.10f32, 0.75, 0.20], // green
        [0.10f32, 0.20, 0.95], // blue
        [0.85f32, 0.80, 0.10], // yellow
    ];
    let mut rgba = vec![0.0f32; (W * H * 4) as usize];
    for y in 0..H {
        let v = y as f32 / (H - 1) as f32;
        // Darken toward the nadir so the horizon is legible.
        let shade = 1.0 - 0.65 * v;
        for x in 0..W {
            let u = x as f32 / W as f32;
            let q = quadrants[((u * 4.0) as usize).min(3)];
            // One bright HDR lobe, inside the first quadrant only.
            let du = u - 0.12;
            let dv = v - 0.30;
            let lobe = (-(du * du * 24.0 + dv * dv * 8.0) * 12.0).exp() * 5.0;
            let i = ((y * W + x) * 4) as usize;
            rgba[i] = q[0] * shade + lobe;
            rgba[i + 1] = q[1] * shade + lobe * 0.9;
            rgba[i + 2] = q[2] * shade + lobe * 0.7;
            rgba[i + 3] = 1.0;
        }
    }
    EnvMapData::from_rgba32f(W, H, rgba, 2048)
}

/// A UV sphere — a mirror ball is the canonical way to pin a probe's
/// orientation, because it reflects *every* direction at once, so a yaw error
/// cannot hide behind a flat facet.
fn uv_sphere(radius: f32, segments: u32, rings: u32) -> Mesh {
    let mut vertices = Vec::new();
    for ring in 0..=rings {
        let phi = std::f32::consts::PI * ring as f32 / rings as f32;
        for segment in 0..=segments {
            let theta = std::f32::consts::TAU * segment as f32 / segments as f32;
            let position = [
                radius * phi.sin() * theta.cos(),
                radius * phi.cos(),
                radius * phi.sin() * theta.sin(),
            ];
            vertices.push(Vertex {
                position,
                color: [1.0, 1.0, 1.0],
                uv: [segment as f32 / segments as f32, ring as f32 / rings as f32],
            });
        }
    }
    let mut indices = Vec::new();
    let stride = segments + 1;
    for ring in 0..rings {
        for segment in 0..segments {
            let a = ring * stride + segment;
            let b = a + stride;
            indices.extend([a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    Mesh {
        vertices,
        indices,
        shading: None,
    }
}

/// **The probe yaw is one value** (#182 P9): the sky and the reflections on the
/// object in front of it are driven by the same scene-level
/// [`EnvironmentLight::rotation`], so they can no longer disagree.
///
/// The scene is assembled through the **shared** [`Scene::from_draws`] path from
/// a wire draw list plus [`RenderOptions`], exactly as every front-end assembles
/// a frame: a **yaw-asymmetric** probe, a **non-zero** yaw, a **visible** sky
/// (`env_background`), and a **near-mirror metallic** ball. Before #235 R2 this
/// had to bypass that assembly and set `Background::environment` by hand,
/// because `from_draws` hard-coded it to `None` — so the CLI and both browser
/// renderers could never draw a sky at all. The golden is unchanged by that
/// move, which is the point: the shared assembly yields the same frame the
/// hand-built scene did.
///
/// No *fixture* can express it — every golden fixture composites a frames-table
/// background plane **over** the sky — so the draw list is built here.
/// Before P9 the same picture needed two rotations set in agreement by hand; a
/// regression that lets them drift shows up here as a sphere reflecting one
/// quadrant color against a sky of another.
#[test]
#[ignore = "requires a GPU adapter"]
fn golden_environment_light_syncs_sky_and_reflection() {
    let _serial = GPU_SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let mesh = uv_sphere(0.85, 48, 24);
    let (mut renderer, target) = pollster::block_on(Renderer::with_meshes(WIDTH, HEIGHT, &[mesh]))
        .expect("build renderer for the environment-light golden");
    renderer.set_env_map(quadrant_env());
    renderer.set_disney_material(DisneyMaterial {
        base_color: [1.0, 1.0, 1.0],
        metallic: 1.0,
        roughness: 0.06,
        specular: 1.0,
        ..DisneyMaterial::default()
    });
    renderer.set_image_based_lighting(ImageBasedLighting { intensity: 1.0 });
    renderer.set_tone_mapping(ToneMapping {
        operator: Tonemap::Aces,
        exposure: 1.0,
    });

    // A yaw no symmetry can hide: 2.2 rad ≈ 126°, inside the second quadrant.
    let scene = Scene::from_draws(
        &[Draw {
            mesh_id: 0,
            model: Matrix4::IDENTITY.to_cols_array(),
            selection: DrawSelection::Mesh(Some(RenderMode::Shaded)),
        }],
        &RenderOptions {
            mode: RenderMode::Shaded,
            // The sky is an ordinary appearance option (#235 R2) — the same one
            // `--env-background` and the browsers' `setEnvBackground` set.
            env_background: Some(EnvironmentBackground {
                exposure: 1.0,
                blur: 0.0,
                tonemap: Tonemap::Aces,
            }),
            ..RenderOptions::default()
        },
        None,
    )
    .with_lighting(Lighting {
        // Kill the direct rig so the picture is *only* the probe: any change is
        // then unambiguously the environment's.
        ambient: 0.0,
        scale: 0.0,
        environment: EnvironmentLight {
            intensity: 1.0,
            rotation: 2.2,
        },
    });

    let viewport = Viewport {
        width: WIDTH,
        height: HEIGHT,
    };
    let camera = Camera::look_at(
        Point3::new(0.0, 0.35, 2.6),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::Y,
        45f32.to_radians(),
        viewport,
    );
    let actual =
        pollster::block_on(renderer.render_layers(&[SceneLayer::new(camera, &scene)], &target))
            .expect("render the environment-light golden");

    let golden = golden_dir().join("environment_light.png");
    if let Err(reason) = compare_or_update(&actual, &golden) {
        dump_actual("environment_light", 0, &actual, &golden);
        panic!("{reason}");
    }
}
