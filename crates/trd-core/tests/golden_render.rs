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
//! ([`trd_core::RenderMode::Pbr`]) with a deterministic synthetic HDR environment
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
    run_stream, DisneyMaterial, EnvMapData, ImageBasedLighting, Lighting, Msaa, PbrConfig,
    RenderMode, RenderOptions, ToneMapping, Tonemap,
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
        mode: RenderMode::Pbr,
        show_aabb: true,
        show_axes: false,
        show_local_axes: true,
        show_local_grid: None,
        show_local_grid_mesh: None,
        show_world_grid: None,

        show_object_grid: None,

        selected: None,

        pbr: Some(PbrConfig {
            material: pbr_material(),
            lighting: Lighting {
                ambient: 0.05,
                ..Lighting::default()
            },
            ibl: ImageBasedLighting {
                intensity: 1.0,
                ..ImageBasedLighting::default()
            },
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
