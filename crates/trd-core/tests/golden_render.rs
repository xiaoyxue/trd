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
//! a few frames at a small resolution and made self-contained (mesh + texture
//! embedded, background dropped) by `scripts/golden_fixtures.py`:
//!
//! * `stage1.arrow` — the reconstructed placement quad (wireframe) + local axes;
//! * `stage2.arrow` — a textured bunny anchored on that quad, with AABB + local
//!   axes + the wireframe quad overlay.
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

use std::path::{Path, PathBuf};

use arrow::array::{Array, FixedSizeListArray, UInt8Array};
use arrow::ipc::reader::StreamReader;
use trd_core::{run_stream, RenderMode, RenderOptions};

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

/// Run the committed fixture through the real pipeline and decode the output
/// Arrow stream back into per-frame tightly-packed RGBA.
fn render_fixture(fixture: &str, options: RenderOptions) -> Vec<Vec<u8>> {
    let bytes = std::fs::read(golden_dir().join(fixture))
        .unwrap_or_else(|e| panic!("read fixture {fixture}: {e}"));

    let mut out = Vec::new();
    run_stream(&bytes[..], &mut out, WIDTH, HEIGHT, options, None)
        .unwrap_or_else(|e| panic!("run_stream on {fixture}: {e:?}"));

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
        if let Err(reason) = compare_or_update(frame, &golden) {
            failures.push(reason);
        }
    }

    if update_goldens() {
        eprintln!("updated {} golden frame(s) for {name}", frames.len());
    }
    assert!(
        failures.is_empty(),
        "golden mismatch:\n{}",
        failures.join("\n")
    );
}

/// Stage 1: the reconstructed placement quad only — a cyan wireframe quad placed
/// by the authored CV camera, with each draw's local axes. Exercises Arrow
/// mesh+params decode, CV `k` normalization, per-draw model + `wireframe` mode,
/// and the local-axes gizmo.
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
        },
    );
}

/// Stage 2: the textured bunny anchored on the placement quad, with its AABB,
/// local axes, and the wireframe quad overlay. Exercises the full textured
/// pipeline (0.0.4 texture table), multi-mesh draw lists, AABB + local-axes
/// gizmos, and per-draw mode inheritance.
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
        },
    );
}
