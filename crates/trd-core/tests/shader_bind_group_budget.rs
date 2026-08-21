//! Bind-group budget guard (issue #126, sub-risk 1).
//!
//! Every `.wgsl` in `src/shader/` must fit the **portable** WebGPU baseline,
//! `wgpu::Limits::downlevel_defaults().max_bind_groups` — 4. That baseline is
//! what `LimitsPreset::Downlevel` requests in `render/gpu_context.rs`, and it is
//! what the browser path depends on.
//!
//! #126 asked for a debug assert at pipeline creation. This is the same check
//! moved earlier and made cheaper: it reads the shader source, so it needs no
//! device, runs on every platform in `nix flake check`, and fails on the machine
//! of whoever added the binding rather than in a browser that refuses to start.
//!
//! The margin is currently **zero** — `pbr.wgsl` binds `@group(0)` through
//! `@group(3)` (camera+material, albedo, environment, material maps). Anything
//! new must fold into an existing group, as `render/bound_uniform.rs` already
//! documents doing for the frame-wide uniform and the per-mesh slot array.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn shader_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shader")
}

/// The distinct `@group(N)` indices a shader declares, in ascending order.
///
/// Deliberately a plain scan rather than a WGSL parse: the attribute form is
/// fixed, and a parser would be a second thing that can be wrong.
fn declared_groups(source: &str) -> BTreeSet<u32> {
    let mut groups = BTreeSet::new();
    let mut rest = source;
    while let Some(at) = rest.find("@group(") {
        rest = &rest[at + "@group(".len()..];
        if let Some(close) = rest.find(')') {
            if let Ok(index) = rest[..close].trim().parse::<u32>() {
                groups.insert(index);
            }
        }
    }
    groups
}

fn shaders() -> Vec<(String, String)> {
    let dir = shader_dir();
    let mut out: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|entry| entry.expect("shader dir entry").path())
        .filter(|path| path.extension().is_some_and(|e| e == "wgsl"))
        .map(|path| {
            let name = path
                .file_name()
                .expect("shader file name")
                .to_string_lossy()
                .into_owned();
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            (name, source)
        })
        .collect();
    out.sort();
    out
}

/// The guard itself: no shader may declare a group index the portable baseline
/// cannot bind.
#[test]
fn every_shader_fits_the_portable_bind_group_limit() {
    let limit = wgpu::Limits::downlevel_defaults().max_bind_groups;
    let shaders = shaders();
    assert!(
        !shaders.is_empty(),
        "no .wgsl found in {} — this guard would pass vacuously",
        shader_dir().display()
    );

    let mut over = Vec::new();
    for (name, source) in &shaders {
        let groups = declared_groups(source);
        let Some(&highest) = groups.iter().next_back() else {
            continue;
        };
        // Group indices are zero-based, so index `highest` needs `highest + 1`
        // slots however sparsely they are used.
        if highest + 1 > limit {
            over.push(format!(
                "{name}: declares @group({highest}), needing {} of {limit} slots (groups: {groups:?})",
                highest + 1
            ));
        }
    }

    assert!(
        over.is_empty(),
        "shader(s) exceed the portable max_bind_groups = {limit}:\n  {}\n\n\
         This compiles and runs natively and fails in the browser, which is the\n\
         failure #126 sub-risk 1 exists to prevent. Fold the new data into an\n\
         existing group — an extra binding, or a dynamic-offset uniform — as\n\
         render/bound_uniform.rs documents doing for the PBR slot array.",
        over.join("\n  ")
    );
}

/// Pins the fact that motivates the guard: the PBR path already spends the whole
/// budget, so the next binding added to it has nowhere to go.
///
/// If this fails because `pbr.wgsl` now uses *fewer* groups, that is good news —
/// update the expectation and say what freed the slot.
#[test]
fn the_pbr_shader_has_no_bind_group_headroom_left() {
    let limit = wgpu::Limits::downlevel_defaults().max_bind_groups;
    let (_, pbr) = shaders()
        .into_iter()
        .find(|(name, _)| name == "pbr.wgsl")
        .expect("pbr.wgsl is the physically-based path and must exist");

    let groups = declared_groups(&pbr);
    let highest = *groups.iter().next_back().expect("pbr.wgsl binds groups");

    assert_eq!(
        highest + 1,
        limit,
        "pbr.wgsl uses {} of {limit} bind groups (groups: {groups:?}); \
         this test records that the budget is exactly spent",
        highest + 1
    );
}

/// The scan must be able to find something, or the guard above proves nothing.
///
/// This exists because an under-scoped search returning "no violations" is
/// indistinguishable from a correct pass.
#[test]
fn the_group_scan_actually_parses_attributes() {
    let parsed = declared_groups(
        "@group(0) @binding(0) var<uniform> a: A;\n@group(2) @binding(1) var b: texture_2d<f32>;",
    );
    assert_eq!(parsed, BTreeSet::from([0, 2]));
    assert!(declared_groups("no attributes here").is_empty());
}
