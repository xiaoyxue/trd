//! Guards the "one browser delivery surface" rule (#180, #299, #302).
//!
//! `AGENTS.md` states that **every** `#[wasm_bindgen]` item in the repo lives in
//! `crates/trd-wasm`, and that no other crate declares a `cdylib`. That is what
//! keeps one wasm build producing one generated JS package for all three `web/`
//! packages. It is also exactly the kind of rule that erodes one convenient
//! export at a time, so it is asserted here rather than left to review.
//!
//! The same scan now covers **`web_sys`** (#302). Naming a browser type is how
//! the boundary actually eroded last time: `trd-core` and `trd-gui` each grew a
//! `web-sys` dependency and eleven `cfg`s to hide it, none of which a native
//! build compiles — so nothing but review stood between the platform-neutral
//! crates and the browser API. `trd_core::ExternalFrame` is the seam that
//! replaced it, and this is what keeps the seam from being routed around.
//!
//! This is a **source scan**, not a compile-time check: a stray binding inside a
//! `#[cfg(target_arch = "wasm32")]` block would not show up in a native build at
//! all, which is precisely the case worth catching.

use std::fs;
use std::path::{Path, PathBuf};

/// Built at runtime so this file does not match its own scan.
fn attribute_needle() -> String {
    format!("#[{}_{}", "wasm", "bindgen")
}

/// Likewise: the crate path, spelled so this file's own prose does not match.
fn web_sys_needle() -> String {
    format!("{}_{}", "web", "sys")
}

fn workspace_root() -> PathBuf {
    // `crates/trd-wasm` → repo root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/trd-wasm sits two levels below the workspace root")
        .to_path_buf()
}

/// Every Rust source under `crates/` and `native/`, skipping build output.
fn rust_sources(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for area in ["crates", "native"] {
        collect(&root.join(area), &mut found);
    }
    found.sort();
    found
}

/// Every crate manifest under `crates/` and `native/`.
fn crate_manifests(root: &Path) -> Vec<PathBuf> {
    let mut manifests = Vec::new();
    for area in ["crates", "native"] {
        let Ok(entries) = fs::read_dir(root.join(area)) else {
            continue;
        };
        for entry in entries.flatten() {
            let manifest = entry.path().join("Cargo.toml");
            if manifest.is_file() {
                manifests.push(manifest);
            }
        }
    }
    manifests.sort();
    manifests
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // `target/` is build output and `pkg/` is generated wasm-bindgen glue.
            if matches!(
                path.file_name().and_then(|n| n.to_str()),
                Some("target" | "pkg")
            ) {
                continue;
            }
            collect(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn contains(path: &Path, needle: &str) -> bool {
    fs::read_to_string(path)
        .map(|text| text.contains(needle))
        .unwrap_or(false)
}

/// Whether `needle` appears in actual code, ignoring anything after a line
/// comment marker.
///
/// Needed because the crates this scan *forbids* are the ones that most need to
/// explain in prose why they no longer use the thing: `external_frame.rs` and
/// the `trd-gui` frame slot both name the browser crate in a doc comment to say
/// what they replaced. A plain substring match reads those as violations.
fn code_contains(path: &Path, needle: &str, comment: &str) -> bool {
    let Ok(text) = fs::read_to_string(path) else {
        return false;
    };
    text.lines()
        .map(|line| line.split(comment).next().unwrap_or(""))
        .any(|code| code.contains(needle))
}

/// Whether a manifest actually *declares* a `cdylib`, ignoring prose.
///
/// `crates/trd-gui/Cargo.toml` names `cdylib` in a comment explaining that it
/// deliberately is not one — a plain substring match reads that as a violation.
fn declares_cdylib(path: &Path) -> bool {
    code_contains(path, "cdylib", "#")
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
        .replace('\\', "/")
}

#[test]
fn wasm_bindgen_lives_only_in_trd_wasm() {
    let root = workspace_root();
    let needle = attribute_needle();
    let allowed = root.join("crates").join("trd-wasm");

    let sources = rust_sources(&root);
    assert!(
        !sources.is_empty(),
        "found no Rust sources under {} — the scan is broken, not the tree",
        root.display()
    );

    let strays: Vec<_> = sources
        .iter()
        .filter(|path| !path.starts_with(&allowed))
        .filter(|path| contains(path, &needle))
        .map(|path| relative(&root, path))
        .collect();

    assert!(
        strays.is_empty(),
        "`{needle}` may only appear in crates/trd-wasm (AGENTS.md, #180); found it in:\n  {}",
        strays.join("\n  ")
    );

    // The rule is only meaningful while the scan can still see the real thing.
    let covered = sources
        .iter()
        .filter(|path| path.starts_with(&allowed))
        .any(|path| contains(path, &needle));
    assert!(
        covered,
        "no `{needle}` found in crates/trd-wasm — the scan stopped matching, \
         so it would no longer catch a stray binding elsewhere"
    );
}

#[test]
fn trd_wasm_is_the_only_cdylib() {
    let root = workspace_root();
    let allowed = root.join("crates").join("trd-wasm").join("Cargo.toml");

    let manifests = crate_manifests(&root);
    assert!(!manifests.is_empty(), "found no crate manifests to check");

    let strays: Vec<_> = manifests
        .iter()
        .filter(|path| **path != allowed)
        .filter(|path| declares_cdylib(path))
        .map(|path| relative(&root, path))
        .collect();

    assert!(
        strays.is_empty(),
        "only crates/trd-wasm may declare a cdylib (AGENTS.md, #180); found one in:\n  {}",
        strays.join("\n  ")
    );

    assert!(
        declares_cdylib(&allowed),
        "crates/trd-wasm no longer declares a cdylib — this guard needs revisiting"
    );
}

/// The browser API stays behind the delivery surface (#302).
///
/// `trd-core` and `trd-gui` build for native *and* wasm, so anything they name
/// from `web-sys` has to be hidden behind a `cfg` that a native build never
/// compiles — which is how eleven of them accumulated unnoticed. The seam that
/// replaced them is [`trd_core::ExternalFrame`]: the shared crates describe the
/// frame, `trd-wasm` implements the copy. Prose that *mentions* the crate to
/// explain the boundary is fine; a dependency or a path in code is not.
#[test]
fn web_sys_lives_only_in_trd_wasm() {
    let root = workspace_root();
    let needle = web_sys_needle();
    let dependency = needle.replace('_', "-");
    let allowed = root.join("crates").join("trd-wasm");

    let sources = rust_sources(&root);
    assert!(
        !sources.is_empty(),
        "found no Rust sources under {} — the scan is broken, not the tree",
        root.display()
    );

    let mut strays: Vec<_> = sources
        .iter()
        .filter(|path| !path.starts_with(&allowed))
        .filter(|path| code_contains(path, &needle, "//"))
        .map(|path| relative(&root, path))
        .collect();

    strays.extend(
        crate_manifests(&root)
            .iter()
            .filter(|path| !path.starts_with(&allowed))
            .filter(|path| code_contains(path, &dependency, "#"))
            .map(|path| relative(&root, path)),
    );

    assert!(
        strays.is_empty(),
        "`{needle}` may only be named in crates/trd-wasm — the shared crates go \
         through `trd_core::ExternalFrame` (AGENTS.md, #302); found it in:\n  {}",
        strays.join("\n  ")
    );

    // The rule is only meaningful while the scan can still see the real thing.
    let covered = sources
        .iter()
        .filter(|path| path.starts_with(&allowed))
        .any(|path| code_contains(path, &needle, "//"));
    assert!(
        covered,
        "no `{needle}` found in crates/trd-wasm — the scan stopped matching, so \
         it would no longer catch a stray reference elsewhere"
    );
}
