//! Decoder parity test (issue #88, guards #84).
//!
//! The native decoder ([`trd_core::read_scene_stream_with_meta`], `stream.rs`)
//! and the wasm push decoder ([`trd_core::InputSession`], `protocol.rs`)
//! reimplement the same Arrow column decode + schema validation independently.
//! Divergence causes "fix the bug in one decoder but not the other" regressions
//! — e.g. the `input field `center` must be non-nullable` bug (`08c113a`), where
//! the wasm decoder rejected a stream the native decoder accepted.
//!
//! This test decodes the **same committed Arrow bytes** (the golden fixtures,
//! `[mesh][texture?][frames][params]`) through both paths and asserts they yield
//! identical per-frame params, draws, external references, and decoded inline
//! background pixels. It needs no GPU, so —
//! unlike the golden render test — it runs in `nix flake check` (`cargo test`)
//! and guards the decoders on every change.

use std::path::{Path, PathBuf};

use trd_core::{read_scene_stream_with_meta, Draw, FrameParams, ImageData, InputSession};

type Frame = (FrameParams, Vec<Draw>, Option<String>, Option<ImageData>);

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name)
}

/// Drive the whole stream (leading mesh/texture tables + params) through the
/// native decoder, collecting the resolved scene and inline image per frame.
fn native_frames(bytes: &[u8]) -> Vec<Frame> {
    let mut frames = Vec::new();
    read_scene_stream_with_meta(
        bytes,
        |_meshes| {},
        |_texture| {},
        |_rate| {},
        |params, draws, frame_ref, inline| {
            frames.push((params, draws, frame_ref, inline.as_deref().cloned()));
        },
    )
    .expect("native decode");
    frames
}

/// Feed the same bytes to the wasm push decoder, flattening its decoded frames.
/// Resolves each frame's draws (via [`trd_core::DecodedFrame::resolved_draws`])
/// so it compares against the native path's already-resolved draw list.
fn wasm_frames(bytes: &[u8]) -> Vec<Frame> {
    let mut session = InputSession::new();
    let mut frames = Vec::new();
    for batch in session.push(bytes).expect("wasm push") {
        for frame in batch {
            let inline = frame.frame_id.map(|id| {
                session.frames()[id as usize]
                    .decode()
                    .expect("inline frame decode")
            });
            frames.push((
                frame.params,
                frame.resolved_draws(),
                frame.frame_ref,
                inline,
            ));
        }
    }
    session.finish().expect("wasm finish");
    frames
}

fn assert_parity(fixture_name: &str) {
    let bytes =
        std::fs::read(fixture(fixture_name)).unwrap_or_else(|e| panic!("read {fixture_name}: {e}"));
    let native = native_frames(&bytes);
    let wasm = wasm_frames(&bytes);

    assert!(
        !native.is_empty(),
        "{fixture_name}: native decoded no frames"
    );
    assert_eq!(
        native.len(),
        wasm.len(),
        "{fixture_name}: native decoded {} frames, wasm decoded {}",
        native.len(),
        wasm.len()
    );
    for (i, (n, w)) in native.iter().zip(wasm.iter()).enumerate() {
        assert_eq!(
            n, w,
            "{fixture_name} frame {i}: native and wasm decoders disagree \
             on (FrameParams, draws, frame_ref, inline pixels)"
        );
    }
}

#[test]
fn native_and_wasm_decoders_agree_stage1() {
    assert_parity("stage1.arrow");
}

#[test]
fn native_and_wasm_decoders_agree_stage2() {
    assert_parity("stage2.arrow");
}
