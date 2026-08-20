//! Decoder parity test (issue #88, guards #84).
//!
//! The column decode is **no longer duplicated**: since #104/#108 unified the
//! per-batch decoders and #296 split transport from format, both paths run the
//! same [`trd_core::InputSession`] over the one decoder in
//! `protocol/arrow_decode.rs`. The native side ([`trd_core::InputStream`],
//! `io/input_stream.rs`) is a *byte transport* that owns a `Read` and feeds that
//! session; the browser pushes bytes into it directly.
//!
//! So what this test guards is no longer decoder-versus-decoder divergence — it
//! is **driver** divergence. A pull loop over an IPC stream and a push session
//! differ in framing, chunk boundaries, accumulation across sub-streams,
//! external-reference resolution and inline-image decode, and a bug in any of
//! those appears on one path only. The original motivating defect — the
//! `input field `center` must be non-nullable` bug (`08c113a`), where the wasm
//! decoder rejected a stream the native decoder accepted — is the shape of
//! failure still worth catching, even though its specific cause is now shared
//! code.
//!
//! This test decodes the **same committed Arrow bytes** (the golden fixtures,
//! `[mesh][texture?][frames][params]`) through both drivers and asserts they
//! yield identical per-frame params, draws, external references, and decoded
//! inline background pixels. It needs no GPU, so —
//! unlike the golden render test — it runs in `nix flake check` (`cargo test`)
//! and guards both drivers on every change.

use std::path::{Path, PathBuf};

use trd_core::{Draw, FrameParams, ImageData, InlineFrameCache, InputSession, InputStream};

type Frame = (FrameParams, Vec<Draw>, Option<String>, Option<ImageData>);

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name)
}

/// Drive the whole stream (leading mesh/texture tables + params) through the
/// native decoder, collecting the resolved scene and inline image per frame.
fn native_frames(bytes: &[u8]) -> Vec<Frame> {
    let mut input = InputStream::new(bytes);
    input.prologue().expect("native prologue");
    let mut cache = InlineFrameCache::default();
    let mut frames = Vec::new();
    while let Some(batch) = input.next_batch() {
        for frame in batch.expect("native batch") {
            let inline = cache
                .resolve(frame.frame_id, input.frames())
                .expect("native inline frame")
                .map(|(image, _changed)| (*image).clone());
            frames.push((
                frame.params,
                frame.resolved_draws(),
                frame.frame_ref,
                inline,
            ));
        }
    }
    input.finish().expect("native decode");
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
