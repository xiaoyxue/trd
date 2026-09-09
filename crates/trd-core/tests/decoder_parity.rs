//! Parity of the current `[params][mesh]` document entry points.
//!
//! Native `run_stream` uses [`trd_core::SceneDocument::read_from`]; browser
//! `ArrowSceneDocument.fromArrow` uses [`trd_core::SceneDocument::read`].
//! Exercise fragmented native reads against the browser's complete buffer,
//! then compare per-row rendering views, resolved bindings, external frame
//! references and all retained source data. The legacy mesh-first
//! `InputSession` is not the entry point for these migrated fixtures.

use std::io::Read;
use std::path::{Path, PathBuf};

use trd_core::SceneDocument;

struct ChunkedReader<'a> {
    bytes: &'a [u8],
    chunk_size: usize,
}

impl Read for ChunkedReader<'_> {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        let count = self.bytes.len().min(self.chunk_size).min(output.len());
        output[..count].copy_from_slice(&self.bytes[..count]);
        self.bytes = &self.bytes[count..];
        Ok(count)
    }
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(name)
}

fn assert_parity(fixture_name: &str) {
    let bytes =
        std::fs::read(fixture(fixture_name)).unwrap_or_else(|e| panic!("read {fixture_name}: {e}"));
    let browser = SceneDocument::read(&bytes).expect("browser document");
    let frames = browser.frames().expect("browser rendering views");
    assert!(!frames.is_empty(), "{fixture_name}: no frames");
    assert!(!browser.meshes().is_empty(), "{fixture_name}: no GLBs");

    for chunk_size in [1, 7, 64 * 1024] {
        let native = SceneDocument::read_from(ChunkedReader {
            bytes: &bytes,
            chunk_size,
        })
        .unwrap_or_else(|error| panic!("{fixture_name}, chunk {chunk_size}: {error}"));
        assert_eq!(native.schema(), browser.schema());
        assert_eq!(native.batches(), browser.batches());
        assert_eq!(native.meshes(), browser.meshes());
        assert_eq!(native.render_config(), browser.render_config());
        assert_eq!(
            native.row_count(),
            frames.len(),
            "{fixture_name}: frame count"
        );
        for (row, expected) in frames.iter().enumerate() {
            let actual = native.frame(row).expect("native rendering view");
            assert_eq!(
                &actual, expected,
                "{fixture_name} row {row}, chunk {chunk_size}: rendering view"
            );
            assert_eq!(
                native.frame_ref(row).unwrap(),
                browser.frame_ref(row).unwrap(),
                "{fixture_name} row {row}: external background reference"
            );
            assert!(
                !actual.objects.is_empty(),
                "{fixture_name} row {row}: no draws"
            );
            for object in 0..actual.objects.len() {
                assert_eq!(
                    native.object_mesh_slot(&actual, object).unwrap(),
                    browser.object_mesh_slot(expected, object).unwrap(),
                    "{fixture_name} row {row}, object {object}: resolved mesh"
                );
            }
        }
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

#[test]
fn shadow_metadata_edits_survive_fragmented_read_and_preserve_source_data() {
    let original = SceneDocument::read(&std::fs::read(fixture("stage2.arrow")).unwrap()).unwrap();
    for enable in [true, false] {
        let mut document = original.clone();
        let mut config = document.render_config();
        config.shadow.enable = enable;
        document.set_render_config(config).unwrap();
        let bytes = document.write().unwrap();
        let browser = SceneDocument::read(&bytes).unwrap();
        let native = SceneDocument::read_from(ChunkedReader {
            bytes: &bytes,
            chunk_size: 1,
        })
        .unwrap();
        assert_eq!(native.render_config(), config);
        assert_eq!(native.schema(), browser.schema());
        assert_eq!(native.batches(), browser.batches());
        assert_eq!(native.meshes(), original.meshes());
        assert_eq!(native.frames().unwrap(), original.frames().unwrap());
        assert_eq!(native.schema().fields(), original.schema().fields());
        for (after, before) in native.batches().iter().zip(original.batches()) {
            assert_eq!(after.columns(), before.columns());
            assert_eq!(after.num_rows(), before.num_rows());
        }
    }
}
