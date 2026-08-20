//! [`OutputSession`] — the semantic half of the output protocol.
//!
//! Holds the schema and turns a batch of tightly packed RGBA frames into its
//! `RecordBatch`. Owns no transport, which is what makes it a `*Session` rather
//! than a `*Stream`, and what keeps it non-generic and usable on every platform.

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;

use super::image_encode::{output_batch, output_schema_with_frame_rate, OutputError};

pub struct OutputSession {
    schema: Arc<Schema>,
    width: u32,
    height: u32,
}

impl OutputSession {
    /// Builds the session for a `width` × `height` image stream, optionally
    /// stamping the playback rate (`trd.stream.frame_rate`) into the schema.
    pub fn new(width: u32, height: u32, frame_rate: Option<f64>) -> Result<Self, OutputError> {
        Ok(Self {
            schema: Arc::new(output_schema_with_frame_rate(width, height, frame_rate)?),
            width,
            height,
        })
    }

    /// The stream's schema, as written by the IPC writer's header.
    pub fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }

    /// Encodes one batch of tightly packed RGBA frames into its planar
    /// `r`/`g`/`b`/`a` [`RecordBatch`].
    pub fn encode(&self, frames: &[Vec<u8>]) -> Result<RecordBatch, OutputError> {
        output_batch(self.schema.clone(), frames, self.width, self.height)
    }
}
