//! [`OutputStream`] — the transport half of the output protocol.
//!
//! Owns an [`OutputSession`] *and* the arrow `StreamWriter<W>` that writes into
//! the sink. arrow ships no transport-free encoder — `StreamWriter` is welded to
//! its `W` — so the generic parameter lives out here on the outer layer, exactly
//! as `InputStream<R>` carries it for the input side, and the semantic layer
//! stays clean.
//!
//! Unlike [`InputStream`](super::InputStream) this is **not** native-only: the
//! output side has a single execution model (push) and `Vec<u8>` already
//! implements `Write`, so the browser uses [`SharedBuffer`] and gets the same
//! type.

use std::cell::RefCell;
use std::io::Write;
use std::rc::Rc;

use arrow::ipc::writer::StreamWriter;

use crate::protocol::{OutputError, OutputSession};
use crate::session_state::SessionState;

/// An in-memory [`Write`] sink shared with its writer, for callers that have no
/// real transport to write into — the browser's `OffscreenRenderer`, which hands
/// finished IPC bytes to JS as a `Uint8Array`.
///
/// Native callers pass a real `W` (`StdoutLock`, a socket, a file) instead and
/// never need [`OutputStream::drain_new`].
#[derive(Clone, Default)]
pub struct SharedBuffer(Rc<RefCell<Vec<u8>>>);

impl SharedBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Removes and returns everything written since the last take.
    pub fn take(&self) -> Vec<u8> {
        std::mem::take(&mut *self.0.borrow_mut())
    }
}

impl Write for SharedBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The **transport** half of the output protocol: an [`OutputSession`] plus the
/// arrow IPC writer that owns the sink `W`.
///
/// `StreamWriter<W>` is welded to its `W` — arrow ships no transport-free
/// encoder — so the generic lives out here and `OutputSession` stays clean. The
/// header is written at construction, so a caller that passes a real `W` never
/// touches [`drain_new`](Self::drain_new).
pub struct OutputStream<W: Write> {
    session: OutputSession,
    writer: StreamWriter<W>,
    state: SessionState,
}

impl<W: Write> OutputStream<W> {
    /// Opens the stream over `sink`, writing the IPC header immediately.
    pub fn new(
        sink: W,
        width: u32,
        height: u32,
        frame_rate: Option<f64>,
    ) -> Result<Self, OutputError> {
        let session = OutputSession::new(width, height, frame_rate)?;
        let writer = StreamWriter::try_new(sink, &session.schema())?;
        Ok(Self {
            session,
            writer,
            state: SessionState::default(),
        })
    }

    fn ensure_open(&self) -> Result<(), OutputError> {
        self.state.ensure_open(
            OutputError::OutputSessionFinished,
            OutputError::OutputSessionFailed,
        )
    }

    fn fail<T>(&mut self, error: OutputError) -> Result<T, OutputError> {
        self.state.fail(error)
    }

    /// Encodes and writes one batch of tightly packed RGBA frames.
    pub fn write_rgba_batch(&mut self, frames: &[Vec<u8>]) -> Result<(), OutputError> {
        self.ensure_open()?;

        let batch = match self.session.encode(frames) {
            Ok(batch) => batch,
            Err(error) => return self.fail(error),
        };

        if let Err(error) = self.writer.write(&batch) {
            return self.fail(OutputError::Arrow(error));
        }

        Ok(())
    }

    /// Writes the end-of-stream marker. The stream is unusable afterwards.
    pub fn finish(&mut self) -> Result<(), OutputError> {
        self.ensure_open()?;

        let finished = self.writer.finish().map_err(OutputError::Arrow);
        self.state.close(finished)
    }
}

impl OutputStream<SharedBuffer> {
    /// Opens a stream over a fresh in-memory [`SharedBuffer`], for callers with
    /// no `Write` target of their own.
    pub fn buffered(width: u32, height: u32, frame_rate: Option<f64>) -> Result<Self, OutputError> {
        Self::new(SharedBuffer::new(), width, height, frame_rate)
    }

    /// Removes and returns all output bytes produced since the last drain.
    ///
    /// Only meaningful for the buffered form: a caller writing into a real `W`
    /// has already received the bytes. Fails if the stream previously failed, so
    /// a partially-written batch is never handed back as success-shaped bytes.
    pub fn drain_new(&mut self) -> Result<Vec<u8>, OutputError> {
        if self.state == SessionState::Failed {
            return Err(OutputError::OutputSessionFailed);
        }
        Ok(self.writer.get_ref().take())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{read_image_stream, PROTOCOL_VERSION, PROTOCOL_VERSION_KEY};
    use arrow::array::RecordBatch;
    use arrow::ipc::reader::StreamReader;

    #[test]
    fn output_session_drains_schema_batches_and_eos_once() {
        let mut output = OutputStream::buffered(2, 1, None).unwrap();

        let schema = output.drain_new().unwrap();
        assert!(!schema.is_empty());
        assert!(output.drain_new().unwrap().is_empty());

        output
            .write_rgba_batch(&[vec![1, 2, 3, 255, 4, 5, 6, 255]])
            .unwrap();
        let first = output.drain_new().unwrap();

        output
            .write_rgba_batch(&[vec![7, 8, 9, 255, 10, 11, 12, 255]])
            .unwrap();
        let second = output.drain_new().unwrap();

        output.finish().unwrap();
        let eos = output.drain_new().unwrap();

        let bytes = [schema, first, second, eos].concat();
        let reader = StreamReader::try_new(bytes.as_slice(), None).unwrap();

        assert_eq!(
            reader
                .schema()
                .metadata()
                .get(PROTOCOL_VERSION_KEY)
                .map(String::as_str),
            Some(PROTOCOL_VERSION)
        );

        let batches = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(
            batches
                .iter()
                .map(RecordBatch::num_rows)
                .collect::<Vec<_>>(),
            vec![1, 1]
        );
    }
    #[test]
    fn read_image_stream_roundtrips_written_frames() {
        let (w, h) = (2u32, 2u32);
        let frame0: Vec<u8> = (0..(w * h * 4) as u8).collect();
        let frame1: Vec<u8> = (0..(w * h * 4) as u8).map(|b| 255 - b).collect();

        let mut output = OutputStream::buffered(w, h, None).unwrap();
        let mut bytes = output.drain_new().unwrap();
        output
            .write_rgba_batch(&[frame0.clone(), frame1.clone()])
            .unwrap();
        bytes.extend(output.drain_new().unwrap());
        output.finish().unwrap();
        bytes.extend(output.drain_new().unwrap());

        let frames = read_image_stream(bytes.as_slice(), w, h).unwrap();
        assert_eq!(frames, vec![frame0, frame1]);
    }
    #[test]
    fn output_session_drain_releases_drained_bytes() {
        let mut output = OutputStream::buffered(2, 1, None).unwrap();

        assert!(!output.drain_new().unwrap().is_empty());
        assert!(output.writer.get_ref().0.borrow().is_empty());

        output
            .write_rgba_batch(&[vec![1, 2, 3, 255, 4, 5, 6, 255]])
            .unwrap();
        assert!(!output.drain_new().unwrap().is_empty());
        assert!(output.writer.get_ref().0.borrow().is_empty());
    }
    #[test]
    fn output_session_finish_without_batches_emits_eos() {
        let mut output = OutputStream::buffered(2, 1, None).unwrap();

        let schema = output.drain_new().unwrap();
        output.finish().unwrap();
        let eos = output.drain_new().unwrap();

        let bytes = [schema, eos].concat();
        let reader = StreamReader::try_new(bytes.as_slice(), None).unwrap();
        assert!(reader.collect::<Result<Vec<_>, _>>().unwrap().is_empty());

        assert!(matches!(
            output.write_rgba_batch(&[vec![0; 8]]),
            Err(OutputError::OutputSessionFinished)
        ));
        assert!(matches!(
            output.finish(),
            Err(OutputError::OutputSessionFinished)
        ));
    }
    #[test]
    fn output_session_is_terminal_after_batch_failure() {
        let mut output = OutputStream::buffered(2, 1, None).unwrap();

        assert!(matches!(
            output.write_rgba_batch(&[vec![0; 7]]),
            Err(OutputError::InvalidRgbaFrameLength {
                actual: 7,
                expected: 8
            })
        ));
        assert!(matches!(
            output.drain_new(),
            Err(OutputError::OutputSessionFailed)
        ));
        assert!(matches!(
            output.finish(),
            Err(OutputError::OutputSessionFailed)
        ));
    }
}
