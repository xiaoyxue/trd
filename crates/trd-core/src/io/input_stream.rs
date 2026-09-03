//! [`InputStream`] — the sync-pull byte transport over an [`InputSession`].
//!
//! `InputSession` (the format decoder) is deliberately transport-free: it is fed
//! `push(&[u8])` from whatever source a platform has, which is what lets the
//! browser drive it from its event loop. Natively there *is* a byte source, so
//! this owns it — the `R: Read`, the read buffer, and the loop that pumps one
//! into the other — and hands out decoded frames.
//!
//! There is no browser counterpart on purpose: `Read::read` returning `Ok(0)`
//! **means EOF** by contract, and "the bytes have not arrived yet" is not EOF.

use std::collections::VecDeque;
use std::io::Read;

use crate::protocol::FrameBatch;
use crate::stream_filter::StreamError;
use crate::{InlineFrame, InputSession, Mesh, MeshAsset, MeshReference};

/// How many bytes are pulled from the source per read.
const CHUNK: usize = 64 * 1024;

/// The mesh-first prologue: everything the stream declares *before* its first
/// frame, borrowed from the session that decoded it.
///
/// Reaching this is what makes the accessors meaningful, so it is a value rather
/// than a set of "only valid once ready" getters: holding a `Prologue` **is** the
/// proof that the params schema arrived.
#[derive(Debug, Clone)]
pub struct Prologue<'a> {
    /// The required leading mesh table, in stream order (mesh id = index).
    pub meshes: &'a [Mesh],
    /// Resolved material and glTF texture resources, parallel to `meshes`.
    pub mesh_assets: &'a [MeshAsset],
    /// References that must be resolved before `meshes` becomes available.
    pub mesh_references: Vec<(u32, MeshReference)>,
    /// The optional inline background resources, indexed by a frame's `frame_id`.
    pub frames: &'a [InlineFrame],
    /// The declared playback rate, defaulted when the stream omits it.
    pub frame_rate: f64,
}

/// A trd input stream read from `R`.
///
/// Yields one [`FrameBatch`] per input record batch through [`Iterator`], with
/// the read loop — and the fact that a single 64 KiB read may produce zero, one
/// or several batches — hidden inside [`next`](Iterator::next).
pub struct InputStream<R: Read> {
    source: R,
    buf: Box<[u8]>,
    session: InputSession,
    /// Batches decoded but not yet yielded: `push` returns 0..N per read, while
    /// `next` hands back exactly one.
    pending: VecDeque<FrameBatch>,
    /// Whether the params schema has been reached and the mesh-first contract
    /// checked, so the prologue is complete.
    ready: bool,
    /// Whether the source has signalled EOF; the session is finished once.
    eof: bool,
}

impl<R: Read> InputStream<R> {
    pub fn new(source: R) -> Self {
        Self {
            source,
            buf: vec![0u8; CHUNK].into_boxed_slice(),
            session: InputSession::new(),
            pending: VecDeque::new(),
            ready: false,
            eof: false,
        }
    }

    /// Pumps the source until the mesh-first prologue is complete, then borrows
    /// it.
    ///
    /// Enforces the mesh-first contract here, once: a stream whose params schema
    /// arrives with no leading mesh table — or that ends before any schema at
    /// all — is a [`StreamError::MissingMeshStream`].
    pub fn prologue(&mut self) -> Result<Prologue<'_>, StreamError> {
        while !self.ready {
            if !self.pump()? {
                // The source ended before a params schema was ever reached.
                return Err(StreamError::MissingMeshStream);
            }
        }
        Ok(Prologue {
            meshes: self.session.meshes(),
            mesh_assets: self.session.mesh_assets(),
            mesh_references: self.session.unresolved_mesh_references(),
            frames: self.session.frames(),
            frame_rate: self
                .session
                .frame_rate()
                .unwrap_or(crate::DEFAULT_FRAME_RATE),
        })
    }

    /// The stream's inline background resources.
    ///
    /// The same slice [`Prologue::frames`] borrows, reachable *between* batches:
    /// a `for batch in &mut stream` loop holds the stream mutably for its whole
    /// body, so a caller that needs the frames per batch drives
    /// [`next_batch`](Self::next_batch) instead and reads them here.
    pub fn frames(&self) -> &[InlineFrame] {
        self.session.frames()
    }

    pub fn resolve_gltf(&mut self, index: u32, bytes: &[u8]) -> Result<(), StreamError> {
        self.session.resolve_gltf(index, bytes)?;
        Ok(())
    }

    /// The next decoded batch, or `None` at end of stream.
    ///
    /// The inherent form of [`Iterator::next`], for callers that also need
    /// `&self` access (e.g. [`frames`](Self::frames)) inside their loop.
    pub fn next_batch(&mut self) -> Option<Result<FrameBatch, StreamError>> {
        loop {
            if let Some(batch) = self.pending.pop_front() {
                return Some(Ok(batch));
            }
            match self.pump() {
                Ok(true) => continue,
                Ok(false) => return None,
                Err(error) => return Some(Err(error)),
            }
        }
    }

    /// Closes the stream, validating that it was well-formed to the end.
    pub fn finish(&mut self) -> Result<(), StreamError> {
        if !self.ready {
            return Err(StreamError::MissingMeshStream);
        }
        Ok(())
    }

    /// Reads one chunk and decodes it. Returns `false` at EOF, having finished
    /// the session exactly once.
    fn pump(&mut self) -> Result<bool, StreamError> {
        if self.eof {
            return Ok(false);
        }
        let n = self.source.read(&mut self.buf)?;
        if n == 0 {
            self.eof = true;
            self.session.finish()?;
            return Ok(false);
        }
        self.pending.extend(self.session.push(&self.buf[..n])?);
        if !self.ready && self.session.has_schema() {
            // The protocol is mesh-first; a params-only stream is rejected.
            if self.session.mesh_resource_count() == 0 {
                return Err(StreamError::MissingMeshStream);
            }
            self.ready = true;
        }
        Ok(true)
    }
}

impl<R: Read> Iterator for InputStream<R> {
    type Item = Result<FrameBatch, StreamError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}
