//! Native-only Arrow streaming protocol (trd protocol 0.0.6).
//!
//! The protocol is **not backward compatible**: only `0.0.6` is accepted (see
//! `AGENTS.md`). Input is a `[mesh][texture?][frames?][params]` byte stream of
//! concatenated Arrow IPC streams on stdin:
//! a **required** leading **mesh** table (one row = one mesh, all rows decoded
//! by [`Mesh::from_arrow_all`]), an optional **texture** table (one row = one
//! `fixed_shape_tensor<u8>[H,W,4]` image, decoded by [`ImageTexture::from_arrow`]
//! and bound as the sampled albedo), then the **params** stream (one row per
//! frame: optional camera columns `model`/`k`/`pose`/`eye`/`target`/`direction`/
//! `up`/`fovy`/`aspect`/`znear`/`zfar`, an optional per-frame instanced draw list
//! `draw_mesh` (`List<UInt32>`) / `draw_model`
//! (`List<FixedSizeList<Float32>[16]>`) placing instances of the loaded meshes,
//! and an optional per-frame background `frame_path` reference). When the draw
//! list is absent, one instance of mesh 0 is placed by the frame's own `model`
//! (identity when absent). A params stream with no leading mesh table is an error
//! ([`StreamError::MissingMeshStream`]). Output: one row per frame, four
//! `fixed_shape_tensor<u8>` channels `r,g,b,a` of shape `[H, W]`.

use arrow::array::RecordBatch;
use std::io::{Read, Write};
use std::sync::Arc;

// `Matrix4` is referenced only by the `#[cfg(test)]` unit tests (imported there).
use crate::protocol::ProtocolError;
use crate::render::{
    check_dimensions, FrameParams, Mesh, RenderOptions, Renderer, TargetError, TextureTarget,
};
use crate::render::{Draw, FrameFit};
use crate::texture::ImageTexture;
use crate::OutputStream;

/// Errors from decoding, validating, rendering, or encoding a trd stream.
///
/// Each layer keeps its own error and is wrapped **transparently**, so a message
/// is identical whether it surfaces here, in `trd-wasm` (which reports
/// [`ProtocolError`] directly) or from the renderer. Only the two genuinely
/// stream-level conditions — a draw naming a mesh the stream never sent, and a
/// stream that is not mesh-first — are declared here.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    /// Decoding or validating the input protocol failed.
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
    /// Encoding the rendered image stream failed.
    #[error(transparent)]
    Output(#[from] crate::OutputError),
    /// Rendering failed, including invalid dimensions and render-target
    /// allocation ([`TargetError`] arrives through
    /// [`RenderError::Target`](crate::render::RenderError::Target)).
    #[error(transparent)]
    Render(#[from] crate::render::RenderError),
    /// I/O error reading or writing the stream.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Assembling a frame's scene failed — today, a draw naming a mesh the
    /// stream never sent.
    #[error(transparent)]
    Scene(#[from] crate::SceneError),
    /// The input is not mesh-first: the protocol requires a leading mesh table
    /// before the params stream (`[mesh][texture?][frames?][params]`). Params-only
    /// streams are no longer accepted.
    #[error("input is missing the required leading mesh table (protocol is mesh-first)")]
    MissingMeshStream,
}

/// [`TargetError`] reaches [`StreamError`] through
/// [`RenderError`](crate::render::RenderError), which already wraps it — this
/// only spares call sites an explicit hop.
impl From<TargetError> for StreamError {
    fn from(error: TargetError) -> Self {
        StreamError::Render(error.into())
    }
}

/// Decodes every row of `batch` into [`FrameParams`]. Delegates to the single
/// shared per-batch decoder [`crate::protocol::decode_batch`] (the source of
/// truth for both the native and wasm paths).
pub fn decode_frames(batch: &RecordBatch) -> Result<Vec<FrameParams>, StreamError> {
    Ok(crate::protocol::decode_batch(batch)?)
}

/// The draw list a decoded frame resolves to, with every `mesh_id` checked
/// against the uploaded mesh set — the callback path hands callers a `Vec<Draw>`
/// rather than a `Scene`, so it validates through the same rule
/// [`Scene::try_from_frame`](crate::Scene::try_from_frame) applies.
fn validated_draws(
    frame: &crate::DecodedFrame,
    mesh_count: usize,
) -> Result<Vec<Draw>, StreamError> {
    let draws = frame.resolved_draws();
    for draw in &draws {
        if draw.mesh_id as usize >= mesh_count {
            return Err(crate::SceneError::MeshIndexOutOfRange {
                mesh_id: draw.mesh_id,
                mesh_count,
            }
            .into());
        }
    }
    Ok(draws)
}

/// Reads a trd input stream **mesh-aware** — the same
/// `[mesh][texture?][frames?][params]`
/// framing [`run_stream`] uses — for a live front-end (e.g. the windowed
/// `trd-app`) that owns its own render target and encodes each frame's
/// [`Scene`](crate::Scene) itself, rather than the headless byte-stream path
/// [`run_stream`] drives.
///
/// Invokes `on_meshes` **once** with the decoded (required) leading mesh table,
/// then `on_texture` **once** with the optional bound texture (`Some` only when
/// the stream carries a texture table), then `on_meta` with the stream's declared
/// playback rate, then `on_frame` for each frame's `(FrameParams, draws)` in
/// order. A frame carrying no wire draw list defaults to one instance of mesh 0
/// placed by the frame's own model — matching [`run_stream`]. The mesh table's
/// rows are referenced by 0-based index; out-of-range `mesh_id`s are an error. A
/// params-only stream with no leading mesh table is a
/// [`StreamError::MissingMeshStream`].
pub fn read_scene_stream_with_meta<R: Read>(
    input: R,
    on_meshes: impl FnOnce(Vec<Mesh>),
    on_texture: impl FnOnce(Option<ImageTexture>),
    on_meta: impl FnOnce(f64),
    mut on_frame: impl FnMut(FrameParams, Vec<Draw>, Option<String>, Option<Arc<crate::ImageData>>),
) -> Result<(), StreamError> {
    let mut input = crate::InputStream::new(input);
    let prologue = input.prologue()?;
    let mesh_count = prologue.meshes.len();
    on_meshes(prologue.meshes.to_vec());
    on_texture(prologue.texture.cloned());
    on_meta(prologue.frame_rate);

    let mut inline_cache = crate::InlineFrameCache::default();
    while let Some(batch) = input.next_batch() {
        for frame in batch? {
            let draws = validated_draws(&frame, mesh_count)?;
            let inline = inline_cache
                .resolve(frame.frame_id, input.frames())
                .map_err(ProtocolError::from)?
                .map(|(image, _changed)| image);
            on_frame(frame.params, draws, frame.frame_ref, inline);
        }
    }
    input.finish()
}

/// A shell-provided closure that resolves a per-frame background frame reference
/// (a `frame_path`/`frame_url` string) into decoded RGBA pixels. Kept
/// out of `trd-core` so the core performs no file/network I/O: the native CLI
/// supplies one backed by the `image` crate + a `--frames-base` dir; a stream
/// without background frames (or a shell that doesn't load them) passes `None`.
/// Returning `None` for a given reference renders that frame without a
/// background plane (the shell decides how to report the miss).
pub type FrameResolver<'a> = &'a dyn Fn(&str) -> Option<crate::texture::ImageData>;

/// The **external** background reference currently uploaded, so consecutive
/// frames naming the same `frame_path`/`frame_url` skip the resolver + upload.
/// Its inline-`frame_id` counterpart is the shared
/// [`InlineFrameCache`](crate::InlineFrameCache).
#[derive(Default)]
struct FrameBackgroundState {
    last_ref: Option<String>,
}

/// Renders one decoded [`FrameBatch`](crate::FrameBatch) and writes its output
/// batch, mirroring one Arrow output batch per input record batch. When
/// `frame_resolver` is `Some`, a frame carrying a `frame_path`/`frame_url`
/// reference has its background image resolved + uploaded and composited
/// beneath the scene via the scene's [`Background::frame`](crate::Background::frame).
/// `last_frame_ref` tracks the currently uploaded background so consecutive
/// frames sharing it skip the decode + re-upload.
#[allow(clippy::too_many_arguments)]
fn render_and_write_batch<W: Write>(
    renderer: &mut Renderer,
    target: &TextureTarget,
    options: &RenderOptions,
    output: &mut OutputStream<W>,
    batch: &crate::FrameBatch,
    inline_frames: &[crate::InlineFrame],
    frame_resolver: Option<FrameResolver>,
    background_state: &mut FrameBackgroundState,
    inline_cache: &mut crate::InlineFrameCache,
) -> Result<(), StreamError> {
    let mesh_count = renderer.mesh_count();
    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(batch.len());
    for frame in batch {
        let mut frame_fit = None;
        if let Some((image, changed)) = inline_cache
            .resolve(frame.frame_id, inline_frames)
            .map_err(ProtocolError::from)?
        {
            if changed {
                renderer.update_frame_texture(&image);
            }
            background_state.last_ref = None;
            frame_fit = Some(FrameFit::Stretch);
        } else if let (Some(path), Some(resolve)) = (frame.frame_ref.as_deref(), frame_resolver) {
            if background_state.last_ref.as_deref() != Some(path) {
                if let Some(image) = resolve(path) {
                    renderer.update_frame_texture(&image);
                    background_state.last_ref = Some(path.to_owned());
                    frame_fit = Some(FrameFit::Stretch);
                }
            } else {
                frame_fit = Some(FrameFit::Stretch);
            }
        } else {
            background_state.last_ref = None;
        }
        // The scene is assembled here, from the wire draw list plus the CLI's
        // appearance options — the same `scene_with_overlays` every other
        // front-end uses, so they cannot drift apart (#180).
        let scene = crate::render::Scene::try_from_frame(frame, mesh_count, options, frame_fit)?;
        // `run_stream` is a synchronous `Read`/`Write` filter, while the renderer
        // is async because GPU read-back is (the browser must not block its event
        // loop). Natively blocking here is free: the future is already complete
        // when `poll_for_map` returns. This is the only bridge between the two.
        planes.push(pollster::block_on(renderer.render_params(
            frame.params,
            &scene,
            target,
        ))?);
    }
    // `OutputStream` owns the sink, so encoding *is* writing — no drain + hand
    // -off pair at the call site.
    output.write_rgba_batch(&planes)?;
    Ok(())
}

/// Reads a trd input stream, renders each frame, and writes an Arrow IPC stream
/// of `fixed_shape_tensor` images to `output`. Output batch boundaries mirror
/// input batches (one batch in flight).
///
/// The protocol is `[mesh][texture?][frames?][params]`: the **required**
/// leading mesh table is decoded once (via [`Mesh::from_arrow_all`]) and
/// uploaded, then an optional texture table is uploaded as the bound albedo,
/// then the following params stream drives per-frame rendering. A params-only
/// stream with no leading mesh table is a [`StreamError::MissingMeshStream`].
///
/// Framing is driven by the single shared [`InputSession`](crate::InputSession)
/// (also used by the wasm renderers): input bytes are read in chunks and pushed
/// through it, so all the mesh-first sub-stream sniffing + boundary handling
/// lives in exactly one place. The only native-specific bit is the blocking
/// [`Read`] byte source.
pub fn run_stream<R: Read, W: Write>(
    input: R,
    output: W,
    width: u32,
    height: u32,
    options: RenderOptions,
    frame_resolver: Option<FrameResolver>,
) -> Result<(), StreamError> {
    // Validate dimensions up front so schema construction (which multiplies
    // width*height) can't overflow before Renderer's guard runs.
    check_dimensions(width, height)?;

    let mut input = crate::InputStream::new(input);
    // The mesh-first prologue is complete here, so the renderer can be built
    // from it eagerly rather than lazily inside the frame loop. The renderer and
    // its texture target are a matched pair (#203): the target is a call
    // argument, not a field, so both are held here.
    let prologue = input.prologue()?;
    let frame_rate = prologue.frame_rate;
    let (mut renderer, target) = pollster::block_on(Renderer::with_meshes_sample_count(
        width,
        height,
        prologue.meshes,
        options.msaa.sample_count(),
    ))?;
    if let Some(pbr) = &options.pbr {
        renderer.set_disney_material(pbr.material.clone());
        renderer.set_image_based_lighting(pbr.ibl);
        renderer.set_tone_mapping(pbr.tone_mapping);
        if let Some(env) = &pbr.env_map {
            renderer.set_env_map(env.clone());
        }
    }
    if let Some(texture) = prologue.texture {
        renderer.set_texture(texture);
    }

    // Opening the stream writes its IPC header straight into `output`.
    let mut output = OutputStream::new(output, width, height, Some(frame_rate))?;
    // The background currently uploaded, so consecutive frames sharing it skip
    // the decode + re-upload.
    let mut background_state = FrameBackgroundState::default();
    let mut inline_cache = crate::InlineFrameCache::default();

    // `next_batch` rather than `for batch in &mut input`: the loop body needs
    // `input.frames()` too, which a `for` loop's borrow would forbid.
    while let Some(batch) = input.next_batch() {
        render_and_write_batch(
            &mut renderer,
            &target,
            &options,
            &mut output,
            &batch?,
            input.frames(),
            frame_resolver,
            &mut background_state,
            &mut inline_cache,
        )?;
    }
    input.finish()?;
    output.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests;
