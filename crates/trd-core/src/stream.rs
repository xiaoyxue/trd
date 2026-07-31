//! Native-only Arrow streaming protocol (trd protocol 0.0.5).
//!
//! The protocol is **not backward compatible**: only `0.0.5` is accepted (see
//! `AGENTS.md`). Input is a mesh-first `[mesh][texture?][params]` byte stream of
//! one to three concatenated Arrow IPC streams on stdin:
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
use arrow::datatypes::DataType;
use arrow::error::ArrowError;
use std::io::{Read, Write};

// `Schema` is only referenced by the `#[cfg(test)]` decode wrappers + unit tests.
#[cfg(test)]
use arrow::datatypes::Schema;

// `Matrix4` is referenced by `with_meshes_sample_count` (the base-model fit) and
// the `#[cfg(test)]` unit tests.
use crate::math::Matrix4;
use crate::protocol::{ProtocolError, PROTOCOL_VERSION};
use crate::render::{
    Draw, DrawableObject, FrameFit, FrameParams, GridPlane, Mesh, MeshRenderer, OffscreenError,
    OffscreenTarget, RenderMode, OFFSCREEN_FORMAT,
};
use crate::texture::ImageTexture;
use crate::OutputSession;

/// Errors from decoding, validating, rendering, or encoding a trd stream.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    /// An underlying Arrow or IPC error.
    #[error("arrow error: {0}")]
    Arrow(#[from] ArrowError),
    /// I/O error reading or writing the stream.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// A required input column is missing.
    #[error("input is missing required column `{0}`")]
    MissingColumn(&'static str),
    /// A required input column has the wrong Arrow type.
    #[error("input column `{column}` has type {actual:?}, expected {expected}")]
    ColumnType {
        column: &'static str,
        expected: &'static str,
        actual: DataType,
    },
    /// A required input column contains null values (protocol requires non-null).
    #[error("input column `{0}` contains null values")]
    NullValues(&'static str),
    /// The stream mixes the CV (`k`/`pose`) and CG (`eye`/`target`/`direction`/
    /// `fovy`) camera forms; exactly one must be used.
    #[error(
        "conflicting camera forms: use either CV (`k`/`pose`) or CG \
         (`eye`/`target`/`direction`/`fovy`), not both"
    )]
    ConflictingCameraForms,
    /// The CG camera form is incomplete (an `eye` without a look
    /// `target`/`direction`, or vice versa).
    #[error("incomplete CG camera: `eye` requires a look `target`/`direction` (and vice versa)")]
    IncompleteCameraForm,
    /// A frame's per-instance `draw_mesh` and `draw_model` lists differ in
    /// length; each drawn instance needs exactly one mesh id and one model.
    #[error(
        "frame {row}: draw_mesh has {mesh_len} entries but draw_model has {model_len} \
         (each instance needs one mesh id and one model)"
    )]
    MismatchedDrawLists {
        row: usize,
        mesh_len: usize,
        model_len: usize,
    },
    /// A frame's per-instance `draw_mode` list differs in length from its
    /// `draw_mesh`/`draw_model` lists; each drawn instance needs exactly one
    /// mode byte when the column is present.
    #[error(
        "frame {row}: draw_mode has {mode_len} entries but there are {draw_len} \
         draw(s) (each instance needs one mode byte)"
    )]
    MismatchedDrawModes {
        row: usize,
        mode_len: usize,
        draw_len: usize,
    },
    /// A `draw_mode` byte is not a recognized [`crate::RenderMode`] encoding
    /// (`0`=filled, `1`=wireframe, `2`=textured, `255`=inherit global).
    #[error("draw_mode byte {value} is not a valid render mode (0/1/2/255)")]
    InvalidDrawMode { value: u8 },
    /// A draw references a mesh index outside the uploaded mesh set.
    #[error("draw references mesh index {mesh_id} but only {mesh_count} mesh(es) are loaded")]
    MeshIndexOutOfRange { mesh_id: u32, mesh_count: usize },
    /// The requested image dimensions are invalid or too large.
    #[error("invalid image dimensions {width}x{height}: {reason}")]
    InvalidDimensions {
        width: u32,
        height: u32,
        reason: &'static str,
    },
    /// The stream declares a protocol version this build does not support.
    #[error("unsupported protocol version `{0}` (expected `{PROTOCOL_VERSION}`)")]
    UnsupportedVersion(String),
    /// The input is not mesh-first: the protocol requires a leading mesh table
    /// before the params stream (`[mesh][texture?][params]`). Legacy params-only
    /// streams are no longer accepted.
    #[error("input is missing the required leading mesh table (protocol is mesh-first)")]
    MissingMeshStream,
    /// GPU rendering failed.
    #[error("render error: {0}")]
    Render(String),
    /// The leading mesh table could not be decoded into a [`Mesh`].
    #[error("mesh decode error: {0}")]
    Mesh(#[from] crate::MeshError),
    /// The optional leading texture table could not be decoded.
    #[error("texture decode error: {0}")]
    Texture(#[from] crate::TextureError),
    #[error(transparent)]
    Output(#[from] crate::OutputError),
}

/// Maps the shared [`ProtocolError`] (from the single decoder in
/// [`crate::protocol`]) onto this module's [`StreamError`], so the native
/// `run_stream` path keeps its flat error surface while the per-batch decode
/// logic lives in exactly one place.
impl From<ProtocolError> for StreamError {
    fn from(error: ProtocolError) -> Self {
        match error {
            ProtocolError::Arrow(e) => StreamError::Arrow(e),
            ProtocolError::MissingColumn(c) => StreamError::MissingColumn(c),
            ProtocolError::ColumnType {
                column,
                expected,
                actual,
            } => StreamError::ColumnType {
                column,
                expected,
                actual,
            },
            ProtocolError::NullValues(c) => StreamError::NullValues(c),
            ProtocolError::ConflictingCameraForms => StreamError::ConflictingCameraForms,
            ProtocolError::IncompleteCameraForm => StreamError::IncompleteCameraForm,
            ProtocolError::UnsupportedVersion(v) => StreamError::UnsupportedVersion(v),
            ProtocolError::Mesh(e) => StreamError::Mesh(e),
            ProtocolError::Texture(e) => StreamError::Texture(e),
            ProtocolError::MismatchedDrawLists {
                row,
                mesh_len,
                model_len,
            } => StreamError::MismatchedDrawLists {
                row,
                mesh_len,
                model_len,
            },
            ProtocolError::MismatchedDrawModes {
                row,
                mode_len,
                draw_len,
            } => StreamError::MismatchedDrawModes {
                row,
                mode_len,
                draw_len,
            },
            ProtocolError::InvalidDrawMode { value } => StreamError::InvalidDrawMode { value },
            // The session-framing errors can't arise from the per-batch decoders
            // used by `run_stream`; surface them as a generic render error if they
            // ever do.
            other @ (ProtocolError::SessionFinished
            | ProtocolError::SessionFailed
            | ProtocolError::MissingSchema
            | ProtocolError::NoProgress) => StreamError::Render(other.to_string()),
        }
    }
}

/// Maps the shared offscreen render harness's [`OffscreenError`] onto this
/// module's [`StreamError`], so [`BatchRenderer`] keeps its flat error surface
/// while the target/readback plumbing lives in [`crate::render::OffscreenTarget`].
impl From<OffscreenError> for StreamError {
    fn from(error: OffscreenError) -> Self {
        match error {
            OffscreenError::InvalidDimensions { width, height } => StreamError::InvalidDimensions {
                width,
                height,
                reason: "dimensions must be non-zero",
            },
            OffscreenError::ExceedsMaxDimension { width, height, .. } => {
                StreamError::InvalidDimensions {
                    width,
                    height,
                    reason: "exceeds adapter max_texture_dimension_2d",
                }
            }
            OffscreenError::RowOverflow { .. } | OffscreenError::Gpu(_) => {
                StreamError::Render(error.to_string())
            }
            OffscreenError::Output(e) => StreamError::Output(e),
        }
    }
}

/// If the schema declares a protocol version, require it to be supported.
/// Delegates to the shared [`crate::protocol::check_version`]. A test-only
/// [`StreamError`]-typed wrapper: `run_stream` now validates the version inside
/// the shared [`InputSession`](crate::InputSession), so this only exists to
/// exercise the [`ProtocolError`] → [`StreamError`] mapping in unit tests.
#[cfg(test)]
pub fn check_version(schema: &Schema) -> Result<(), StreamError> {
    Ok(crate::protocol::check_version(schema)?)
}

/// Decodes every row of `batch` into [`FrameParams`]. Delegates to the single
/// shared per-batch decoder [`crate::protocol::decode_batch`] (the source of
/// truth for both the native and wasm paths).
pub fn decode_frames(batch: &RecordBatch) -> Result<Vec<FrameParams>, StreamError> {
    Ok(crate::protocol::decode_batch(batch)?)
}

/// Decodes the optional per-frame **instanced draw list** columns into one
/// `Vec<Draw>` per row. A test-only [`StreamError`]-typed wrapper over
/// [`crate::protocol::decode_draws`] — the native/wasm paths use the shared
/// [`InputSession`](crate::InputSession) decoder directly; this only exercises
/// the [`ProtocolError`] → [`StreamError`] mapping in unit tests.
#[cfg(test)]
fn decode_draws(batch: &RecordBatch) -> Result<Option<Vec<Vec<Draw>>>, StreamError> {
    Ok(crate::protocol::decode_draws(batch)?)
}

/// Decodes the optional per-frame **background frame reference** column (`0.0.5`)
/// into one `Option<String>` per row. A test-only [`StreamError`]-typed wrapper
/// over [`crate::protocol::decode_frame_refs`], exercising the [`ProtocolError`]
/// → [`StreamError`] mapping in unit tests.
#[cfg(test)]
fn decode_frame_refs(batch: &RecordBatch) -> Result<Option<Vec<Option<String>>>, StreamError> {
    Ok(crate::protocol::decode_frame_refs(batch)?)
}

const BYTES_PER_PIXEL: u32 = 4;

/// Validates image dimensions against zero and `u32` overflow, returning the
/// pixel count. Does not check device limits (that needs an adapter). Called
/// before any `width * height` / `width * 4` arithmetic so absurd dimensions
/// produce a clean [`StreamError::InvalidDimensions`] instead of an overflow.
fn check_dimensions(width: u32, height: u32) -> Result<u32, StreamError> {
    let pixels = width
        .checked_mul(height)
        .filter(|&p| p > 0 && p <= i32::MAX as u32)
        .ok_or(StreamError::InvalidDimensions {
            width,
            height,
            reason: "width*height must be non-zero and <= i32::MAX",
        })?;
    width
        .checked_mul(BYTES_PER_PIXEL)
        .ok_or(StreamError::InvalidDimensions {
            width,
            height,
            reason: "row byte size overflows u32",
        })?;
    Ok(pixels)
}

/// A persistent GPU context that renders one [`FrameParams`] to tightly-packed
/// row-major RGBA bytes (`width*height*4`) per call.
pub struct BatchRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: MeshRenderer,
    /// The shared offscreen render target + readback buffer (#103, Part B).
    target: OffscreenTarget,
    /// Render mode (filled/wireframe) applied to every mesh drawable this
    /// renderer builds into its per-frame [`Scene`](crate::Scene).
    mode: RenderMode,
    /// Whether to add a [`DrawableObject::AabbBox`] gizmo per drawn instance.
    show_aabb: bool,
    /// Whether to add a single origin [`DrawableObject::CoordinateAxes`] gizmo.
    show_axes: bool,
    /// Whether to add a [`DrawableObject::CoordinateAxes`] at *each* drawn
    /// instance's own `model` — the object's local coordinate frame.
    show_local_axes: bool,
    /// If `Some(plane)`, add a [`DrawableObject::PlaneGrid`] on that coordinate
    /// plane at *each* drawn instance's own `model` — a grid lattice in the
    /// object's local frame (e.g. an `xy` grid tiling a placement quad).
    show_local_grid: Option<GridPlane>,
    /// If `Some(id)`, narrow the [`show_local_grid`](Self::show_local_grid)
    /// overlay to draws of that `mesh_id` only (the placement quad), so a
    /// wireframe *content* mesh doesn't also pick up a floor grid (#114).
    show_local_grid_mesh: Option<u32>,
}

impl BatchRenderer {
    /// Builds the GPU context (instance/adapter/device/pipeline/target/readback)
    /// once for a fixed `width` x `height`, rendering the `meshes` of the stream's
    /// leading mesh table, applying each mesh's [`Mesh::preview_transform`]
    /// (center + uniform scale-to-fit) beneath its per-frame model so an
    /// arbitrary-unit asset renders centered and at a reasonable size. Per-frame
    /// draw lists place instances of these meshes by index. The mesh pass renders
    /// at 4× MSAA; use [`with_meshes_sample_count`](Self::with_meshes_sample_count)
    /// to override (e.g. `1` = no MSAA).
    pub fn with_meshes(width: u32, height: u32, meshes: &[Mesh]) -> Result<Self, StreamError> {
        Self::with_meshes_sample_count(width, height, meshes, crate::render::MSAA_SAMPLE_COUNT)
    }

    /// Like [`with_meshes`](Self::with_meshes) but with an explicit mesh-pass MSAA
    /// `sample_count` (`4` = anti-aliased, `1` = single-sampled / no MSAA).
    pub fn with_meshes_sample_count(
        width: u32,
        height: u32,
        meshes: &[Mesh],
        sample_count: u32,
    ) -> Result<Self, StreamError> {
        let base_models: Vec<Matrix4> = meshes
            .iter()
            .map(|mesh| {
                mesh.preview_transform(crate::DEFAULT_PREVIEW_TARGET)
                    .matrix()
            })
            .collect();
        pollster::block_on(Self::new_async(
            width,
            height,
            meshes,
            &base_models,
            sample_count,
        ))
    }

    async fn new_async(
        width: u32,
        height: u32,
        meshes: &[Mesh],
        base_models: &[Matrix4],
        sample_count: u32,
    ) -> Result<Self, StreamError> {
        // Guard against zero / overflow before allocating (device limits below).
        check_dimensions(width, height)?;

        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| StreamError::Render(e.to_string()))?;
        let info = adapter.get_info();
        log::info!(
            "using {:?} adapter \"{}\" ({:?})",
            info.backend,
            info.name,
            info.device_type
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("trd device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| StreamError::Render(e.to_string()))?;

        let format = OFFSCREEN_FORMAT;
        let renderer =
            MeshRenderer::with_sample_count(&device, format, meshes, base_models, sample_count);

        // The shared offscreen harness owns the render target + readback buffer
        // and re-validates the size against the adapter's max dimension.
        let target = OffscreenTarget::new(&device, width, height)?;

        Ok(Self {
            device,
            queue,
            renderer,
            target,
            mode: RenderMode::Filled,
            show_aabb: false,
            show_axes: false,
            show_local_axes: false,
            show_local_grid: None,
            show_local_grid_mesh: None,
        })
    }

    /// The number of loaded meshes; valid [`Draw::mesh_id`]s are `0..mesh_count`.
    pub fn mesh_count(&self) -> usize {
        self.renderer.mesh_count()
    }

    /// Sets the [`RenderMode`] (filled or wireframe) applied to later `render`s.
    pub fn set_mode(&mut self, mode: RenderMode) {
        self.mode = mode;
    }

    /// Binds `texture` as the source sampled by [`RenderMode::Textured`] meshes
    /// (`0.0.4`). Delegates to [`MeshRenderer::set_texture`]; the image is
    /// (re)uploaded on the next `render`.
    pub fn set_texture(&mut self, texture: &dyn crate::texture::Texture) {
        self.renderer.set_texture(texture);
    }

    /// Sets the Disney [`PbrMaterial`](crate::PbrMaterial) applied globally to
    /// [`RenderMode::Pbr`] meshes. Delegates to [`MeshRenderer::set_pbr_material`].
    pub fn set_pbr_material(&mut self, material: crate::PbrMaterial) {
        self.renderer.set_pbr_material(material);
    }

    /// Binds `env` as the equirectangular HDR environment map reflected by
    /// [`RenderMode::Pbr`] meshes. Delegates to [`MeshRenderer::set_env_map`]; the
    /// probe is (re)uploaded on the next `render`.
    pub fn set_env_map(&mut self, env: crate::EnvMapData) {
        self.renderer.set_env_map(env);
    }

    /// Enables/disables the per-instance AABB overlay box: when on, each drawn
    /// instance also contributes a [`DrawableObject::AabbBox`] to the scene.
    pub fn set_show_aabb(&mut self, show: bool) {
        self.show_aabb = show;
    }

    /// Enables/disables the origin coordinate-axes overlay gizmo: when on, the
    /// scene gains a single [`DrawableObject::CoordinateAxes`] at the world
    /// origin.
    pub fn set_show_axes(&mut self, show: bool) {
        self.show_axes = show;
    }

    /// Enables/disables the per-instance *local* coordinate-axes overlay: when
    /// on, each drawn instance also gains a [`DrawableObject::CoordinateAxes`]
    /// placed by its own `model`, visualizing that object's local frame (e.g.
    /// #77's `(e1,e2,e3)` quad placement).
    pub fn set_show_local_axes(&mut self, show: bool) {
        self.show_local_axes = show;
    }

    /// Selects the per-instance *local* coordinate-plane grid overlay: when
    /// `Some(plane)`, each drawn instance also gains a
    /// [`DrawableObject::PlaneGrid`] on that plane placed by its own `model`,
    /// laying a grid lattice across the object's local frame (e.g. an `xy` grid
    /// tiling a placement quad's floor). `None` disables it.
    pub fn set_show_local_grid(&mut self, plane: Option<GridPlane>) {
        self.show_local_grid = plane;
    }

    /// Narrows the [`set_show_local_grid`](Self::set_show_local_grid) overlay to
    /// draws of a single `mesh_id` (the placement quad). `Some(id)` lays the grid
    /// only under that mesh — so a *content* mesh drawn wireframe (e.g. a
    /// wireframe-reveal intro) doesn't also pick up a floor grid; `None` keeps the
    /// grid on every wireframe draw (#114).
    pub fn set_show_local_grid_mesh(&mut self, mesh: Option<u32>) {
        self.show_local_grid_mesh = mesh;
    }

    /// Uploads `image` as the **background frame texture** (#63) sampled by a
    /// [`DrawableObject::FramePlane`]. The GPU texture is reused across frames
    /// (grown only on a resolution change). Call before a
    /// [`render_frame`](Self::render_frame) with a `Some(fit)` to composite the
    /// image beneath the mesh scene.
    pub fn update_frame_texture(&mut self, image: &crate::texture::ImageData) {
        self.renderer.update_frame_texture_rgba(
            &self.queue,
            &image.rgba,
            image.width,
            image.height,
        );
    }

    /// Builds the per-frame [`Scene`](crate::Scene) from a wire `draws` list and
    /// this renderer's mode/overlay flags (delegates to [`build_scene`]). A
    /// `Some(fit)` prepends a background [`DrawableObject::FramePlane`] (#63).
    fn build_scene(&self, draws: &[Draw], frame: Option<FrameFit>) -> Vec<DrawableObject> {
        crate::render::build_scene(
            draws,
            self.mode,
            self.show_aabb,
            self.show_axes,
            self.show_local_axes,
            self.show_local_grid,
            self.show_local_grid_mesh,
            frame,
        )
    }

    /// Renders `params` with the given per-frame instance `draws`, compositing a
    /// background [`DrawableObject::FramePlane`] (#63) beneath the scene when
    /// `frame` is `Some(fit)` and a frame texture has been uploaded via
    /// [`update_frame_texture`](Self::update_frame_texture). Returns
    /// tightly-packed row-major RGBA bytes (`width*height*4`).
    pub fn render_frame(
        &mut self,
        params: FrameParams,
        draws: &[Draw],
        frame: Option<FrameFit>,
    ) -> Result<Vec<u8>, StreamError> {
        let scene = self.build_scene(draws, frame);
        Ok(pollster::block_on(self.target.render(
            &self.device,
            &self.queue,
            &mut self.renderer,
            params,
            &scene,
        ))?)
    }
}

/// Resolves one decoded frame's instanced draw list: its wire `draws` when
/// present (an explicit empty list ⇒ no meshes, so just the background plate),
/// else one instance of mesh 0 placed by the frame's own model (legacy
/// single-object behavior) — see [`DecodedFrame::resolved_draws`]. Every
/// referenced `mesh_id` is validated against `mesh_count`. Shared by the headless
/// [`run_stream`] path and the live [`read_scene_stream_with_meta`] front-end so
/// both resolve draws identically.
fn resolve_frame_draws(
    frame: &crate::DecodedFrame,
    mesh_count: usize,
) -> Result<Vec<Draw>, StreamError> {
    let draws = frame.resolved_draws();
    for draw in &draws {
        if draw.mesh_id as usize >= mesh_count {
            return Err(StreamError::MeshIndexOutOfRange {
                mesh_id: draw.mesh_id,
                mesh_count,
            });
        }
    }
    Ok(draws)
}

/// Reads a trd input stream **mesh-aware** — the same `[mesh][texture?][params]`
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
    mut input: R,
    on_meshes: impl FnOnce(Vec<Mesh>),
    on_texture: impl FnOnce(Option<ImageTexture>),
    on_meta: impl FnOnce(f64),
    mut on_frame: impl FnMut(FrameParams, Vec<Draw>, Option<String>),
) -> Result<(), StreamError> {
    let mut session = crate::InputSession::new();
    // FnOnce callbacks fired exactly once, when the params schema is first
    // reached (meshes + texture + fps complete); `Option::take` moves each out on
    // that single iteration so the borrow checker accepts calling them in a loop.
    let mut on_meshes = Some(on_meshes);
    let mut on_texture = Some(on_texture);
    let mut on_meta = Some(on_meta);
    let mut mesh_count = 0usize;
    let mut ready = false;

    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            break;
        }
        let batches = session.push(&buf[..n])?;

        if !ready && session.has_schema() {
            if session.meshes().is_empty() {
                // The protocol is mesh-first; a params-only stream is rejected.
                return Err(StreamError::MissingMeshStream);
            }
            mesh_count = session.meshes().len();
            if let Some(cb) = on_meshes.take() {
                cb(session.meshes().to_vec());
            }
            if let Some(cb) = on_texture.take() {
                cb(session.texture().cloned());
            }
            if let Some(cb) = on_meta.take() {
                cb(session.frame_rate().unwrap_or(crate::DEFAULT_FRAME_RATE));
            }
            ready = true;
        }

        for batch in batches {
            for frame in batch {
                let draws = resolve_frame_draws(&frame, mesh_count)?;
                on_frame(frame.params, draws, frame.frame_ref);
            }
        }
    }
    session.finish()?;

    if !ready {
        // No params schema was ever reached (empty input) — mesh-first unmet.
        return Err(StreamError::MissingMeshStream);
    }
    Ok(())
}

/// A shell-provided closure that resolves a per-frame background frame reference
/// (a `frame_path`/`frame_url` string, `0.0.5`) into decoded RGBA pixels. Kept
/// out of `trd-core` so the core performs no file/network I/O: the native CLI
/// supplies one backed by the `image` crate + a `--frames-base` dir; a stream
/// without background frames (or a shell that doesn't load them) passes `None`.
/// Returning `None` for a given reference renders that frame without a
/// background plane (the shell decides how to report the miss).
pub type FrameResolver<'a> = &'a dyn Fn(&str) -> Option<crate::texture::ImageData>;

/// Renders one decoded [`FrameBatch`](crate::FrameBatch) and writes its output
/// batch, mirroring one Arrow output batch per input record batch. When
/// `frame_resolver` is `Some`, a frame carrying a `frame_path`/`frame_url`
/// reference (`0.0.5`) has its background image resolved + uploaded and composited
/// beneath the scene via a [`DrawableObject`](crate::render::DrawableObject)`::FramePlane`.
/// `last_frame_ref` tracks the currently uploaded background so consecutive
/// frames sharing it skip the decode + re-upload.
fn render_and_write_batch<W: Write>(
    renderer: &mut BatchRenderer,
    output_session: &mut OutputSession,
    batch: &crate::FrameBatch,
    frame_resolver: Option<FrameResolver>,
    last_frame_ref: &mut Option<String>,
    output: &mut W,
) -> Result<(), StreamError> {
    let mesh_count = renderer.mesh_count();
    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(batch.len());
    for frame in batch {
        let draws = resolve_frame_draws(frame, mesh_count)?;
        let mut frame_fit = None;
        if let (Some(path), Some(resolve)) = (frame.frame_ref.as_deref(), frame_resolver) {
            if last_frame_ref.as_deref() != Some(path) {
                if let Some(image) = resolve(path) {
                    renderer.update_frame_texture(&image);
                    *last_frame_ref = Some(path.to_owned());
                    frame_fit = Some(FrameFit::Stretch);
                }
            } else {
                frame_fit = Some(FrameFit::Stretch);
            }
        }
        planes.push(renderer.render_frame(frame.params, &draws, frame_fit)?);
    }
    output_session.write_rgba_batch(&planes)?;
    output.write_all(&output_session.drain_new()?)?;
    Ok(())
}

/// The Disney PBR configuration threaded through [`RenderOptions`]: the global
/// [`PbrMaterial`](crate::PbrMaterial) plus an optional equirectangular HDR
/// environment map. When present (and the mode is [`RenderMode::Pbr`]), meshes
/// are shaded with the physically-based `disney.wgsl` path.
#[derive(Debug, Clone, Default)]
pub struct PbrConfig {
    /// The Disney material applied to every PBR mesh.
    pub material: crate::PbrMaterial,
    /// The HDR environment probe reflected by metallic surfaces (`None` ⇒ no
    /// environment reflection).
    pub env_map: Option<crate::EnvMapData>,
}

/// The mesh-pass multisample anti-aliasing setting threaded through
/// [`RenderOptions`]. [`Msaa::X4`] (the default) renders the 4×-multisampled mesh
/// pass — smooth wireframe / gizmo / AABB / silhouette edges; [`Msaa::Off`]
/// renders single-sampled (aliased edges, the raw rasterized coverage). Both are
/// covered by the golden-render test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Msaa {
    /// 4× multisampling — the default anti-aliased mesh pass.
    #[default]
    X4,
    /// No multisampling: render the mesh pass single-sampled (aliased edges).
    Off,
}

impl Msaa {
    /// The wgpu sample count for this setting (`4` for [`Msaa::X4`], `1` for
    /// [`Msaa::Off`]).
    pub(crate) fn sample_count(self) -> u32 {
        match self {
            Msaa::X4 => crate::render::MSAA_SAMPLE_COUNT,
            Msaa::Off => 1,
        }
    }
}

/// Appearance options for [`run_stream`]: the mesh draw [`RenderMode`] plus the
/// optional AABB / coordinate-axes gizmo overlays. Bundled into one value so the
/// entry point threads a single struct instead of many positional flags (and
/// stays within clippy's argument budget). [`Default`] is filled, no overlays,
/// 4× MSAA.
#[derive(Debug, Clone, Default)]
pub struct RenderOptions {
    /// How meshes are drawn (filled / wireframe / textured / PBR).
    pub mode: RenderMode,
    /// Overlay each drawn mesh instance's axis-aligned bounding box (#42).
    pub show_aabb: bool,
    /// Overlay a world-origin coordinate-axes gizmo (#42).
    pub show_axes: bool,
    /// Overlay a coordinate-axes gizmo at *each* drawn object's local (model)
    /// frame — its model-space X/Y/Z axes as placed (e.g. #77's `(e1,e2,e3)`).
    pub show_local_axes: bool,
    /// Overlay a coordinate-plane grid lattice on the given plane at *each*
    /// drawn object's local (model) frame — e.g. `Some(GridPlane::Xy)` tiles a
    /// grid across a placement quad's local floor. `None` disables it.
    pub show_local_grid: Option<GridPlane>,
    /// Narrows [`show_local_grid`](Self::show_local_grid) to draws of a single
    /// `mesh_id` (the placement quad), so a wireframe *content* mesh doesn't also
    /// pick up a floor grid. `None` keeps the grid on every wireframe draw (#114).
    pub show_local_grid_mesh: Option<u32>,
    /// Disney PBR material + environment map, applied when `mode` is
    /// [`RenderMode::Pbr`] (also honoured for any per-draw PBR-mode draws).
    pub pbr: Option<PbrConfig>,
    /// Mesh-pass multisample anti-aliasing (default [`Msaa::X4`]).
    pub msaa: Msaa,
}

/// Reads a trd input stream, renders each frame, and writes an Arrow IPC stream
/// of `fixed_shape_tensor` images to `output`. Output batch boundaries mirror
/// input batches (one batch in flight).
///
/// The protocol is mesh-first `[mesh][texture?][params]`: the **required**
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
    mut input: R,
    mut output: W,
    width: u32,
    height: u32,
    options: RenderOptions,
    frame_resolver: Option<FrameResolver>,
) -> Result<(), StreamError> {
    // Validate dimensions up front so schema construction (which multiplies
    // width*height) can't overflow before BatchRenderer's guard runs.
    check_dimensions(width, height)?;

    let mut session = crate::InputSession::new();
    // Built once the params schema is reached (meshes + texture + fps known).
    let mut renderer: Option<BatchRenderer> = None;
    let mut output_session: Option<OutputSession> = None;
    // The background currently uploaded, so consecutive frames sharing it skip
    // the decode + re-upload.
    let mut last_frame_ref: Option<String> = None;

    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            break;
        }
        let batches = session.push(&buf[..n])?;

        // The mesh-first protocol delivers meshes + optional texture before the
        // params schema, so `has_schema()` flips true only once they're complete.
        if renderer.is_none() && session.has_schema() {
            if session.meshes().is_empty() {
                return Err(StreamError::MissingMeshStream);
            }
            let mut built = BatchRenderer::with_meshes_sample_count(
                width,
                height,
                session.meshes(),
                options.msaa.sample_count(),
            )?;
            built.set_mode(options.mode);
            built.set_show_aabb(options.show_aabb);
            built.set_show_axes(options.show_axes);
            built.set_show_local_axes(options.show_local_axes);
            built.set_show_local_grid(options.show_local_grid);
            built.set_show_local_grid_mesh(options.show_local_grid_mesh);
            if let Some(pbr) = &options.pbr {
                built.set_pbr_material(pbr.material);
                if let Some(env) = &pbr.env_map {
                    built.set_env_map(env.clone());
                }
            }
            if let Some(texture) = session.texture() {
                built.set_texture(texture);
            }
            renderer = Some(built);

            let frame_rate = session.frame_rate().unwrap_or(crate::DEFAULT_FRAME_RATE);
            let mut session_out = OutputSession::with_frame_rate(width, height, Some(frame_rate))?;
            output.write_all(&session_out.drain_new()?)?;
            output_session = Some(session_out);
        }

        if let (Some(renderer), Some(output_session)) = (renderer.as_mut(), output_session.as_mut())
        {
            for batch in &batches {
                render_and_write_batch(
                    renderer,
                    output_session,
                    batch,
                    frame_resolver,
                    &mut last_frame_ref,
                    &mut output,
                )?;
            }
        }
    }
    session.finish()?;

    // A stream that never reached a params schema (empty input) — the mesh-first
    // contract wasn't satisfied.
    let mut output_session = output_session.ok_or(StreamError::MissingMeshStream)?;
    output_session.finish()?;
    output.write_all(&output_session.drain_new()?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::PROTOCOL_VERSION_KEY;
    use crate::render::build_scene;
    use arrow::array::{
        Array, ArrayRef, FixedSizeListArray, FixedSizeListArray as U8List, Float32Array, ListArray,
        StringArray, UInt32Array, UInt8Array,
    };
    use arrow::datatypes::Field;
    use arrow::ipc::reader::StreamReader;
    use arrow::ipc::writer::StreamWriter;
    use std::sync::Arc;

    fn build_input_batch(frames: &[FrameParams]) -> RecordBatch {
        // A minimal 0.0.5 params batch carries a single `model` column; every
        // params column is optional, and `model` alone drives the row count.
        let flat: Vec<f32> = frames
            .iter()
            .flat_map(|f| f.model.unwrap_or(IDENTITY_MODEL))
            .collect();
        let schema = Arc::new(
            Schema::new(vec![model_field()]).with_metadata(
                [(
                    PROTOCOL_VERSION_KEY.to_string(),
                    PROTOCOL_VERSION.to_string(),
                )]
                .into_iter()
                .collect(),
            ),
        );
        RecordBatch::try_new(schema, vec![list_col(16, flat)]).unwrap()
    }

    /// Column-major identity 4×4, the default `model` for the test helpers.
    const IDENTITY_MODEL: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    /// The `model` column `Field` (`FixedSizeList<Float32>[16]`).
    fn model_field() -> Field {
        Field::new(
            "model",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 16),
            false,
        )
    }

    /// A `model` column of `n` identity matrices.
    fn model_col(n: usize) -> ArrayRef {
        list_col(16, (0..n).flat_map(|_| IDENTITY_MODEL).collect())
    }

    #[test]
    fn decodes_frames_roundtrip() {
        let frames = vec![
            FrameParams {
                model: Some(IDENTITY_MODEL),
                ..FrameParams::IDENTITY
            },
            FrameParams {
                model: Some([
                    0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1, -0.2, 0.0, 1.0,
                ]),
                ..FrameParams::IDENTITY
            },
        ];
        let batch = build_input_batch(&frames);
        let decoded = decode_frames(&batch).unwrap();
        assert_eq!(decoded, frames);
    }

    /// A non-null `FixedSizeList<Float32>[len]` column from flat values.
    fn list_col(len: i32, flat: Vec<f32>) -> ArrayRef {
        Arc::new(FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, false)),
            len,
            Arc::new(Float32Array::from(flat)),
            None,
        )) as ArrayRef
    }

    /// Builds a one-row batch of an identity `model` plus the given extra
    /// `(field, column)` pairs.
    fn camera_batch(extra: Vec<(Field, ArrayRef)>) -> RecordBatch {
        let mut fields = vec![model_field()];
        let mut columns: Vec<ArrayRef> = vec![model_col(1)];
        for (field, column) in extra {
            fields.push(field);
            columns.push(column);
        }
        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, columns).unwrap()
    }

    #[test]
    fn decodes_cg_camera_columns() {
        let list3 = |name| {
            Field::new(
                name,
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 3),
                false,
            )
        };
        let batch = camera_batch(vec![
            (list3("eye"), list_col(3, vec![1.0, 2.0, 3.0])),
            (list3("target"), list_col(3, vec![0.1, 0.2, 0.3])),
            (
                Field::new("fovy", DataType::Float32, false),
                Arc::new(Float32Array::from(vec![0.9_f32])) as ArrayRef,
            ),
        ]);
        let frames = decode_frames(&batch).unwrap();
        assert_eq!(frames[0].eye, Some([1.0, 2.0, 3.0]));
        assert_eq!(frames[0].target, Some([0.1, 0.2, 0.3]));
        assert_eq!(frames[0].fovy, Some(0.9));
    }

    #[test]
    fn rejects_incomplete_and_conflicting_camera_forms() {
        let list_field = |name, len| {
            Field::new(
                name,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    len,
                ),
                false,
            )
        };
        // `eye` alone is incomplete.
        let incomplete = camera_batch(vec![(
            list_field("eye", 3),
            list_col(3, vec![1.0, 2.0, 3.0]),
        )]);
        assert!(matches!(
            decode_frames(&incomplete),
            Err(StreamError::IncompleteCameraForm)
        ));
        // CV `k` mixed with CG `eye` is conflicting.
        let conflicting = camera_batch(vec![
            (list_field("k", 9), list_col(9, vec![1.0; 9])),
            (list_field("eye", 3), list_col(3, vec![1.0, 2.0, 3.0])),
        ]);
        assert!(matches!(
            decode_frames(&conflicting),
            Err(StreamError::ConflictingCameraForms)
        ));
    }

    use arrow::buffer::OffsetBuffer;

    /// A `List<UInt32>` column with the given per-row id lists.
    fn draw_mesh_col(rows: &[Vec<u32>]) -> ArrayRef {
        let field = Arc::new(Field::new("item", DataType::UInt32, false));
        let flat: Vec<u32> = rows.iter().flatten().copied().collect();
        let offsets = OffsetBuffer::from_lengths(rows.iter().map(Vec::len));
        Arc::new(ListArray::new(
            field,
            offsets,
            Arc::new(UInt32Array::from(flat)),
            None,
        )) as ArrayRef
    }

    /// A `List<FixedSizeList<Float32>[16]>` column with the given per-row model
    /// lists (each model is 16 flat column-major floats).
    fn draw_model_col(rows: &[Vec<[f32; 16]>]) -> ArrayRef {
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let flat: Vec<f32> = rows.iter().flatten().flatten().copied().collect();
        let fsl = FixedSizeListArray::new(item, 16, Arc::new(Float32Array::from(flat)), None);
        let field = Arc::new(Field::new(
            "item",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 16),
            false,
        ));
        let offsets = OffsetBuffer::from_lengths(rows.iter().map(Vec::len));
        Arc::new(ListArray::new(field, offsets, Arc::new(fsl), None)) as ArrayRef
    }

    /// The `Field` for a `draw_mesh` / `draw_model` column.
    fn draw_field(name: &str, item: DataType) -> Field {
        Field::new(
            name,
            DataType::List(Arc::new(Field::new("item", item, false))),
            false,
        )
    }

    fn draw_batch(mesh_rows: &[Vec<u32>], model_rows: &[Vec<[f32; 16]>]) -> RecordBatch {
        let n = mesh_rows.len();
        let schema = Arc::new(Schema::new(vec![
            model_field(),
            draw_field("draw_mesh", DataType::UInt32),
            draw_field(
                "draw_model",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 16),
            ),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                model_col(n),
                draw_mesh_col(mesh_rows),
                draw_model_col(model_rows),
            ],
        )
        .unwrap()
    }

    #[test]
    fn decode_draws_absent_returns_none() {
        let batch = build_input_batch(&[FrameParams::IDENTITY]);
        assert!(decode_draws(&batch).unwrap().is_none());
    }

    /// A `Utf8` column of `frame_path`/`frame_url` references from optional strings.
    fn frame_ref_batch(name: &str, refs: &[Option<&str>]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Utf8, true)]));
        let col = StringArray::from(refs.to_vec());
        RecordBatch::try_new(schema, vec![Arc::new(col) as ArrayRef]).unwrap()
    }

    #[test]
    fn decode_frame_refs_absent_returns_none() {
        // A stream with no background-frame column decodes to `None` (soft skip).
        let batch = build_input_batch(&[FrameParams::IDENTITY]);
        assert!(decode_frame_refs(&batch).unwrap().is_none());
    }

    #[test]
    fn decode_frame_refs_reads_paths_nulls_and_empty() {
        // Native prefers `frame_path`; per-row null or empty ⇒ `None` (no
        // background for that frame), a non-empty string ⇒ the reference.
        let batch = frame_ref_batch(
            "frame_path",
            &[Some("frames/frame_000000.png"), None, Some("")],
        );
        let refs = decode_frame_refs(&batch).unwrap().unwrap();
        assert_eq!(
            refs,
            vec![Some("frames/frame_000000.png".to_owned()), None, None]
        );
    }

    #[test]
    fn decode_frame_refs_falls_back_to_frame_url() {
        // With no `frame_path`, the `frame_url` column (browser) is used instead.
        let batch = frame_ref_batch("frame_url", &[Some("https://host/a.png"), None]);
        let refs = decode_frame_refs(&batch).unwrap().unwrap();
        assert_eq!(refs, vec![Some("https://host/a.png".to_owned()), None]);
    }

    #[test]
    fn decode_frame_refs_prefers_frame_path_over_url() {
        // Both columns present ⇒ native path wins.
        let schema = Arc::new(Schema::new(vec![
            Field::new("frame_path", DataType::Utf8, true),
            Field::new("frame_url", DataType::Utf8, true),
        ]));
        let path = StringArray::from(vec![Some("local/a.png")]);
        let url = StringArray::from(vec![Some("https://host/a.png")]);
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(path) as ArrayRef, Arc::new(url) as ArrayRef],
        )
        .unwrap();
        let refs = decode_frame_refs(&batch).unwrap().unwrap();
        assert_eq!(refs, vec![Some("local/a.png".to_owned())]);
    }

    #[test]
    fn decodes_variable_length_draw_lists() {
        let a = [
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let b = [
            2.0f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 5.0, 6.0, 7.0, 1.0,
        ];
        // Row 0 draws two instances (meshes 0 and 1); row 1 draws one (mesh 1).
        let batch = draw_batch(&[vec![0, 1], vec![1]], &[vec![a, b], vec![b]]);
        let rows = decode_draws(&batch).unwrap().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows[0],
            vec![
                Draw {
                    mesh_id: 0,
                    model: a,
                    mode: None
                },
                Draw {
                    mesh_id: 1,
                    model: b,
                    mode: None
                },
            ]
        );
        assert_eq!(
            rows[1],
            vec![Draw {
                mesh_id: 1,
                model: b,
                mode: None
            }]
        );
    }

    #[test]
    fn rejects_mismatched_draw_lists() {
        let m = [0.0f32; 16];
        // Row 0: two mesh ids but only one model.
        let batch = draw_batch(&[vec![0, 1]], &[vec![m]]);
        assert!(matches!(
            decode_draws(&batch),
            Err(StreamError::MismatchedDrawLists {
                row: 0,
                mesh_len: 2,
                model_len: 1,
            })
        ));
    }

    // Build a `[draw_mesh, draw_model, draw_mode]` batch (`draw_mode` optional).
    fn draw_batch_with_modes(
        mesh_rows: &[Vec<u32>],
        model_rows: &[Vec<[f32; 16]>],
        mode_rows: Option<&[Vec<u8>]>,
    ) -> RecordBatch {
        let n = mesh_rows.len();
        let mut fields = vec![
            model_field(),
            draw_field("draw_mesh", DataType::UInt32),
            draw_field(
                "draw_model",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 16),
            ),
        ];
        let mut cols: Vec<ArrayRef> = vec![
            model_col(n),
            draw_mesh_col(mesh_rows),
            draw_model_col(model_rows),
        ];
        if let Some(mode_rows) = mode_rows {
            fields.push(draw_field("draw_mode", DataType::UInt8));
            let flat: Vec<u8> = mode_rows.iter().flatten().copied().collect();
            let offsets = OffsetBuffer::from_lengths(mode_rows.iter().map(Vec::len));
            cols.push(Arc::new(ListArray::new(
                Arc::new(Field::new("item", DataType::UInt8, false)),
                offsets,
                Arc::new(UInt8Array::from(flat)),
                None,
            )) as ArrayRef);
        }
        RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
    }

    #[test]
    fn decodes_per_draw_render_modes() {
        let m = [0.0f32; 16];
        // Row 0 mixes a global-inheriting draw (255) with an explicit wireframe
        // (1); row 1's textured (2) and filled (0). Absent column ⇒ all None.
        let batch = draw_batch_with_modes(
            &[vec![0, 1], vec![2]],
            &[vec![m, m], vec![m]],
            Some(&[vec![255, 1], vec![2]]),
        );
        let rows = decode_draws(&batch).unwrap().unwrap();
        assert_eq!(rows[0][0].mode, None);
        assert_eq!(rows[0][1].mode, Some(RenderMode::Wireframe));
        assert_eq!(rows[1][0].mode, Some(RenderMode::Textured));

        // Absent `draw_mode` column ⇒ every draw inherits (None).
        let plain = draw_batch_with_modes(&[vec![0, 1]], &[vec![m, m]], None);
        let plain_rows = decode_draws(&plain).unwrap().unwrap();
        assert!(plain_rows[0].iter().all(|d| d.mode.is_none()));
    }

    #[test]
    fn rejects_invalid_and_mismatched_draw_modes() {
        let m = [0.0f32; 16];
        // A byte outside {0,1,2,255} is rejected.
        let bad = draw_batch_with_modes(&[vec![0]], &[vec![m]], Some(&[vec![7]]));
        assert!(matches!(
            decode_draws(&bad),
            Err(StreamError::InvalidDrawMode { value: 7 })
        ));
        // A `draw_mode` list shorter than the draw list is rejected.
        let short = draw_batch_with_modes(&[vec![0, 1]], &[vec![m, m]], Some(&[vec![0]]));
        assert!(matches!(
            decode_draws(&short),
            Err(StreamError::MismatchedDrawModes {
                row: 0,
                mode_len: 1,
                draw_len: 2,
            })
        ));
    }

    #[test]
    fn draw_columns_must_come_as_a_pair() {
        let batch = build_input_batch(&[FrameParams::IDENTITY]);
        let schema = Arc::new(Schema::new(vec![
            batch.schema().field(0).clone(),
            draw_field("draw_mesh", DataType::UInt32),
        ]));
        let with_mesh_only = RecordBatch::try_new(
            schema,
            vec![batch.column(0).clone(), draw_mesh_col(&[vec![0]])],
        )
        .unwrap();
        assert!(matches!(
            decode_draws(&with_mesh_only),
            Err(StreamError::MissingColumn("draw_model"))
        ));
    }

    #[test]
    fn build_scene_maps_draws_and_overlays_in_bucket_order() {
        // #41: the draw list + mode/overlay flags become an ordered `Scene` of
        // `DrawableObject`s — one Mesh per draw (in `mode`), then one AabbBox per
        // draw when enabled, then a single origin CoordinateAxes when enabled.
        let a = [1.0f32; 16];
        let b = [2.0f32; 16];
        let draws = [
            Draw {
                mesh_id: 0,
                model: a,
                mode: None,
            },
            Draw {
                mesh_id: 1,
                model: b,
                mode: None,
            },
        ];

        // Plain filled: exactly one Mesh drawable per draw, no gizmos.
        assert_eq!(
            build_scene(
                &draws,
                RenderMode::Filled,
                false,
                false,
                false,
                None,
                None,
                None
            ),
            vec![
                DrawableObject::Mesh {
                    mesh_id: 0,
                    model: a,
                    mode: RenderMode::Filled,
                },
                DrawableObject::Mesh {
                    mesh_id: 1,
                    model: b,
                    mode: RenderMode::Filled,
                },
            ]
        );

        // Wireframe propagates the mode to every mesh drawable.
        assert_eq!(
            build_scene(
                &draws,
                RenderMode::Wireframe,
                false,
                false,
                false,
                None,
                None,
                None
            ),
            vec![
                DrawableObject::Mesh {
                    mesh_id: 0,
                    model: a,
                    mode: RenderMode::Wireframe,
                },
                DrawableObject::Mesh {
                    mesh_id: 1,
                    model: b,
                    mode: RenderMode::Wireframe,
                },
            ]
        );

        // Both overlays: meshes, then a tracking box per draw, then one gizmo.
        assert_eq!(
            build_scene(
                &draws,
                RenderMode::Filled,
                true,
                true,
                false,
                None,
                None,
                None
            ),
            vec![
                DrawableObject::Mesh {
                    mesh_id: 0,
                    model: a,
                    mode: RenderMode::Filled,
                },
                DrawableObject::Mesh {
                    mesh_id: 1,
                    model: b,
                    mode: RenderMode::Filled,
                },
                DrawableObject::AabbBox {
                    mesh_id: 0,
                    model: a,
                },
                DrawableObject::AabbBox {
                    mesh_id: 1,
                    model: b,
                },
                DrawableObject::CoordinateAxes {
                    model: Matrix4::IDENTITY.to_cols_array(),
                },
            ]
        );

        // Local axes: one CoordinateAxes per draw at its own model (in the mesh
        // bucket order, before the world-origin gizmo), each tracking its draw.
        assert_eq!(
            build_scene(
                &draws,
                RenderMode::Filled,
                false,
                false,
                true,
                None,
                None,
                None
            ),
            vec![
                DrawableObject::Mesh {
                    mesh_id: 0,
                    model: a,
                    mode: RenderMode::Filled,
                },
                DrawableObject::Mesh {
                    mesh_id: 1,
                    model: b,
                    mode: RenderMode::Filled,
                },
                DrawableObject::CoordinateAxes { model: a },
                DrawableObject::CoordinateAxes { model: b },
            ]
        );

        // Per-draw mode override: a draw's own `mode` wins over the global one,
        // so one frame can mix (e.g.) a textured mesh with a wireframe overlay.
        let mixed = [
            Draw {
                mesh_id: 0,
                model: a,
                mode: None,
            },
            Draw {
                mesh_id: 1,
                model: b,
                mode: Some(RenderMode::Wireframe),
            },
        ];
        assert_eq!(
            build_scene(
                &mixed,
                RenderMode::Textured,
                false,
                false,
                false,
                None,
                None,
                None
            ),
            vec![
                DrawableObject::Mesh {
                    mesh_id: 0,
                    model: a,
                    mode: RenderMode::Textured,
                },
                DrawableObject::Mesh {
                    mesh_id: 1,
                    model: b,
                    mode: RenderMode::Wireframe,
                },
            ]
        );
    }

    #[test]
    fn child_null_in_camera_list_is_error() {
        // A non-null camera-list row whose child float is null must be rejected.
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let eye_values = Float32Array::from(vec![Some(0.0), Some(0.0), None]);
        let eye = FixedSizeListArray::new(item, 3, Arc::new(eye_values), None);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "eye",
            eye.data_type().clone(),
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(eye) as ArrayRef]).unwrap();
        assert!(matches!(
            decode_frames(&batch),
            Err(StreamError::NullValues("eye"))
        ));
    }

    #[test]
    fn wrong_type_is_error() {
        use arrow::array::Int32Array;
        // A camera scalar column of the wrong Arrow type must be rejected.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "fovy",
            DataType::Int32,
            false,
        )]));
        let fovy = Int32Array::from(vec![3]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(fovy) as ArrayRef]).unwrap();
        assert!(matches!(
            decode_frames(&batch),
            Err(StreamError::ColumnType { column: "fovy", .. })
        ));
    }

    #[test]
    fn version_check_rejects_mismatch() {
        let schema = Schema::empty().with_metadata(
            [(PROTOCOL_VERSION_KEY.to_string(), "9.9.9".to_string())]
                .into_iter()
                .collect(),
        );
        assert!(matches!(
            check_version(&schema),
            Err(StreamError::UnsupportedVersion(v)) if v == "9.9.9"
        ));
    }

    #[test]
    fn version_check_allows_absent_and_matching() {
        assert!(check_version(&Schema::empty()).is_ok());
        let versioned = Schema::empty().with_metadata(
            [(
                PROTOCOL_VERSION_KEY.to_string(),
                PROTOCOL_VERSION.to_string(),
            )]
            .into_iter()
            .collect(),
        );
        assert!(check_version(&versioned).is_ok());
    }

    #[test]
    fn check_dimensions_rejects_zero_and_overflow() {
        assert!(check_dimensions(4, 3).is_ok());
        assert!(matches!(
            check_dimensions(0, 3),
            Err(StreamError::InvalidDimensions { .. })
        ));
        // width*height overflows u32 / exceeds i32::MAX.
        assert!(matches!(
            check_dimensions(100_000, 100_000),
            Err(StreamError::InvalidDimensions { .. })
        ));
    }

    // ---- two-stream [mesh][params] framing ----

    /// Serializes a mesh as a one-row Arrow IPC **mesh stream** (nested list
    /// columns: `position`/`color` `List<FixedSizeList<Float32>[3]>`, `index`
    /// `List<UInt32>`), tagged with the 0.0.3 protocol version.
    fn write_mesh_stream(buf: &mut Vec<u8>, mesh: &Mesh) {
        use arrow::array::{ListArray, UInt32Array};
        use arrow::buffer::OffsetBuffer;

        let fsl_type =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 3);
        let geometry = |flat: Vec<f32>| -> ArrayRef {
            let fsl = FixedSizeListArray::new(
                Arc::new(Field::new("item", DataType::Float32, false)),
                3,
                Arc::new(Float32Array::from(flat)),
                None,
            );
            let field = Arc::new(Field::new("item", fsl_type.clone(), false));
            let offsets = OffsetBuffer::from_lengths([fsl.len()]);
            Arc::new(ListArray::new(field, offsets, Arc::new(fsl), None))
        };
        let positions: Vec<f32> = mesh.vertices.iter().flat_map(|v| v.position).collect();
        let colors: Vec<f32> = mesh.vertices.iter().flat_map(|v| v.color).collect();
        let idx_values = UInt32Array::from(mesh.indices.clone());
        let index: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new("item", DataType::UInt32, false)),
            OffsetBuffer::from_lengths([idx_values.len()]),
            Arc::new(idx_values),
            None,
        ));

        let list_of_fsl = DataType::List(Arc::new(Field::new("item", fsl_type.clone(), false)));
        let schema = Schema::new(vec![
            Field::new("position", list_of_fsl.clone(), false),
            Field::new("color", list_of_fsl, false),
            Field::new(
                "index",
                DataType::List(Arc::new(Field::new("item", DataType::UInt32, false))),
                false,
            ),
        ])
        .with_metadata(
            [(
                PROTOCOL_VERSION_KEY.to_string(),
                PROTOCOL_VERSION.to_string(),
            )]
            .into_iter()
            .collect(),
        );
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![geometry(positions), geometry(colors), index],
        )
        .unwrap();
        let mut wr = StreamWriter::try_new(buf, &schema).unwrap();
        wr.write(&batch).unwrap();
        wr.finish().unwrap();
    }

    /// Serializes frames as an Arrow IPC **params stream**.
    fn write_params_stream(buf: &mut Vec<u8>, frames: &[FrameParams]) {
        let batch = build_input_batch(frames);
        let mut wr = StreamWriter::try_new(buf, batch.schema().as_ref()).unwrap();
        wr.write(&batch).unwrap();
        wr.finish().unwrap();
    }

    #[test]
    fn two_stream_mesh_then_params_split_and_decode() {
        // Build a concatenated [mesh][params] byte stream in memory.
        let mesh = Mesh::hello_triangle();
        let frames = vec![
            FrameParams {
                model: Some(IDENTITY_MODEL),
                ..FrameParams::IDENTITY
            },
            FrameParams {
                model: Some([
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.3, 0.0, 0.0, 1.0,
                ]),
                ..FrameParams::IDENTITY
            },
        ];
        let mut bytes = Vec::new();
        write_mesh_stream(&mut bytes, &mesh);
        write_params_stream(&mut bytes, &frames);

        // The single shared `InputSession` framing driver must recover the mesh,
        // then the params that follow it in the same byte stream (the mesh
        // sub-stream boundary must not swallow the params).
        let mut session = crate::InputSession::new();
        let mut decoded = Vec::new();
        for batch in session.push(&bytes).unwrap() {
            for frame in batch {
                decoded.push(frame.params);
            }
        }
        session.finish().unwrap();
        assert_eq!(session.meshes(), &[mesh]);
        assert_eq!(decoded, frames);
    }

    #[test]
    fn params_stream_is_not_mesh_schema() {
        let mut bytes = Vec::new();
        write_params_stream(&mut bytes, &[FrameParams::IDENTITY]);
        let reader = StreamReader::try_new(bytes.as_slice(), None).unwrap();
        assert!(!crate::protocol::is_mesh_schema(reader.schema().as_ref()));
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn run_stream_renders_mesh_first_stream() {
        let (w, h) = (32u32, 32u32);
        // A full-screen quad as the leading mesh; two params frames follow.
        let mesh =
            Mesh::from_obj("v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nf 1 2 3\nf 1 3 4\n").unwrap();
        let frames = vec![
            FrameParams::IDENTITY,
            FrameParams {
                model: Some([
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.2, 0.0, 0.0, 1.0,
                ]),
                ..FrameParams::IDENTITY
            },
        ];
        let mut input_bytes = Vec::new();
        write_mesh_stream(&mut input_bytes, &mesh);
        write_params_stream(&mut input_bytes, &frames);

        let mut output_bytes = Vec::new();
        run_stream(
            &input_bytes[..],
            &mut output_bytes,
            w,
            h,
            RenderOptions::default(),
            None,
        )
        .unwrap();

        let reader = StreamReader::try_new(&output_bytes[..], None).unwrap();
        let batches = reader.collect::<Result<Vec<_>, _>>().unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, frames.len());

        // The white quad covers the frame, so the center pixel must be lit.
        let get = |batch: &RecordBatch, name: &str| -> U8List {
            batch
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<U8List>()
                .unwrap()
                .clone()
        };
        let r = get(&batches[0], "r");
        let center = (h as usize / 2) * w as usize + w as usize / 2;
        let value = r
            .value(0)
            .as_any()
            .downcast_ref::<UInt8Array>()
            .unwrap()
            .value(center);
        assert!(value > 0, "mesh quad should cover the center pixel");
    }
}
