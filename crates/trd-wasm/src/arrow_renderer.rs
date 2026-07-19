use futures_channel::oneshot;
use wasm_bindgen::prelude::*;

use trd_core::{
    build_scene, tightly_pack_rgba, Draw, DrawableObject, FrameBatch, FrameParams, InputSession,
    Matrix4, Mesh, MeshRenderer, OutputSession, RenderMode, Viewport, DEFAULT_PREVIEW_TARGET,
};

fn error_message(context: &str, error: impl std::fmt::Display) -> String {
    format!("{context}: {error}")
}

#[derive(Debug, Clone)]
enum RendererState {
    Open,
    Finished,
    Failed(String),
}

impl RendererState {
    fn ensure_open(&self) -> Result<(), String> {
        match self {
            Self::Open => Ok(()),
            Self::Finished => Err("ArrowRenderer is already finished".to_string()),
            Self::Failed(message) => Err(format!("ArrowRenderer is failed: {message}")),
        }
    }
}

#[wasm_bindgen]
pub struct ArrowRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    format: wgpu::TextureFormat,
    /// Built lazily on the first rendered frame from the stream's leading mesh
    /// table, or the built-in hello-triangle for a legacy params-only stream.
    renderer: Option<MeshRenderer>,
    mode: RenderMode,
    show_aabb: bool,
    show_axes: bool,
    target: wgpu::Texture,
    staging: wgpu::Buffer,
    input: InputSession,
    output: OutputSession,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    state: RendererState,
}

#[wasm_bindgen]
impl ArrowRenderer {
    #[wasm_bindgen(js_name = create)]
    pub async fn create(width: u32, height: u32) -> Result<Self, JsValue> {
        console_error_panic_hook::set_once();

        let output = OutputSession::new(width, height).map_err(|error| {
            crate::js_error(error_message("invalid ArrowRenderer dimensions", error))
        })?;

        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .map_err(|error| crate::js_error(error_message("request_adapter failed", error)))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("trd ArrowRenderer device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|error| crate::js_error(error_message("request_device failed", error)))?;

        let max_dimension = device.limits().max_texture_dimension_2d;
        if width > max_dimension || height > max_dimension {
            return Err(crate::js_error(format!(
                "ArrowRenderer dimensions {width}x{height} exceed max_texture_dimension_2d {max_dimension}"
            )));
        }

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("trd ArrowRenderer target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let unpadded = width.checked_mul(4).ok_or_else(|| {
            crate::js_error(format!(
                "ArrowRenderer row byte count overflows for width {width}"
            ))
        })?;
        let padded_bytes_per_row = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trd ArrowRenderer staging"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            format,
            renderer: None,
            mode: RenderMode::Filled,
            show_aabb: false,
            show_axes: false,
            target,
            staging,
            input: InputSession::new(),
            output,
            width,
            height,
            padded_bytes_per_row,
            state: RendererState::Open,
        })
    }

    #[wasm_bindgen(js_name = pushIpc)]
    pub async fn push_ipc(&mut self, chunk: Vec<u8>) -> Result<Vec<u8>, JsValue> {
        if let Err(message) = self.state.ensure_open() {
            return Err(crate::js_error(message));
        }

        match self.push_open(chunk).await {
            Ok(bytes) => Ok(bytes),
            Err(message) => self.fail(message),
        }
    }

    pub fn finish(&mut self) -> Result<Vec<u8>, JsValue> {
        if let Err(message) = self.state.ensure_open() {
            return Err(crate::js_error(message));
        }

        let result = (|| {
            self.input
                .finish()
                .map_err(|error| error_message("input IPC finish failed", error))?;
            self.output
                .finish()
                .map_err(|error| error_message("output IPC finish failed", error))?;
            self.state = RendererState::Finished;
            self.output
                .drain_new()
                .map_err(|error| error_message("output IPC drain failed", error))
        })();

        match result {
            Ok(bytes) => Ok(bytes),
            Err(message) => self.fail(message),
        }
    }

    /// Selects filled (`false`) or wireframe (`true`) rendering for later frames.
    #[wasm_bindgen(js_name = setWireframe)]
    pub fn set_wireframe(&mut self, enabled: bool) {
        self.mode = if enabled {
            RenderMode::Wireframe
        } else {
            RenderMode::Filled
        };
    }

    /// Toggles the per-instance AABB overlay box for later frames.
    #[wasm_bindgen(js_name = setShowAabb)]
    pub fn set_show_aabb(&mut self, enabled: bool) {
        self.show_aabb = enabled;
    }

    /// Toggles the origin coordinate-axes overlay gizmo for later frames.
    #[wasm_bindgen(js_name = setShowAxes)]
    pub fn set_show_axes(&mut self, enabled: bool) {
        self.show_axes = enabled;
    }
}

impl ArrowRenderer {
    fn fail<T>(&mut self, message: String) -> Result<T, JsValue> {
        self.state = RendererState::Failed(message.clone());
        Err(crate::js_error(message))
    }

    /// Lazily builds the mesh renderer on first use: a multi-mesh renderer over
    /// the stream's leading mesh table (each mesh under its `preview_transform`
    /// base model), or the built-in hello-triangle for a legacy params-only
    /// stream.
    fn ensure_renderer(&mut self) -> &mut MeshRenderer {
        if self.renderer.is_none() {
            let renderer = if self.input.has_meshes() {
                let meshes = self.input.meshes();
                let base_models: Vec<Matrix4> = meshes
                    .iter()
                    .map(|mesh| mesh.preview_transform(DEFAULT_PREVIEW_TARGET).matrix())
                    .collect();
                MeshRenderer::with_meshes(&self.device, self.format, meshes, &base_models)
            } else {
                MeshRenderer::with_base_model(
                    &self.device,
                    self.format,
                    &Mesh::hello_triangle(),
                    Matrix4::IDENTITY,
                )
            };
            self.renderer = Some(renderer);
        }
        self.renderer.as_mut().expect("renderer just built")
    }

    async fn push_open(&mut self, chunk: Vec<u8>) -> Result<Vec<u8>, String> {
        let frame_batches: Vec<FrameBatch> = self
            .input
            .push(&chunk)
            .map_err(|error| error_message("input IPC decode failed", error))?;

        for frame_batch in frame_batches {
            let mut images = Vec::with_capacity(frame_batch.len());

            for frame in frame_batch {
                let params = frame.params;
                // Absent per-frame draw list ⇒ one instance of mesh 0 placed by
                // the frame's own model (legacy single-object behavior).
                let draws: Vec<Draw> = if frame.draws.is_empty() {
                    vec![Draw {
                        mesh_id: 0,
                        model: params.model_matrix().to_cols_array(),
                    }]
                } else {
                    frame.draws
                };

                let mesh_count = self.ensure_renderer().mesh_count();
                for draw in &draws {
                    if draw.mesh_id as usize >= mesh_count {
                        return Err(format!(
                            "draw references mesh {} but only {mesh_count} mesh(es) are loaded",
                            draw.mesh_id
                        ));
                    }
                }
                let scene = build_scene(&draws, self.mode, self.show_aabb, self.show_axes);
                images.push(self.render_frame(params, &scene).await?);
            }

            self.output
                .write_rgba_batch(&images)
                .map_err(|error| error_message("output IPC write failed", error))?;
        }

        self.output
            .drain_new()
            .map_err(|error| error_message("output IPC drain failed", error))
    }

    async fn render_frame(
        &mut self,
        params: FrameParams,
        scene: &[DrawableObject],
    ) -> Result<Vec<u8>, String> {
        let view = self
            .target
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trd ArrowRenderer frame"),
            });

        let viewport = Viewport {
            width: self.width,
            height: self.height,
        };
        self.renderer
            .as_mut()
            .expect("renderer built before render_frame")
            .encode(&self.queue, &mut encoder, &view, params, scene, viewport);

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging.slice(..);
        let (sender, receiver) = oneshot::channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| error_message("GPU poll failed", error))?;

        let map_result = receiver
            .await
            .map_err(|_| "GPU readback callback was cancelled".to_string())?;

        map_result.map_err(|error| error_message("GPU readback failed", error))?;

        let packed = match slice.get_mapped_range() {
            Ok(mapped) => {
                tightly_pack_rgba(&mapped, self.width, self.height, self.padded_bytes_per_row)
                    .map_err(|error| error_message("GPU row unpack failed", error))
            }
            Err(error) => Err(error_message("GPU mapped range failed", error)),
        };

        self.staging.unmap();
        packed
    }
}
