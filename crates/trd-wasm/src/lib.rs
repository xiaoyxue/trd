#![cfg(target_arch = "wasm32")]

use std::fmt::Display;

use trd_core::{
    build_scene, Draw, Matrix4, Mesh, MeshRenderer, RenderMode, Viewport, DEFAULT_PREVIEW_TARGET,
};
use wasm_bindgen::prelude::*;

mod arrow_renderer;

pub use arrow_renderer::ArrowRenderer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CanvasState {
    Open,
    Finished,
    Failed,
}

struct AcquiredFrame {
    texture: wgpu::SurfaceTexture,
    reconfigure_after_present: bool,
}

#[wasm_bindgen]
pub struct CanvasRenderer {
    instance: wgpu::Instance,
    canvas: web_sys::HtmlCanvasElement,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    /// Built lazily on the first rendered frame: a multi-mesh renderer over the
    /// stream's leading mesh table, or the built-in hello-triangle for a legacy
    /// params-only stream. `None` until the first frame arrives (the mesh table,
    /// if any, has been decoded by then).
    renderer: Option<MeshRenderer>,
    mode: RenderMode,
    show_aabb: bool,
    show_axes: bool,
    input: trd_core::InputSession,
    state: CanvasState,
}

pub(crate) fn js_error(message: impl Display) -> JsValue {
    js_sys::Error::new(&message.to_string()).into()
}

#[wasm_bindgen]
impl CanvasRenderer {
    #[wasm_bindgen(js_name = create)]
    pub async fn create(canvas: web_sys::HtmlCanvasElement) -> Result<Self, JsValue> {
        console_error_panic_hook::set_once();

        let width = canvas.width();
        let height = canvas.height();
        if width == 0 || height == 0 {
            return Err(js_error("canvas width and height must be non-zero"));
        }

        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|error| js_error(format!("create_surface failed: {error}")))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(|error| js_error(format!("request_adapter failed: {error}")))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("trd canvas device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|error| js_error(format!("request_device failed: {error}")))?;
        let config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| js_error("surface is unsupported by the selected adapter"))?;
        surface.configure(&device, &config);

        Ok(Self {
            renderer: None,
            mode: RenderMode::Filled,
            show_aabb: false,
            show_axes: false,
            instance,
            canvas,
            surface,
            device,
            queue,
            config,
            input: trd_core::InputSession::new(),
            state: CanvasState::Open,
        })
    }

    #[wasm_bindgen(js_name = pushIpc)]
    pub fn push_ipc(&mut self, chunk: &[u8]) -> Result<u32, JsValue> {
        self.require_open()?;

        let result = (|| {
            let batches = measure("trd.ipc.decode", || {
                self.input
                    .push(chunk)
                    .map_err(|error| js_error(format!("Arrow IPC input failed: {error}")))
            })?;
            let rendered = batches.iter().try_fold(0_u32, |total, batch| {
                let rows = u32::try_from(batch.len())
                    .map_err(|_| js_error("decoded batch row count does not fit u32"))?;
                total
                    .checked_add(rows)
                    .ok_or_else(|| js_error("rendered row count would overflow u32"))
            })?;

            for batch in batches {
                for frame in batch {
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
                            return Err(js_error(format!(
                                "draw references mesh {} but only {mesh_count} \
                                 mesh(es) are loaded",
                                draw.mesh_id
                            )));
                        }
                    }
                    let scene = build_scene(&draws, self.mode, self.show_aabb, self.show_axes);

                    measure("trd.canvas.render-submit", || {
                        let acquired = self.acquire_frame()?;
                        let view = acquired
                            .texture
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());
                        let mut encoder =
                            self.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("trd canvas frame"),
                                });
                        let viewport = Viewport {
                            width: self.config.width,
                            height: self.config.height,
                        };
                        self.renderer
                            .as_mut()
                            .expect("renderer built above")
                            .encode(&self.queue, &mut encoder, &view, params, &scene, viewport);
                        self.queue.submit(Some(encoder.finish()));
                        self.queue.present(acquired.texture);
                        if acquired.reconfigure_after_present {
                            self.surface.configure(&self.device, &self.config);
                        }
                        Ok(())
                    })?;
                }
            }

            Ok(rendered)
        })();

        if result.is_err() {
            self.state = CanvasState::Failed;
        }
        result
    }

    pub fn finish(&mut self) -> Result<(), JsValue> {
        self.require_open()?;
        match self
            .input
            .finish()
            .map_err(|error| js_error(format!("Arrow IPC finish failed: {error}")))
        {
            Ok(()) => {
                self.state = CanvasState::Finished;
                Ok(())
            }
            Err(error) => {
                self.state = CanvasState::Failed;
                Err(error)
            }
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

    /// Selects textured (`true`) rendering — sampling the stream's bound texture
    /// table at each vertex UV — or per-vertex color (`false`) for later frames.
    /// Textured meshes without a stream texture sample the default 1×1 white.
    #[wasm_bindgen(js_name = setTextured)]
    pub fn set_textured(&mut self, enabled: bool) {
        self.mode = if enabled {
            RenderMode::Textured
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

impl CanvasRenderer {
    fn require_open(&self) -> Result<(), JsValue> {
        match self.state {
            CanvasState::Open => Ok(()),
            CanvasState::Finished => Err(js_error("CanvasRenderer is finished")),
            CanvasState::Failed => Err(js_error("CanvasRenderer is failed")),
        }
    }

    /// Lazily builds the mesh renderer on first use. If the stream carried a
    /// leading mesh table (`input.has_meshes()`), builds a multi-mesh renderer
    /// with each mesh's [`preview_transform`](trd_core::Mesh::preview_transform)
    /// base model; otherwise falls back to the built-in hello-triangle so legacy
    /// params-only streams keep rendering.
    fn ensure_renderer(&mut self) -> &mut MeshRenderer {
        if self.renderer.is_none() {
            let renderer = if self.input.has_meshes() {
                let meshes = self.input.meshes();
                let base_models: Vec<Matrix4> = meshes
                    .iter()
                    .map(|mesh| mesh.preview_transform(DEFAULT_PREVIEW_TARGET).matrix())
                    .collect();
                MeshRenderer::with_meshes(&self.device, self.config.format, meshes, &base_models)
            } else {
                MeshRenderer::with_base_model(
                    &self.device,
                    self.config.format,
                    &Mesh::hello_triangle(),
                    Matrix4::IDENTITY,
                )
            };
            self.renderer = Some(renderer);

            // Bind the stream's texture (0.0.4) as the sampled albedo so
            // RenderMode::Textured meshes show it; absent ⇒ the default 1×1 white.
            if let Some(texture) = self.input.texture() {
                self.renderer
                    .as_mut()
                    .expect("renderer just built")
                    .set_texture(texture);
            }
        }
        self.renderer.as_mut().expect("renderer just built")
    }

    fn acquire_frame(&mut self) -> Result<AcquiredFrame, JsValue> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => Ok(AcquiredFrame {
                texture,
                reconfigure_after_present: false,
            }),
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => Ok(AcquiredFrame {
                texture,
                reconfigure_after_present: true,
            }),
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                self.acquire_after_recovery("reconfiguration")
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface = self
                    .instance
                    .create_surface(wgpu::SurfaceTarget::Canvas(self.canvas.clone()))
                    .map_err(|error| js_error(format!("surface recreation failed: {error}")))?;
                self.surface.configure(&self.device, &self.config);
                self.acquire_after_recovery("recreation")
            }
            wgpu::CurrentSurfaceTexture::Timeout => Err(js_error("surface acquisition timed out")),
            wgpu::CurrentSurfaceTexture::Occluded => Err(js_error("surface is occluded")),
            wgpu::CurrentSurfaceTexture::Validation => Err(js_error("surface validation failed")),
        }
    }

    fn acquire_after_recovery(&self, recovery: &str) -> Result<AcquiredFrame, JsValue> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => Ok(AcquiredFrame {
                texture,
                reconfigure_after_present: false,
            }),
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => Ok(AcquiredFrame {
                texture,
                reconfigure_after_present: true,
            }),
            wgpu::CurrentSurfaceTexture::Timeout => Err(js_error(format!(
                "surface acquisition timed out after {recovery}"
            ))),
            wgpu::CurrentSurfaceTexture::Occluded => {
                Err(js_error(format!("surface is occluded after {recovery}")))
            }
            wgpu::CurrentSurfaceTexture::Outdated => Err(js_error(format!(
                "surface remains outdated after {recovery}"
            ))),
            wgpu::CurrentSurfaceTexture::Lost => {
                Err(js_error(format!("surface remains lost after {recovery}")))
            }
            wgpu::CurrentSurfaceTexture::Validation => Err(js_error(format!(
                "surface validation failed after {recovery}"
            ))),
        }
    }
}

fn measure<T>(name: &str, work: impl FnOnce() -> Result<T, JsValue>) -> Result<T, JsValue> {
    let performance = web_sys::window()
        .and_then(|window| window.performance())
        .ok_or_else(|| js_error("Performance API is unavailable"))?;
    let start = format!("{name}:start");
    let end = format!("{name}:end");

    performance.mark(&start)?;
    // Record the measure regardless of whether `work` succeeded, so the `:start`
    // mark is never leaked on an error path. The measure itself is only emitted
    // on success; on error we still clear both scratch marks before returning.
    let outcome = work().and_then(|value| {
        performance.mark(&end)?;
        performance.measure_with_start_mark_and_end_mark(name, &start, &end)?;
        Ok(value)
    });
    performance.clear_marks_with_mark_name(&start);
    performance.clear_marks_with_mark_name(&end);
    outcome
}
