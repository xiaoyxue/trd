#![cfg(target_arch = "wasm32")]

use std::fmt::Display;

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
    renderer: trd_core::TriangleRenderer,
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
            renderer: trd_core::TriangleRenderer::new(&device, config.format),
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
                        self.renderer.encode(
                            &self.queue,
                            &mut encoder,
                            &view,
                            frame,
                            self.config.width,
                            self.config.height,
                        );
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
}

impl CanvasRenderer {
    fn require_open(&self) -> Result<(), JsValue> {
        match self.state {
            CanvasState::Open => Ok(()),
            CanvasState::Finished => Err(js_error("CanvasRenderer is finished")),
            CanvasState::Failed => Err(js_error("CanvasRenderer is failed")),
        }
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
