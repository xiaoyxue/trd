//! [`WasmRenderer`] — the browser render backend (#97, Slice 4).
//!
//! The wasm twin of the native [`InProcRenderer`](crate::render_backend::InProcRenderer):
//! it builds a `trd-core` [`MeshRenderer`] once (its own wgpu 30 device, no
//! surface), then renders the interactive [`SceneState`] to an **offscreen**
//! texture and reads it back to RGBA — the pixels the eframe app uploads as an
//! egui texture (Strategy A: only CPU RGBA crosses, so egui's WebGL backend stays
//! independent of `trd-core`'s wgpu). Mirrors the async readback path of
//! `trd-wasm`'s offscreen `ArrowRenderer` (`map_async` + `device.poll` + await).
//!
//! Rendering is **async** on wasm (GPU readback can't block the browser event
//! loop), so — unlike the native `SceneRenderer` trait — `render` is an
//! `async fn`; the app schedules it with `wasm_bindgen_futures::spawn_local`.

use futures_channel::oneshot;

use trd_core::{
    build_scene, tightly_pack_rgba, MeshRenderer, Texture, Viewport, DEFAULT_PREVIEW_TARGET,
};

use crate::error::GuiError;
use crate::render_backend::ImageRgba;
use crate::scene::SceneState;

/// A browser offscreen renderer over a `trd-core` [`MeshRenderer`].
pub struct WasmRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: MeshRenderer,
    target: wgpu::Texture,
    staging: wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
}

impl WasmRenderer {
    /// Builds the offscreen renderer (async: wgpu device creation) from the
    /// static meshes and an optional bound `texture` (sampled by
    /// [`RenderMode::Textured`](trd_core::RenderMode::Textured)).
    pub async fn new(
        meshes: &[trd_core::Mesh],
        texture: Option<&dyn Texture>,
        width: u32,
        height: u32,
    ) -> Result<Self, GuiError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .map_err(|e| GuiError::WasmRender(format!("request_adapter failed: {e}")))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("trd-gui wasm device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GuiError::WasmRender(format!("request_device failed: {e}")))?;

        let max_dim = device.limits().max_texture_dimension_2d;
        if width == 0 || height == 0 || width > max_dim || height > max_dim {
            return Err(GuiError::WasmRender(format!(
                "invalid render size {width}x{height} (max {max_dim})"
            )));
        }

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let base_models: Vec<trd_core::Matrix4> = meshes
            .iter()
            .map(|m| m.preview_transform(DEFAULT_PREVIEW_TARGET).matrix())
            .collect();
        let mut renderer = MeshRenderer::new(&device, format, meshes, &base_models);
        if let Some(texture) = texture {
            renderer.set_texture(texture);
        }

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("trd-gui wasm target"),
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

        let unpadded = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded.div_ceil(align) * align;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trd-gui wasm staging"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            renderer,
            target,
            staging,
            width,
            height,
            padded_bytes_per_row,
        })
    }

    /// The fixed render dimensions.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Renders the current scene state to an RGBA image (async GPU readback).
    pub async fn render(&mut self, state: &SceneState) -> Result<ImageRgba, GuiError> {
        let aspect = self.width as f32 / self.height.max(1) as f32;
        let params = state.frame_params(aspect);
        let scene = build_scene(
            &state.draws(),
            state.mode,
            state.show_aabb,
            state.show_axes,
            state.show_local_axes,
            None,
        );

        let view = self
            .target
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trd-gui wasm frame"),
            });
        let viewport = Viewport {
            width: self.width,
            height: self.height,
        };
        self.renderer
            .encode(&self.queue, &mut encoder, &view, params, &scene, viewport);

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
            .map_err(|e| GuiError::WasmRender(format!("GPU poll failed: {e}")))?;
        receiver
            .await
            .map_err(|_| GuiError::WasmRender("GPU readback cancelled".to_owned()))?
            .map_err(|e| GuiError::WasmRender(format!("GPU readback failed: {e}")))?;

        let rgba = match slice.get_mapped_range() {
            Ok(mapped) => {
                tightly_pack_rgba(&mapped, self.width, self.height, self.padded_bytes_per_row)
                    .map_err(|e| GuiError::WasmRender(format!("row unpack failed: {e}")))
            }
            Err(e) => Err(GuiError::WasmRender(format!("mapped range failed: {e}"))),
        };
        self.staging.unmap();

        Ok(ImageRgba {
            width: self.width,
            height: self.height,
            rgba: rgba?,
        })
    }
}
