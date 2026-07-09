//! trd-wasm: the browser (wasm) entry point for the trd rendering core.
//!
//! This is a thin bootstrap. It obtains a wgpu surface from the given canvas and
//! calls the shared [`trd_core::render_triangle`]. The WebGPU API is never used
//! from JavaScript; all rendering logic lives in the Rust core.
//!
//! The crate is empty on non-wasm targets so that native workspace builds skip
//! the web-only wgpu surface APIs.
#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;

fn js_err(context: &str, error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&format!("{context}: {error}"))
}

/// Renders the hello-triangle into `canvas` using the shared render core.
#[wasm_bindgen]
pub async fn start(canvas: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let width = canvas.width();
    let height = canvas.height();
    if width == 0 || height == 0 {
        return Err(JsValue::from_str("canvas width/height must be non-zero"));
    }

    let instance = wgpu::Instance::default();
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
        .map_err(|e| js_err("create_surface failed", e))?;

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .map_err(|e| js_err("request_adapter failed", e))?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("trd wasm device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| js_err("request_device failed", e))?;

    let config = surface
        .get_default_config(&adapter, width, height)
        .ok_or_else(|| JsValue::from_str("surface is not supported by the adapter"))?;
    surface.configure(&device, &config);

    let frame = match surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(frame)
        | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
        _ => return Err(JsValue::from_str("failed to acquire surface texture")),
    };
    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    trd_core::render_triangle(&device, &queue, &view, config.format);

    queue.present(frame);
    Ok(())
}
