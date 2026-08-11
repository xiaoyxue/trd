//! Shared wgpu device/adapter acquisition (#103, Part B / #128).
//!
//! The one place trd turns a [`wgpu::Instance`] into an adapter + `(device,
//! queue)` pair, so all shells pick the GPU, limits, memory hints, and adapter
//! logging identically. The only genuinely shell-specific bits stay outside:
//! how the [`wgpu::Instance`] is built (native honours `WGPU_BACKEND`, wasm is
//! `default`) and whether an adapter must be compatible with a surface.
//!
//! This is the substrate under both [`OffscreenTarget`](super::OffscreenTarget)
//! and the on-screen front-ends: device/adapter acquisition is orthogonal to
//! *what* you render into, and two of the five shells must create a surface
//! *before* requesting the adapter, so it lives as its own primitive rather than
//! inside a target harness.

use thiserror::Error;

/// Builds the platform-appropriate [`wgpu::Instance`].
///
/// Native uses `new_without_display_handle_from_env()` so `WGPU_BACKEND`
/// (e.g. `gl` on WSL2, per `AGENTS.md`) is honoured; wasm uses
/// [`wgpu::Instance::default`].
pub fn create_instance() -> wgpu::Instance {
    #[cfg(not(target_arch = "wasm32"))]
    {
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env())
    }
    #[cfg(target_arch = "wasm32")]
    {
        wgpu::Instance::default()
    }
}

/// Which device limits to request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitsPreset {
    /// The adapter's real limits (`default().using_resolution(adapter.limits())`)
    /// so large / high-DPI targets fit. Used by the window, canvas, and both
    /// browser-offscreen shells.
    Adapter,
    /// Conservative `downlevel_defaults()` (2048 texture cap). The headless CLI's
    /// historical choice — kept as an explicit opt-in so the golden render stays
    /// byte-identical.
    Downlevel,
}

impl LimitsPreset {
    /// Resolves this preset against `adapter` into concrete [`wgpu::Limits`].
    fn resolve(self, adapter: &wgpu::Adapter) -> wgpu::Limits {
        match self {
            LimitsPreset::Adapter => wgpu::Limits::default().using_resolution(adapter.limits()),
            LimitsPreset::Downlevel => wgpu::Limits::downlevel_defaults(),
        }
    }
}

/// The knobs for a device/adapter request. [`Default`] is the trd house style:
/// `HighPerformance` power preference (never a weak iGPU/display GPU, per
/// `AGENTS.md`), the adapter's real limits, no compatible surface, and default
/// memory hints.
pub struct GpuRequest<'a, 'b> {
    /// The `wgpu::DeviceDescriptor` label.
    pub label: &'a str,
    /// Adapter power preference. Defaults to
    /// [`wgpu::PowerPreference::HighPerformance`].
    pub power_preference: wgpu::PowerPreference,
    /// A surface the adapter must be able to present to (on-screen shells only).
    pub compatible_surface: Option<&'a wgpu::Surface<'b>>,
    /// Which [`LimitsPreset`] to request.
    pub limits: LimitsPreset,
    /// The device's memory-usage hint.
    pub memory_hints: wgpu::MemoryHints,
}

impl Default for GpuRequest<'_, '_> {
    fn default() -> Self {
        Self {
            label: "trd device",
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            limits: LimitsPreset::Adapter,
            memory_hints: wgpu::MemoryHints::default(),
        }
    }
}

/// A failure acquiring the GPU adapter or device.
#[derive(Debug, Error)]
pub enum GpuInitError {
    /// No adapter satisfied the [`GpuRequest`] (wrong power preference, no
    /// surface-compatible adapter, or no GPU at all).
    #[error("no suitable graphics adapter found: {0}")]
    NoAdapter(String),
    /// The adapter could not create the requested device.
    #[error("failed to create GPU device: {0}")]
    NoDevice(String),
}

/// An acquired adapter + device + queue.
///
/// The [`wgpu::Instance`] and any [`wgpu::Surface`] stay owned by the shell,
/// which may need them for surface (re)configuration; this owns only the trio
/// every shell needs.
pub struct GpuContext {
    /// The selected adapter (on-screen shells reuse it for surface config /
    /// capabilities).
    pub adapter: wgpu::Adapter,
    /// The logical device.
    pub device: wgpu::Device,
    /// The device's command queue.
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Requests an adapter and device from `instance` per `req`, emitting the
    /// `AGENTS.md`-mandated `using … adapter "…"` log line so which GPU trd
    /// actually chose is always recorded (native *and* the three wasm shells).
    pub async fn request(
        instance: &wgpu::Instance,
        req: &GpuRequest<'_, '_>,
    ) -> Result<Self, GpuInitError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: req.power_preference,
                compatible_surface: req.compatible_surface,
                ..Default::default()
            })
            .await
            .map_err(|e| GpuInitError::NoAdapter(e.to_string()))?;

        // AGENTS.md-mandated: always record which GPU trd actually chose.
        let info = adapter.get_info();
        log::info!(
            "using {:?} adapter \"{}\" ({:?}), driver: {}",
            info.backend,
            info.name,
            info.device_type,
            info.driver_info
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some(req.label),
                required_features: wgpu::Features::empty(),
                required_limits: req.limits.resolve(&adapter),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: req.memory_hints.clone(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GpuInitError::NoDevice(e.to_string()))?;

        Ok(Self {
            adapter,
            device,
            queue,
        })
    }
}
