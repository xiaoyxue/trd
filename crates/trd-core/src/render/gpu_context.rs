//! Shared wgpu device/adapter acquisition (#103, Part B / #128).
//!
//! The one place trd turns a [`wgpu::Instance`] into an adapter + `(device,
//! queue)` pair, so all shells pick the GPU, limits, memory hints, and adapter
//! logging identically. The only genuinely shell-specific bits stay outside:
//! how the [`wgpu::Instance`] is built (native honours `WGPU_BACKEND`, wasm is
//! `default`) and whether an adapter must be compatible with a surface.
//!
//! This is the substrate under both [`TextureTarget`](super::TextureTarget)
//! and the on-screen front-ends: device/adapter acquisition is orthogonal to
//! *what* you render into, and two of the five shells must create a surface
//! *before* requesting the adapter, so it lives as its own primitive rather than
//! inside a target harness.

use std::sync::Arc;

use thiserror::Error;

/// Builds the platform-appropriate [`wgpu::Instance`].
///
/// Re-exported from [`platform`](super::platform), where the native/wasm split
/// lives; see there for why the two differ.
pub use super::platform::create_instance;

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
/// The adapter, device and queue, created together and shared as one
/// refcounted [`Arc<GpuContext>`] rather than cloned apart into separate fields
/// — which is what let six renderer wrappers each keep their own `device` +
/// `queue` pair while the type owning all three was destructured and thrown away
/// at every construction site (#180).
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
    ) -> Result<Arc<Self>, GpuInitError> {
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

        // On `wasm32-unknown-unknown` wgpu's handles are neither `Send` nor `Sync`
        // (no atomics), which trips `arc_with_non_send_sync`. `Arc` is still the
        // right vehicle: this is shared *ownership* of one context, never shared
        // across threads, and using it unconditionally keeps one type for both
        // platforms instead of an `Rc`/`Arc` cfg split through every renderer.
        #[allow(clippy::arc_with_non_send_sync)]
        Ok(Arc::new(Self {
            adapter,
            device,
            queue,
        }))
    }

    /// Adopts an **already-created** adapter/device/queue instead of requesting
    /// its own.
    ///
    /// This is what lets a front-end share one GPU device with its UI toolkit
    /// (`eframe`'s `wgpu_render_state` exposes exactly this trio). Two devices on
    /// the same adapter cannot share textures, so a shell that renders trd
    /// content *into* its UI must build the context from the UI's device or pay
    /// a GPU→CPU→GPU round trip per frame.
    ///
    /// The `adopting …` line mirrors [`request`](Self::request)'s `using …`, so
    /// the log says which path a run took — and a shell that accidentally opens
    /// a second device is visible as two lines instead of one.
    ///
    /// The caller is responsible for the device having the limits trd needs;
    /// [`request`](Self::request) is still the path for a standalone context.
    pub fn adopt(adapter: wgpu::Adapter, device: wgpu::Device, queue: wgpu::Queue) -> Arc<Self> {
        let info = adapter.get_info();
        log::info!(
            "adopting {:?} adapter \"{}\" ({:?}), driver: {}",
            info.backend,
            info.name,
            info.device_type,
            info.driver_info
        );
        #[allow(clippy::arc_with_non_send_sync)]
        Arc::new(Self {
            adapter,
            device,
            queue,
        })
    }

    /// wgpu-free adapter facts for diagnostics panels.
    ///
    /// Returns a plain trd-core value rather than a front-end type, so the GUI's
    /// own display struct can be built from it without trd-core depending on the
    /// GUI. Previously every shell re-derived this from `adapter.get_info()`
    /// after discarding the adapter.
    pub fn adapter_facts(&self) -> AdapterFacts {
        let info = self.adapter.get_info();
        AdapterFacts {
            name: info.name,
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
        }
    }
}

/// Adapter identity, as a plain value (no `wgpu` types) so front-ends can show
/// it without reaching for the adapter themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterFacts {
    pub name: String,
    pub backend: String,
    pub device_type: String,
}
