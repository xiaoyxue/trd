//! trd-gui: the interactive egui front-end (issue #97).
//!
//! **Scaffold.** This crate currently opens an empty eframe/egui window as the
//! baseline for the interactive viewer. The real work lands as the vertical
//! slices tracked in issue #97:
//!
//! 1. **In-process backend** — load the bunny (`Mesh::from_obj`), render headless
//!    RGBA via [`trd_core::BatchRenderer`], show it as an egui texture.
//! 2. **External process via `trd`** — author a `[mesh][texture?][params]` Arrow
//!    scene, pipe it through the `trd` CLI, read the Arrow image stream back.
//! 3. **External process via wasm** — the browser target (trd-core wasm offscreen
//!    render → egui texture).
//! 4. **Interaction** — orbit (rotate about axes), zoom in/out, translation:
//!    each gesture computes an updated model/camera matrix and re-renders.
//!
//! Strategy A (decoupled CPU-RGBA handoff): eframe's own renderer draws the UI
//! while trd-core renders the scene to CPU RGBA, so the toolkits stay
//! independent of trd-core's `wgpu 30`. All rendering logic stays in trd-core;
//! this crate only owns UI, interaction, scene authoring, and the display
//! texture.

// The interactive stack is native-only for now; on wasm the crate is an empty
// `main` so workspace-wide wasm builds skip it (mirrors trd-app). Browser
// support (Slice 3 in #97) is added later.
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,trd_gui=info,trd_core=info"),
    )
    .init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 700.0]),
        ..Default::default()
    };
    eframe::run_native(
        "trd-gui",
        options,
        Box::new(|_cc| Ok(Box::<TrdGuiApp>::default())),
    )
}

/// The interactive viewer application. A scaffold today: it draws a placeholder
/// panel. The scene state, render backends, and interaction controller (issue
/// #97) attach here.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct TrdGuiApp {}

#[cfg(not(target_arch = "wasm32"))]
impl eframe::App for TrdGuiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Frame::central_panel(ui.style()).show(ui, |ui| {
            ui.heading("trd-gui — interactive viewer (scaffold)");
            ui.label(format!(
                "Rendering core: trd-core (protocol {}). \
                 Implementation is tracked in issue #97.",
                trd_core::PROTOCOL_VERSION
            ));
            ui.separator();
            ui.label(
                "Next: in-process backend renders the bunny to an egui texture, \
                 then orbit / zoom / translate interaction.",
            );
        });
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
