//! trd-gui native binary: parse args, load the mesh, build the in-process
//! render backend, and run the eframe/egui window (issue #97).
//!
//! The interactive stack (eframe/egui + `trd-core`'s native `Renderer`) is
//! native-only, and nothing builds this binary for wasm. The browser delivery
//! shell lives in `web/gui-viewer` and calls the wasm entry exported by
//! `crates/trd-gui`.

mod app;
mod cli;

fn main() -> eframe::Result<()> {
    use crate::app::TrdGuiApp;
    use crate::cli::Cli;
    use clap::Parser;
    use trd_gui::interaction::InteractionController;
    use trd_gui::renderer::{mesh_has_uvs, GuiRenderer};

    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,trd_gui=info,trd_core=info"),
    )
    .init();

    let cli = Cli::parse();

    // Load the mesh and build the backend up front so a bad mesh / adapter fails
    // fast on the console rather than inside the window.
    let loaded = match cli.load_mesh() {
        Ok(loaded) => loaded,
        Err(err) => {
            log::error!("{err}");
            std::process::exit(1);
        }
    };
    let texture = match cli.load_texture() {
        Ok(texture) => texture,
        Err(err) => {
            log::error!("{err}");
            std::process::exit(1);
        }
    };
    let env = match cli.load_env() {
        Ok(env) => env,
        Err(err) => {
            log::error!("{err}");
            std::process::exit(1);
        }
    };
    let has_env = env.is_some();
    // A texture only maps meaningfully onto a UV-mapped mesh; warn otherwise so a
    // flat/"wrong" Textured render is explained rather than mysterious.
    if texture.is_some() && !mesh_has_uvs(&loaded.mesh) {
        log::warn!(
            "the loaded mesh has no UV coordinates; the bound texture will sample \
             a single texel in Textured mode. Use a UV-mapped mesh, e.g. \
             assets/meshes/bunny_with_texture/bunny.obj"
        );
    }

    let albedo = texture.as_ref().map(|t| t as &dyn trd_core::Texture);
    let mut renderer = match pollster::block_on(GuiRenderer::new(
        std::slice::from_ref(&loaded.mesh),
        &[albedo],
        &[],
        env,
        cli.width,
        cli.height,
    )) {
        Ok(renderer) => renderer,
        Err(err) => {
            log::error!("failed to create renderer: {err}");
            std::process::exit(1);
        }
    };
    if let Some(parts) = &loaded.parts {
        // `--texture` wins over the GLB's own base colour, so an explicit flag
        // is never silently ignored.
        let mut parts = parts.clone();
        if let Some(texture) = &texture {
            for part in &mut parts {
                part.base_color_texture = Some(texture.clone());
            }
        }
        if let Err(err) = renderer.set_mesh_parts(0, &parts) {
            log::error!("failed to bind the model's material parts: {err}");
            std::process::exit(1);
        }
    }

    let controller = InteractionController::new(cli.scene_state(&loaded, has_env));
    let app = TrdGuiApp::new(controller, renderer);

    // Size the window to the render plus the side panel, so a high `--width`/
    // `--height` is shown at native scale instead of being scaled down into a
    // fixed 900x700 — clamped so an oversized render still opens on screen.
    const PANEL_WIDTH: f32 = 300.0;
    const MAX_WINDOW: [f32; 2] = [2400.0, 1400.0];
    let window = [
        (cli.width as f32 + PANEL_WIDTH).clamp(900.0, MAX_WINDOW[0]),
        (cli.height as f32).clamp(700.0, MAX_WINDOW[1]),
    ];
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(window),
        ..Default::default()
    };
    eframe::run_native("trd-gui", options, Box::new(|_cc| Ok(Box::new(app))))
}
