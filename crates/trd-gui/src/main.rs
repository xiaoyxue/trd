//! trd-gui native binary: parse args, load the mesh, build the in-process
//! render backend, and run the eframe/egui window (issue #97).
//!
//! The interactive stack (eframe/egui + `trd-core`'s native `BatchRenderer`) is
//! native-only; on wasm the crate compiles to an empty `main` so workspace-wide
//! wasm builds skip it (mirroring `trd-app`). The browser target — egui on a
//! canvas with `trd-core`'s offscreen wasm renderer — is a later slice in #97.

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    use clap::Parser;
    use trd_gui::app::TrdGuiApp;
    use trd_gui::cli::{Backend, Cli};
    use trd_gui::interaction::InteractionController;
    use trd_gui::render_backend::{
        mesh_has_uvs, ArrowRoundTripRenderer, InProcRenderer, SceneRenderer,
    };

    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,trd_gui=info,trd_core=info"),
    )
    .init();

    let cli = Cli::parse();

    // Load the mesh and build the backend up front so a bad mesh / adapter fails
    // fast on the console rather than inside the window.
    let mesh = match cli.load_mesh() {
        Ok(mesh) => mesh,
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
    // A texture only maps meaningfully onto a UV-mapped mesh; warn otherwise so a
    // flat/"wrong" Textured render is explained rather than mysterious.
    if texture.is_some() && !mesh_has_uvs(&mesh) {
        log::warn!(
            "the loaded mesh has no UV coordinates; the bound texture will sample \
             a single texel in Textured mode. Use a UV-mapped mesh, e.g. \
             assets/meshes/bunny_with_texture/bunny.obj"
        );
    }

    let renderer: Result<Box<dyn SceneRenderer>, _> = match cli.backend {
        Backend::Inproc => InProcRenderer::new(
            &[mesh],
            texture.as_ref().map(|t| t as &dyn trd_core::Texture),
            env,
            cli.width,
            cli.height,
        )
        .map(|r| Box::new(r) as Box<dyn SceneRenderer>),
        Backend::Arrow => ArrowRoundTripRenderer::new(
            &[mesh],
            texture.as_ref().map(|t| t as &dyn trd_core::Texture),
            env,
            cli.width,
            cli.height,
        )
        .map(|r| Box::new(r) as Box<dyn SceneRenderer>),
    };
    let renderer = match renderer {
        Ok(renderer) => renderer,
        Err(err) => {
            log::error!("failed to create renderer: {err}");
            std::process::exit(1);
        }
    };

    let controller = InteractionController::new(cli.scene_state());
    let app = TrdGuiApp::new(controller, renderer);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 700.0]),
        ..Default::default()
    };
    eframe::run_native("trd-gui", options, Box::new(|_cc| Ok(Box::new(app))))
}

#[cfg(target_arch = "wasm32")]
fn main() {}
