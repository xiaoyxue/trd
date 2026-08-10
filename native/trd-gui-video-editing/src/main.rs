#[cfg(not(target_arch = "wasm32"))]
mod app;
#[cfg(not(target_arch = "wasm32"))]
mod cli;
#[cfg(not(target_arch = "wasm32"))]
mod error;
#[cfg(not(target_arch = "wasm32"))]
mod media;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    if let Err(error) = run() {
        log::error!("{error}");
        std::process::exit(1);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run() -> Result<(), error::NativeVideoEditingError> {
    use clap::Parser;

    env_logger::Builder::from_env(
        env_logger::Env::default()
            .default_filter_or("warn,trd_gui_video_editing=info,trd_core=info"),
    )
    .init();

    let cli = cli::Cli::parse();
    let bytes =
        std::fs::read(&cli.document).map_err(|source| error::NativeVideoEditingError::Read {
            path: cli.document.display().to_string(),
            source,
        })?;
    let document = trd_core::decode_video_editing_document(&bytes)?;
    let video_source = cli
        .video
        .map(media::NativeVideoSource::Local)
        .or_else(|| cli.video_url.map(media::NativeVideoSource::Url));
    if cli.probe_only {
        if let Some(source) = video_source {
            let video = media::NativeVideo::open(source, &document.video, cli.preview_width)?;
            video.decode_one(0)?;
            println!("native video-editing source validated; decoded frame 0");
        } else {
            println!("native video-editing document validated; no video source supplied");
        }
        return Ok(());
    }
    let app = app::NativeVideoEditingApp::new(document, video_source, cli.preview_width)?;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };
    eframe::run_native(
        "trd GUI video editing",
        options,
        Box::new(|_context| Ok(Box::new(app))),
    )?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
