mod app;
mod cli;
mod error;
mod media;

fn main() {
    if let Err(error) = run() {
        log::error!("{error}");
        std::process::exit(1);
    }
}

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
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };
    let preview_width = cli.preview_width;
    eframe::run_native(
        "trd GUI video editing",
        options,
        // Built inside the creator so the app can adopt eframe's own wgpu device
        // — `CreationContext::wgpu_render_state` exists nowhere else. One device
        // per process is what lets the rendered texture be bound straight into
        // egui instead of round-tripping through CPU memory (#229).
        Box::new(move |context| {
            let gpu = context.wgpu_render_state.as_ref().map(|state| {
                trd_core::GpuContext::adopt(
                    state.adapter.clone(),
                    state.device.clone(),
                    state.queue.clone(),
                )
            });
            if gpu.is_none() {
                log::warn!("eframe has no wgpu render state; falling back to a private device");
            }
            let app = app::NativeVideoEditingApp::new(document, video_source, preview_width, gpu)?;
            Ok(Box::new(app))
        }),
    )?;
    Ok(())
}
