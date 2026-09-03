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

/// What `--probe-only` reports: the frame asked for, the frame that came back,
/// and the timestamp it carries.
///
/// The two indices are printed even when they agree, because agreement is the
/// finding — a variable-rate container can hand back the neighbouring picture,
/// and the old message asserted "decoded frame 0" without ever checking (#319).
fn describe(wanted: u32, frame: &media::DecodedFrame) -> String {
    let landed = if frame.index == wanted {
        format!("decoded frame {wanted}")
    } else {
        format!(
            "asked for frame {wanted}, decoded frame {} instead",
            frame.index
        )
    };
    format!(
        "{landed} at {:.6}s (duration {:.6}s)",
        frame.media_time_seconds, frame.duration_seconds
    )
}

fn run() -> Result<(), error::NativeVideoEditingError> {
    use clap::Parser;

    env_logger::Builder::from_env(
        env_logger::Env::default()
            .default_filter_or("warn,trd_gui_video_editing=info,trd_core=info"),
    )
    .init();

    let cli = cli::Cli::parse();
    // Arrow input is optional: without one the editor is a plain player and the
    // container supplies the timeline (#264).
    let input = cli
        .document
        .as_ref()
        .map(|path| {
            let bytes =
                std::fs::read(path).map_err(|source| error::NativeVideoEditingError::Read {
                    path: path.display().to_string(),
                    source,
                })?;
            trd_gui::video_editing::decode_video_editing_input(&bytes)
                .map_err(error::NativeVideoEditingError::Input)
        })
        .transpose()?;
    let video_source = cli
        .video
        .map(media::NativeVideoSource::Local)
        .or_else(|| cli.video_url.map(media::NativeVideoSource::Url));
    if !cli.probe_only
        && video_source.is_none()
        && matches!(
            input.as_ref(),
            Some(trd_gui::video_editing::VideoEditingInput::Scene(_))
        )
    {
        return Err(error::NativeVideoEditingError::Input(
            "an exported protocol scene requires --video or --video-url".to_owned(),
        ));
    }
    if cli.probe_only {
        let wanted = cli.probe_frame;
        match (video_source, input.as_ref()) {
            (
                Some(source),
                Some(trd_gui::video_editing::VideoEditingInput::Annotation(document)),
            ) => {
                let (video, _) =
                    media::NativeVideo::open(source, &document.video, cli.preview_width)?;
                let frame = video.decode_one(wanted)?;
                println!(
                    "native video-editing source validated; {}",
                    describe(wanted, &frame)
                );
            }
            (Some(source), _) => {
                let (video, info) = media::NativeVideo::probe(source, cli.preview_width)?;
                let frame = video.decode_one(wanted)?;
                println!(
                    "native video probed: {}x{} · {}/{} fps · {} frames; {}",
                    info.width,
                    info.height,
                    info.fps_num,
                    info.fps_den,
                    info.frame_count,
                    describe(wanted, &frame)
                );
            }
            (None, Some(trd_gui::video_editing::VideoEditingInput::Annotation(_))) => println!(
                "native video-editing annotation document validated; no video source supplied"
            ),
            (None, Some(trd_gui::video_editing::VideoEditingInput::Scene(scene))) => println!(
                "protocol scene validated: {} mesh row(s), {} params row(s); no video source supplied",
                scene.meshes.len(),
                scene.frames.len()
            ),
            (None, None) => println!("nothing to probe: pass --video and/or --document"),
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
            let app = app::NativeVideoEditingApp::new(input, video_source, preview_width, gpu)?;
            Ok(Box::new(app))
        }),
    )?;
    Ok(())
}
