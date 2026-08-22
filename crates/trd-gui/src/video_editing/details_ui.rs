//! The Details inspector, drawn in immediate mode (#175).
//!
//! There is no snapshot DTO: each row reads the value it shows at draw time —
//! document metadata straight from [`VideoEditingApp::document`], live host
//! observations from the shared state, and the frame-dependent values from
//! [`DisplayedFacts`], which pins them to the frame actually on screen.
//!
//! The row list exists once. [`EguiRows`] draws it; [`TextRows`] renders the
//! same calls as flat `label: value` text for the Copy button, so the panel and
//! the clipboard cannot drift apart.

use super::diagnostics::{format_matrix, pbr_debug_view_label};
use super::diagnostics::{quad_frame_diagnostics, render_mode_label, tone_map_label};
use super::{DisplayedFacts, VideoSourceKind};

/// One Details section: the rows it emits, independent of how they are shown.
///
/// Takes the document metadata by reference and the frame-dependent values from
/// [`DisplayedFacts`]; nothing is copied into an intermediate snapshot.
type Section = fn(&trd_core::VideoInfo, &DisplayedFacts, &mut dyn Rows);

const SECTIONS: [(&str, Section); 6] = [
    ("Source", source_rows),
    ("Timeline / synchronization", timeline_rows),
    ("Tracking / quad frame", tracking_rows),
    ("Placement / object", placement_rows),
    ("Material / lighting", material_rows),
    ("Renderer", renderer_rows),
];

pub(super) fn details_ui(ui: &mut egui::Ui, video: &trd_core::VideoInfo, facts: &DisplayedFacts) {
    ui.horizontal(|ui| {
        if ui.small_button("Copy details").clicked() {
            let mut text = TextRows::default();
            for (title, section) in SECTIONS {
                text.section(title);
                section(video, facts, &mut text);
            }
            ui.ctx().copy_text(text.0);
        }
        ui.weak("Values follow the displayed render.");
    });

    for (title, section) in SECTIONS {
        ui.collapsing(title, |ui| {
            egui::Grid::new(title)
                .num_columns(3)
                .striped(true)
                .show(ui, |ui| section(video, facts, &mut EguiRows(ui)));
        });
    }
}

// ── row sinks ───────────────────────────────────────────────────────────────

/// The row vocabulary shared by the panel and the clipboard.
trait Rows {
    fn row(&mut self, label: &str, value: &str);
    fn match_row(&mut self, label: &str, expected: &str, observed: Option<&str>);
    fn warning(&mut self, label: &str, value: &str);
    fn error(&mut self, label: &str, value: &str);

    fn match_float(
        &mut self,
        label: &str,
        expected: f64,
        observed: Option<f64>,
        tolerance: f64,
        unit: &str,
    ) {
        let expected_text = format!("{expected:.6}{unit}");
        let observed_text = observed.map(|value| format!("{value:.6}{unit}"));
        match observed {
            Some(value) if (value - expected).abs() <= tolerance => {
                self.match_row(label, &expected_text, observed_text.as_deref());
            }
            _ => self.match_row(label, &expected_text, observed_text.as_deref()),
        }
    }

    fn optional_error(&mut self, label: &str, value: Option<&str>) {
        match value {
            Some(value) => self.error(label, value),
            None => self.row(label, "none"),
        }
    }
}

struct EguiRows<'a>(&'a mut egui::Ui);

impl Rows for EguiRows<'_> {
    fn row(&mut self, label: &str, value: &str) {
        self.0.label(label);
        self.0.monospace(value);
        self.0.label("");
        self.0.end_row();
    }

    fn match_row(&mut self, label: &str, expected: &str, observed: Option<&str>) {
        self.0.label(label);
        self.0.monospace(format!(
            "expected {expected} / observed {}",
            observed.unwrap_or("none")
        ));
        match observed {
            Some(observed) if observed == expected => {
                self.0.colored_label(egui::Color32::LIGHT_GREEN, "MATCH");
            }
            Some(_) => {
                self.0.colored_label(egui::Color32::LIGHT_RED, "MISMATCH");
            }
            None => {
                self.0.colored_label(egui::Color32::YELLOW, "NOT OBSERVED");
            }
        }
        self.0.end_row();
    }

    fn warning(&mut self, label: &str, value: &str) {
        self.0.label(label);
        self.0.monospace(value);
        self.0.colored_label(egui::Color32::YELLOW, "WARNING");
        self.0.end_row();
    }

    fn error(&mut self, label: &str, value: &str) {
        self.0.label(label);
        self.0.monospace(value);
        self.0.colored_label(egui::Color32::LIGHT_RED, "ERROR");
        self.0.end_row();
    }
}

/// Renders the same row calls as flat text for the clipboard.
#[derive(Default)]
struct TextRows(String);

impl TextRows {
    fn section(&mut self, title: &str) {
        self.0.push_str(&format!("\n[{title}]\n"));
    }
}

impl Rows for TextRows {
    fn row(&mut self, label: &str, value: &str) {
        self.0.push_str(&format!("{label}: {value}\n"));
    }

    fn match_row(&mut self, label: &str, expected: &str, observed: Option<&str>) {
        let status = match observed {
            Some(observed) if observed == expected => "MATCH",
            Some(_) => "MISMATCH",
            None => "NOT OBSERVED",
        };
        self.0.push_str(&format!(
            "{label}: expected {expected} / observed {} [{status}]\n",
            observed.unwrap_or("none")
        ));
    }

    fn warning(&mut self, label: &str, value: &str) {
        self.0.push_str(&format!("{label}: {value} [WARNING]\n"));
    }

    fn error(&mut self, label: &str, value: &str) {
        self.0.push_str(&format!("{label}: {value} [ERROR]\n"));
    }
}

// ── sections ────────────────────────────────────────────────────────────────

fn source_rows(video: &trd_core::VideoInfo, facts: &DisplayedFacts, r: &mut dyn Rows) {
    let shared = &facts.shared;
    let observed = shared.video_source.borrow();
    let observed = observed.as_ref();
    let metadata = shared.video_metadata.get();
    let media = shared.video_media.get();

    r.row(
        "kind",
        observed.map_or("not loaded", |source| source_kind_label(source.kind)),
    );
    r.match_row(
        "name",
        &video.source_name,
        observed.map(|source| source.name.as_str()),
    );
    r.match_row(
        "byte length",
        &video.byte_length.to_string(),
        observed
            .and_then(|source| source.byte_length)
            .map(|value| value.to_string())
            .as_deref(),
    );
    r.row("declared MIME", &video.mime);
    r.row("declared codec", &video.codec);
    r.match_row(
        "dimensions",
        &format!("{}x{}", video.width, video.height),
        metadata
            .map(|m| format!("{}x{}", m.width, m.height))
            .as_deref(),
    );
    r.row("FPS", &format!("{}/{}", video.fps_num, video.fps_den));
    r.row("frame count", &video.frame_count.to_string());
    r.row(
        "unpresented tail",
        &match video.unpresented_tail_samples {
            None => "unknown".to_owned(),
            Some(0) => "none".to_owned(),
            Some(1) => "1 sample (AV_PKT_FLAG_DISCARD)".to_owned(),
            Some(n) => format!("{n} samples (AV_PKT_FLAG_DISCARD)"),
        },
    );
    r.match_float(
        "duration",
        video.duration_us as f64 / 1_000_000.0,
        metadata.map(|m| m.duration_seconds),
        f64::from(video.fps_den) / f64::from(video.fps_num.max(1)),
        "s",
    );
    r.row("SHA-256", &video.sha256);
    r.warning("digest", "not browser-verified yet");
    r.row("media readyState", &media.ready_state.to_string());
    r.row("loaded", &shared.video_loaded.get().to_string());
    r.row("playing", &shared.video_playing.get().to_string());
    r.row("ended", &media.ended.to_string());
    r.optional_error("error", shared.error_text().as_deref());
}

fn timeline_rows(video: &trd_core::VideoInfo, facts: &DisplayedFacts, r: &mut dyn Rows) {
    let shared = &facts.shared;
    let frame = facts.timeline_frame.as_ref();

    r.row("media time", &option_f64(facts.media_time_seconds, "s"));
    r.row(
        "frame duration",
        &option_f64(facts.frame_duration_seconds, "s"),
    );
    r.row("requested frame", &facts.requested_frame_index.to_string());
    r.row("presented frame", &option_u32(facts.presented_frame_index));
    r.row("displayed frame", &option_u32(facts.frame_index));
    r.row("rendered frame", &option_u32(facts.rendered_frame_index));
    r.row(
        "Arrow video_frame_index",
        &option_u32(frame.map(|f| f.video_frame_index)),
    );
    r.row(
        "Arrow present_index",
        &option_u32(frame.map(|f| f.present_index)),
    );
    r.row(
        "Arrow timestamp_us",
        &frame.map_or_else(|| "none".to_owned(), |f| f.timestamp_us.to_string()),
    );
    r.row(
        "media/row delta",
        &option_f64(
            facts
                .media_time_seconds
                .zip(frame)
                .map(|(time, frame)| (time - frame.timestamp_us as f64 / 1_000_000.0) * 1_000.0),
            "ms",
        ),
    );
    r.row(
        "tracking state",
        frame.map_or("none", |f| if f.tracked { "tracked" } else { "video-only" }),
    );
    r.row("source size", &format!("{}x{}", video.width, video.height));
    r.row(
        "render size",
        &format!(
            "{}x{}",
            facts.render_target_size.0, facts.render_target_size.1
        ),
    );
    r.row(
        "source generation",
        &shared.source_generation.get().to_string(),
    );
    r.row("render revision", &shared.render_revision.get().to_string());
    r.row(
        "pending render",
        &option_u64(
            shared
                .needs_overlay
                .get()
                .then_some(shared.render_revision.get()),
        ),
    );
    r.row("in-flight frame", &option_u32(facts.in_flight_frame_index));
    r.row("coalesced frame", &option_u32(facts.coalesced_frame_index));
    r.row(
        "last render",
        &option_f64(shared.last_render_latency_ms.get(), "ms"),
    );
    r.row("seek target", &option_u32(facts.seek_target));
    r.row("seek pending", &facts.seek_target.is_some().to_string());
}

fn tracking_rows(_video: &trd_core::VideoInfo, facts: &DisplayedFacts, r: &mut dyn Rows) {
    match facts
        .timeline_frame
        .as_ref()
        .and_then(|frame| frame.placement_quad)
    {
        Some(points) => {
            for (label, point) in ["TL", "TR", "BR", "BL"].into_iter().zip(points) {
                r.row(label, &vec2_label(point));
            }
        }
        None => r.row("quad points", "none"),
    }
    match facts.timeline_frame.as_ref().and_then(|frame| frame.k) {
        Some(k) => r.row(
            "K (fx, fy, cx, cy)",
            &format!("{:.4}, {:.4}, {:.4}, {:.4}", k[0], k[4], k[2], k[5]),
        ),
        None => r.row("K", "none"),
    }
    if let Some(frame) = facts.quad.map(quad_frame_diagnostics) {
        r.row("origin", &vec3_label(frame.origin));
        r.row("e1", &vec3_label(frame.e1));
        r.row("e2", &vec3_label(frame.e2));
        r.row("e3", &vec3_label(frame.e3));
        r.row(
            "half-edge lengths",
            &format!(
                "{:.6}, {:.6}",
                frame.half_edge_lengths[0], frame.half_edge_lengths[1]
            ),
        );
        r.row("axis length", &format!("{:.6}", frame.axis_length));
        r.row(
            "|dot(e1,e2/e3), dot(e2,e3)|",
            &format!(
                "{:.6}, {:.6}, {:.6}",
                frame.orthogonality_errors[0],
                frame.orthogonality_errors[1],
                frame.orthogonality_errors[2]
            ),
        );
        r.row(
            "handedness determinant",
            &format!("{:.6}", frame.handedness_determinant),
        );
    }
    match &facts.pose_delta {
        Some(delta) => {
            r.row(
                "previous tracked frame",
                &delta.previous_frame_index.to_string(),
            );
            r.row(
                "pose translation delta",
                &format!("{:.6}", delta.translation),
            );
            r.row(
                "pose rotation delta",
                &format!("{:.4} deg", delta.rotation_degrees),
            );
            r.row(
                "axis-length ratio",
                &format!("{:.6}", delta.axis_length_ratio),
            );
        }
        None => r.row("pose delta", "none"),
    }
    if facts.normal_sign_warning {
        r.warning("continuity", "quad normal sign changed");
    } else {
        r.row("continuity", "normal sign continuous");
    }
    match &facts.placement_error {
        Some(error) => r.error("placement error", &error.to_string()),
        None => r.row("placement error", "none"),
    }
    r.row("tracking smoothing", "off");
}

fn placement_rows(_video: &trd_core::VideoInfo, facts: &DisplayedFacts, r: &mut dyn Rows) {
    let scene = &facts.scene;
    let object = scene.objects[0];
    let asset = facts
        .renderer
        .as_ref()
        .and_then(|renderer| renderer.asset.as_ref());

    r.row("selected quad", &facts.selected_quad.to_string());
    r.row("selected object", &option_u32(scene.selected));
    r.row(
        "catalog asset",
        facts
            .selected_asset
            .map_or("none", super::CatalogAsset::label),
    );
    r.row(
        "source format",
        asset.map_or("none", |facts| facts.source_format),
    );
    r.row(
        "preview AABB min",
        &asset.map_or_else(|| "none".to_owned(), |a| vec3_label(a.aabb_min)),
    );
    r.row(
        "preview AABB max",
        &asset.map_or_else(|| "none".to_owned(), |a| vec3_label(a.aabb_max)),
    );
    r.row(
        "preview scale",
        &asset.map_or_else(|| "none".to_owned(), |a| format!("{:.6}", a.preview_scale)),
    );
    r.row("Olympic preset", "size 0.24, e1 1.30, e2 -1.70, lift 1.00");
    r.row("object translation", &vec3_label(object.translation));
    r.row(
        "object rotation",
        &format!(
            "yaw {:.3}, pitch {:.3}, roll {:.3} deg",
            object.yaw.to_degrees(),
            object.pitch.to_degrees(),
            object.roll.to_degrees()
        ),
    );
    r.row("object scale", &vec3_label(object.scale));
    r.row("movement basis", &facts.movement_basis.join(" / "));
    r.row("visibility", facts.visibility_reason);
    r.row(
        "draw_model",
        &facts
            .draw_model
            .map_or_else(|| "none".to_owned(), format_matrix),
    );
}

fn material_rows(_video: &trd_core::VideoInfo, facts: &DisplayedFacts, r: &mut dyn Rows) {
    let scene = &facts.scene;
    let material = &scene.materials[0];
    let ibl = scene.image_based_lighting[0];
    let tone_mapping = scene.tone_mappings[0];
    let imported = facts
        .renderer
        .as_ref()
        .and_then(|renderer| renderer.asset.as_ref())
        .map(|asset| &asset.imported_material);

    r.row("render mode", render_mode_label(scene.modes[0]));
    r.row(
        "imported metallic",
        &option_f32(imported.map(|m| m.metallic)),
    );
    r.row(
        "imported roughness",
        &option_f32(imported.map(|m| m.roughness)),
    );
    r.row(
        "base-color map",
        yes_no(imported.is_some_and(|m| m.auxiliary.textures.base_color)),
    );
    r.row(
        "metallic-roughness map",
        yes_no(imported.is_some_and(|m| m.auxiliary.textures.metallic_roughness)),
    );
    r.row(
        "normal map",
        yes_no(imported.is_some_and(|m| m.auxiliary.textures.normal)),
    );
    r.row("metallic", &format!("{:.4}", material.metallic));
    r.row("roughness", &format!("{:.4}", material.roughness));
    r.row("specular", &format!("{:.4}", material.specular));
    r.row("clearcoat", &format!("{:.4}", material.clearcoat));
    r.row(
        "IBL",
        if scene.environment_available {
            "uffizi-large.hdr"
        } else {
            "none"
        },
    );
    r.row(
        "IBL gain (object x scene)",
        &format!(
            "{:.4} x {:.4}",
            ibl.intensity, scene.lighting.environment.intensity
        ),
    );
    r.row(
        "environment yaw",
        &format!(
            "{:.3} deg",
            scene.lighting.environment.rotation.to_degrees()
        ),
    );
    r.row(
        "direct light / ambient",
        &format!(
            "{:.4} / {:.4}",
            scene.lighting.scale, scene.lighting.ambient
        ),
    );
    r.row("exposure", &format!("{:.4}", tone_mapping.exposure));
    r.row("tone map", tone_map_label(tone_mapping.operator));
    r.row("PBR debug", pbr_debug_view_label(scene.pbr_debug_views[0]));
    if facts.reflective_tracking_warning {
        r.warning(
            "tracking/material",
            "reflective/normal-mapped material may amplify raw tracking jitter",
        );
    }
}

/// Human-readable byte count for the transfer rows.
///
/// A frame is megabytes and a `0` must stay visibly a zero, so the unit follows
/// the magnitude rather than being fixed.
fn bytes_label(bytes: usize) -> String {
    match bytes {
        0 => "0 B".to_owned(),
        b if b < 1024 => format!("{b} B"),
        b if b < 1024 * 1024 => format!("{:.1} KiB", b as f64 / 1024.0),
        b => format!("{:.2} MiB", b as f64 / (1024.0 * 1024.0)),
    }
}

fn renderer_rows(video: &trd_core::VideoInfo, facts: &DisplayedFacts, r: &mut dyn Rows) {
    let renderer = facts.renderer.as_ref();
    let identity = renderer.map(|renderer| &renderer.identity);

    r.row(
        "adapter",
        identity.map_or("none", |id| id.adapter_name.as_str()),
    );
    r.row("backend", identity.map_or("none", |id| id.backend.as_str()));
    r.row(
        "device type",
        identity.map_or("none", |id| id.device_type.as_str()),
    );
    r.row("source size", &format!("{}x{}", video.width, video.height));
    r.row(
        "render target",
        &format!(
            "{}x{}",
            facts.render_target_size.0, facts.render_target_size.1
        ),
    );
    r.row("mode", render_mode_label(facts.scene.modes[0]));
    // Observed frame-path CPU↔GPU traffic for the last frame, so a later claim
    // that a copy is gone is read off a meter rather than asserted (#229).
    //
    // Labelled "frame path" because that is the scope: full-resolution image
    // data. Per-frame uniforms and egui's own geometry upload still cross the
    // boundary and are not counted here.
    if let Some(transfers) = renderer.map(|renderer| renderer.transfers) {
        r.row("frame-path crossings", &transfers.crossings().to_string());
        r.row("  frame upload", &bytes_label(transfers.frame_upload));
        r.row("  readback", &bytes_label(transfers.readback));
        r.row("  ui upload", &bytes_label(transfers.ui_upload));
        r.row("  total / frame", &bytes_label(transfers.total_bytes()));
    }
    r.row(
        "MSAA",
        &renderer.map_or_else(
            || "unknown".to_owned(),
            |renderer| format!("{}x", renderer.msaa_samples),
        ),
    );
    r.row(
        "drawables (background/foreground/selection)",
        &format!(
            "{}/{}/{}",
            facts.background_drawables, facts.foreground_drawables, facts.selection_drawables
        ),
    );
    r.row(
        "frame texture upload",
        &facts
            .shared
            .latest_video_frame
            .borrow()
            .as_ref()
            .map_or_else(
                || "none".to_owned(),
                |frame| format!("{} bytes", frame.rgba.len()),
            ),
    );
    r.row(
        "pick target",
        &renderer
            .and_then(|renderer| renderer.pick_target_size)
            .map_or_else(
                || "none".to_owned(),
                |(width, height)| format!("{width}x{height}"),
            ),
    );
    r.row(
        "latest pick",
        &facts.latest_pick_result.map_or_else(
            || "none".to_owned(),
            |hit| hit.map_or_else(|| "miss".to_owned(), |id| format!("object {id}")),
        ),
    );
    r.optional_error(
        "last render error",
        facts.shared.last_render_error.borrow().as_deref(),
    );
    r.optional_error(
        "last pick error",
        facts.shared.last_pick_error.borrow().as_deref(),
    );
}

// ── value formatting ────────────────────────────────────────────────────────

fn source_kind_label(kind: VideoSourceKind) -> &'static str {
    match kind {
        VideoSourceKind::LocalFile => "local file",
        VideoSourceKind::HttpUrl => "HTTP(S) URL",
    }
}

fn option_u32(value: Option<u32>) -> String {
    value.map_or_else(|| "none".to_owned(), |value| value.to_string())
}

fn option_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "none".to_owned(), |value| value.to_string())
}

fn option_f32(value: Option<f32>) -> String {
    value.map_or_else(|| "none".to_owned(), |value| format!("{value:.4}"))
}

fn option_f64(value: Option<f64>, unit: &str) -> String {
    value.map_or_else(|| "none".to_owned(), |value| format!("{value:.3} {unit}"))
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

fn vec2_label(value: [f32; 2]) -> String {
    format!("[{:.3}, {:.3}]", value[0], value[1])
}

fn vec3_label(value: [f32; 3]) -> String {
    format!("[{:.6}, {:.6}, {:.6}]", value[0], value[1], value[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_rows_render_the_same_calls_as_the_panel() {
        let mut rows = TextRows::default();
        rows.section("Source");
        rows.row("declared codec", "h264");
        rows.match_row("name", "shot.mp4", Some("other.mp4"));
        rows.match_row("size", "1920x1080", None);
        rows.warning("digest", "not browser-verified yet");
        rows.optional_error("error", None);
        rows.optional_error("last render error", Some("device lost"));

        assert_eq!(
            rows.0,
            "\n[Source]\n\
             declared codec: h264\n\
             name: expected shot.mp4 / observed other.mp4 [MISMATCH]\n\
             size: expected 1920x1080 / observed none [NOT OBSERVED]\n\
             digest: not browser-verified yet [WARNING]\n\
             error: none\n\
             last render error: device lost [ERROR]\n"
        );
    }

    /// The transfer meter is only useful if its rows are readable **and** its
    /// crossing count follows the bytes, so both are pinned here rather than
    /// checked by eye in the panel.
    #[test]
    fn transfer_rows_report_bytes_and_derive_the_crossing_count() {
        let readback_path = crate::video_editing_renderer::TransferCounts {
            frame_upload: 2_764_800,
            readback: 2_764_800,
            ui_upload: 2_764_800,
        };
        assert_eq!(readback_path.crossings(), 3);
        assert_eq!(readback_path.total_bytes(), 8_294_400);

        // What a shared-device path must look like: the copies are absent, not
        // merely small.
        let bound_directly = crate::video_editing_renderer::TransferCounts {
            frame_upload: 2_764_800,
            ..Default::default()
        };
        assert_eq!(bound_directly.crossings(), 1);
        assert_eq!(
            crate::video_editing_renderer::TransferCounts::default().crossings(),
            0
        );

        let mut rows = TextRows::default();
        rows.row(
            "frame-path crossings",
            &readback_path.crossings().to_string(),
        );
        rows.row("  frame upload", &bytes_label(readback_path.frame_upload));
        rows.row("  readback", &bytes_label(readback_path.readback));
        rows.row("  ui upload", &bytes_label(bound_directly.ui_upload));
        rows.row("  total / frame", &bytes_label(readback_path.total_bytes()));

        assert_eq!(
            rows.0,
            "frame-path crossings: 3\n  \
             frame upload: 2.64 MiB\n  \
             readback: 2.64 MiB\n  \
             ui upload: 0 B\n  \
             total / frame: 7.91 MiB\n"
        );
    }

    #[test]
    fn bytes_label_switches_unit_with_magnitude() {
        assert_eq!(bytes_label(0), "0 B");
        assert_eq!(bytes_label(512), "512 B");
        assert_eq!(bytes_label(1024), "1.0 KiB");
        assert_eq!(bytes_label(1024 * 1024), "1.00 MiB");
    }
}
