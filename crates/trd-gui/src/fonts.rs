//! The one font set, shared by every surface.
//!
//! `eframe`'s `default_fonts` feature embeds four whole font files —
//! Ubuntu-Light, Hack, Noto Emoji and emoji-icon-font, 1,414,018 bytes measured
//! in the shipped wasm, 57% of its data section (#359). Two of those four are
//! emoji sets this UI never draws, and the other two ship every glyph they have
//! for every script.
//!
//! So the feature is off and these are **subsets**, cut to the ranges the UI
//! actually uses (`scripts/subset_fonts.py`): 44,780 bytes for the pair, a 97%
//! reduction. Both surfaces install the same two, so the browser and the native
//! window keep rendering identically — which is the point of one render core,
//! and what lets the `gui_render` goldens cover both.
//!
//! What is deliberately lost: emoji, and any script outside the subset ranges.
//! Text outside them renders as `□`. `scripts/subset_fonts.py --check` reads the
//! generated cmaps and fails if any UI string needs a glyph none of them has —
//! the only guard there is, since a missing glyph panics nothing and logs
//! nothing.

use std::sync::Arc;

/// Proportional UI text.
const UBUNTU_LIGHT_SUBSET: &[u8] = include_bytes!("../../../assets/fonts/Ubuntu-Light.subset.ttf");
/// Monospace, and the fallback for arrows Ubuntu lacks (`→`, `⇒`).
const HACK_REGULAR_SUBSET: &[u8] = include_bytes!("../../../assets/fonts/Hack-Regular.subset.ttf");
/// Cut to almost nothing, but it is the only one of egui's four fonts carrying
/// `⏴⏵⏶⏷` — the arrows every `CollapsingHeader` and `DragValue` draws.
const ICON_SUBSET: &[u8] = include_bytes!("../../../assets/fonts/emoji-icon-font.subset.ttf");

/// Installs trd's fonts on `ctx`, replacing egui's defaults.
///
/// Call once per surface, before the first frame.
pub fn install(ctx: &egui::Context) {
    ctx.set_fonts(definitions());
}

fn definitions() -> egui::FontDefinitions {
    let mut fonts = egui::FontDefinitions::empty();
    for (name, bytes) in [
        ("ubuntu-light", UBUNTU_LIGHT_SUBSET),
        ("hack", HACK_REGULAR_SUBSET),
        ("icons", ICON_SUBSET),
    ] {
        fonts.font_data.insert(
            name.to_owned(),
            Arc::new(egui::FontData::from_static(bytes)),
        );
    }
    // Each family leads with its own font and falls back through the others, so
    // a glyph missing from one is still drawn rather than becoming a `□`.
    fonts.families.insert(
        egui::FontFamily::Proportional,
        vec![
            "ubuntu-light".to_owned(),
            "hack".to_owned(),
            "icons".to_owned(),
        ],
    );
    fonts.families.insert(
        egui::FontFamily::Monospace,
        vec![
            "hack".to_owned(),
            "ubuntu-light".to_owned(),
            "icons".to_owned(),
        ],
    );
    fonts
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A font family with no font renders every glyph as `□`, and the failure is
    /// silent — no panic, no log, just an unreadable window. Both families are
    /// populated here rather than trusted to `FontDefinitions::empty`.
    #[test]
    fn both_families_have_a_font_and_fallbacks() {
        let fonts = definitions();
        for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
            let names = fonts
                .families
                .get(&family)
                .unwrap_or_else(|| panic!("{family:?} has no entry"));
            assert_eq!(names.len(), 3, "{family:?} should list every font");
            for name in names {
                assert!(
                    fonts.font_data.contains_key(name),
                    "{family:?} names {name}, which has no font data"
                );
            }
        }
    }

    /// The subsets are the whole point: the four fonts egui would have embedded
    /// are 1,414,018 bytes. If a regeneration ever drops the `--unicodes` cut,
    /// the bundle silently regains more than a megabyte.
    #[test]
    fn the_embedded_fonts_are_subsets() {
        let total = UBUNTU_LIGHT_SUBSET.len() + HACK_REGULAR_SUBSET.len() + ICON_SUBSET.len();
        assert!(
            total < 300_000,
            "embedded fonts total {total} bytes — subsetting has regressed"
        );
    }
}
