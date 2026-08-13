//! Per-drawable **configuration**: the mesh render mode, the background frame's
//! fit, and which coordinate plane a grid gizmo lattices.
//!
//! Each belongs to one [`DrawableObject`](super::DrawableObject) variant —
//! [`RenderMode`] to `Mesh`, [`FrameFit`] to the background frame plane, [`GridPlane`] to
//! `PlaneGrid` — and each is a value a **front-end selects** (all three are
//! `trd-cli` flags). They are plain configuration: no geometry, no GPU state, no
//! dependency on the rest of the scene model, so adding a render mode does not
//! mean editing the primitive list.
//!
//! Distinct from [`RenderOptions`](crate::RenderOptions), which configures a
//! whole *frame*; these configure a single drawable.

/// How a mesh is **rasterized**: solid filled triangles, an edge wireframe
/// (`LineList` over the derived [`crate::Mesh::edge_indices`] buffer), textured,
/// or physically-based [`Shaded`](Self::Shaded). Default is
/// [`Filled`](Self::Filled).
///
/// Every variant is a way of drawing *the mesh's own geometry*. Choosing to draw
/// something else entirely — a grounding shadow — is a
/// [`DrawSelection`](super::DrawSelection), not a mode (#203).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderMode {
    /// Draw triangles filled with the per-vertex color (the mesh's triangle
    /// index buffer).
    #[default]
    Filled,
    /// Draw only triangle edges as lines (the deduped edge index buffer).
    Wireframe,
    /// Draw triangles filled, sampling the renderer's bound texture at each
    /// vertex UV instead of the vertex color (#20).
    Textured,
    /// **Shaded**: physically-based Disney principled BRDF (`pbr.wgsl`) — the
    /// bound albedo lit by a small virtual light rig plus an optional
    /// equirectangular HDR environment-map reflection, with smooth shading
    /// normals derived at upload. Metallic materials read as shiny reflective
    /// metal (e.g. the coke can). Configured globally via the renderer's
    /// [`DisneyMaterial`](crate::DisneyMaterial) + bound environment map.
    ///
    /// Named for the *appearance* it selects, like its siblings, rather than for
    /// the technique that achieves it — which leaves room for a second
    /// physically-based model without a `Pbr2`-style name (#203). The PBR
    /// machinery keeps its own name: `PbrConfig`, `pbr.wgsl`, `DisneyMaterial`.
    /// The wire byte (`4`) and the external `"pbr"` config string are unchanged.
    Shaded,
}

/// How a scene's [`Background::frame`](super::Background::frame) plane maps its
/// background image onto the viewport (#63). Both modes fill the whole viewport (no letterbox bars); they
/// differ only in how a mismatched image/viewport aspect is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameFit {
    /// Stretch the image to exactly fill the viewport, ignoring aspect ratio.
    /// The natural choice when the frame image already matches the render aspect
    /// (e.g. a 16:9 video rendered at 16:9).
    #[default]
    Stretch,
    /// Scale the image to cover the viewport preserving aspect, center-cropping
    /// the overflowing axis (no bars, some content cropped).
    Cover,
}

/// The centered UV scale that realizes `fit` for an image of `tex_w`×`tex_h`
/// displayed on a `view_w`×`view_h` viewport. Applied in `frame_plane.wgsl` as
/// `uv' = (uv − 0.5)·scale + 0.5`, so `1.0` fills and `< 1.0` crops (zooms in).
/// [`FrameFit::Stretch`] is always `(1, 1)`; [`FrameFit::Cover`] shrinks the UV
/// range on the longer axis so the shorter one fills.
pub(crate) fn frame_fit_uv_scale(
    fit: FrameFit,
    tex_w: u32,
    tex_h: u32,
    view_w: u32,
    view_h: u32,
) -> [f32; 2] {
    match fit {
        FrameFit::Stretch => [1.0, 1.0],
        FrameFit::Cover => {
            let tex_aspect = tex_w.max(1) as f32 / tex_h.max(1) as f32;
            let view_aspect = view_w.max(1) as f32 / view_h.max(1) as f32;
            if tex_aspect > view_aspect {
                // Image wider than the viewport: crop its width (sample a
                // narrower horizontal UV range).
                [view_aspect / tex_aspect, 1.0]
            } else {
                // Image taller than the viewport: crop its height.
                [1.0, tex_aspect / view_aspect]
            }
        }
    }
}

/// Which coordinate plane a [`DrawableObject::PlaneGrid`] lattices, i.e. the two
/// model-space axes it spans (the third is held at 0): `Xy` → the X/Y plane,
/// `Xz` → X/Z, `Yz` → Y/Z. For a #77 placement quad (whose local Z is the plane
/// normal), `Xy` is the quad's own plane — a grid on the reconstructed surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridPlane {
    /// The model-space X/Y plane (Z = 0). The placement-quad's own surface.
    Xy,
    /// The model-space X/Z plane (Y = 0).
    Xz,
    /// The model-space Y/Z plane (X = 0).
    Yz,
}

impl GridPlane {
    /// A stable `0..3` index (`Xy`→0, `Xz`→1, `Yz`→2) used to key the renderer's
    /// per-plane grid vertex buffers.
    pub(crate) fn index(self) -> usize {
        match self {
            GridPlane::Xy => 0,
            GridPlane::Xz => 1,
            GridPlane::Yz => 2,
        }
    }
}

impl std::str::FromStr for GridPlane {
    type Err = String;

    /// Parses `xy` / `xz` / `yz` (case-insensitive) into a [`GridPlane`], so
    /// front-ends can accept the plane as a plain flag value.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "xy" => Ok(GridPlane::Xy),
            "xz" => Ok(GridPlane::Xz),
            "yz" => Ok(GridPlane::Yz),
            other => Err(format!(
                "unknown grid plane {other:?} (expected xy, xz, or yz)"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_plane_from_str_roundtrip() {
        use std::str::FromStr;
        assert_eq!(GridPlane::from_str("xy"), Ok(GridPlane::Xy));
        assert_eq!(GridPlane::from_str("XZ"), Ok(GridPlane::Xz));
        assert_eq!(GridPlane::from_str("yz"), Ok(GridPlane::Yz));
        assert!(GridPlane::from_str("zz").is_err());
    }

    #[test]
    fn frame_fit_uv_scale_stretch_and_cover() {
        // Stretch always fills exactly (no crop), regardless of aspect mismatch.
        assert_eq!(
            frame_fit_uv_scale(FrameFit::Stretch, 200, 100, 100, 100),
            [1.0, 1.0]
        );

        // Cover a 2:1 image on a 1:1 viewport: crop width (sample a narrower
        // horizontal UV range), full height.
        let s = frame_fit_uv_scale(FrameFit::Cover, 200, 100, 100, 100);
        assert!(
            (s[0] - 0.5).abs() < 1e-6 && (s[1] - 1.0).abs() < 1e-6,
            "wide image over square viewport crops width, got {s:?}"
        );

        // Cover a 1:2 image on a 1:1 viewport: crop height, full width.
        let s = frame_fit_uv_scale(FrameFit::Cover, 100, 200, 100, 100);
        assert!(
            (s[0] - 1.0).abs() < 1e-6 && (s[1] - 0.5).abs() < 1e-6,
            "tall image over square viewport crops height, got {s:?}"
        );

        // Matching aspect ⇒ no crop either way.
        assert_eq!(
            frame_fit_uv_scale(FrameFit::Cover, 160, 90, 320, 180),
            [1.0, 1.0]
        );
    }
}
