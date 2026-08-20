// Background frame plane (#63). A screen-aligned fullscreen triangle that samples
// the bound background frame texture, drawn *first* in the mesh pass (depth write
// off, compare Always) so the mesh scene + gizmos composite on top. It ignores
// the camera entirely — the plane is authored in clip space directly.
//
// Group 0 is the background frame **ring** (binding 0) + sampler (binding 1) + a
// small fit uniform (binding 2). The ring is a `texture_2d_array`: the decoder
// fills layers ahead of the renderer, which presents one by index, so showing a
// different frame costs a uniform write rather than a re-upload. `uv_scale` maps
// the fullscreen UVs to the sampled sub-rectangle so the image can `Stretch`
// (scale = (1,1)) or `Cover` the viewport (scale < 1 on the cropped axis),
// centered. The texture is `Rgba8UnormSrgb`, so texels linearize on sample and
// the sRGB target re-encodes on store — matching the mesh textured path and the
// CLI output.

struct Fit {
    // (scale.x, scale.y, layer, _pad) — the fullscreen UV is remapped around its
    // center by `scale`, so `< 1` zooms in (crops) and `1` fills. `layer` selects
    // which ring slot to present; it is the only thing that changes when the
    // renderer moves to a frame the ring already holds.
    uv_scale: vec4<f32>,
};

@group(0) @binding(0) var frame_tex: texture_2d_array<f32>;
@group(0) @binding(1) var frame_samp: sampler;
@group(0) @binding(2) var<uniform> fit: Fit;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // A single oversized triangle covering the whole clip-space square
    // [-1, 1]^2 (positions (-1,-1), (3,-1), (-1,3)).
    let x = f32((vid << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vid & 2u) * 2.0 - 1.0;
    var out: VsOut;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    // v = 0 at the top row: flip Y from clip space to texture space, then remap
    // around the center by the fit scale (centered crop / fill).
    let uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    out.uv = (uv - vec2<f32>(0.5)) * fit.uv_scale.xy + vec2<f32>(0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(frame_tex, frame_samp, in.uv, i32(fit.uv_scale.z));
}
