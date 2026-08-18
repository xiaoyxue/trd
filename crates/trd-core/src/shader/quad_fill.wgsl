// Translucent **placement-quad fill** — the hover/selection highlight for the
// tracked quad (#293 follow-up).
//
// The vertex path matches `shadow.wgsl` (per-instance model · the camera `P·V`
// uniform) and it reuses the very same unit XY quad geometry; only the fragment
// differs. Where the shadow feathers a dark radial blob, this lays a flat,
// translucent green wash across the whole quad, so pointing at a quad tints the
// area an object would be placed on rather than merely outlining it.
//
// Alpha-blended over the background frame plane with depth-write off, drawn
// under the quad outline so the coloured edge still reads on top of the wash.

struct Params {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Per-instance model matrix, one column per attribute (column-major).
    @location(3) model_col0: vec4<f32>,
    @location(4) model_col1: vec4<f32>,
    @location(5) model_col2: vec4<f32>,
    @location(6) model_col3: vec4<f32>,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.model_col0, in.model_col1, in.model_col2, in.model_col3);
    var out: VsOut;
    out.position = params.view_proj * model * vec4<f32>(in.position, 1.0);
    return out;
}

// The same green as the unselected quad outline, at an alpha that reads clearly
// over bright footage while still showing what is underneath — the quad is an
// authoring aid laid over live video, and players walking through it have to
// stay visible. Against a lit hardwood court anything much weaker disappears.
const FILL: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
const ALPHA: f32 = 0.35;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(FILL, ALPHA);
}
