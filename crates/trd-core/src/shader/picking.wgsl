// Object-id ("color index") picking path. Each drawn instance carries a flat
// `id_color` (the object id encoded as RGB, 0 = background); the vertex stage
// reuses the mesh transform `clip = P·V·M·p` and the fragment stage writes the
// id_color unshaded. Rendered single-sampled into a linear (non-sRGB) target so
// the byte values round-trip exactly on read-back, then the pixel under the
// cursor is decoded back to an id.

struct Params {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;

struct VsIn {
    @location(0) position: vec3<f32>,
    // Per-instance model matrix, one column per attribute (column-major).
    @location(3) model_col0: vec4<f32>,
    @location(4) model_col1: vec4<f32>,
    @location(5) model_col2: vec4<f32>,
    @location(6) model_col3: vec4<f32>,
    // Per-instance flat id color (object id encoded as RGB, 0 = background).
    @location(7) id_color: vec4<f32>,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) id_color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.model_col0, in.model_col1, in.model_col2, in.model_col3);
    var out: VsOut;
    out.position = params.view_proj * model * vec4<f32>(in.position, 1.0);
    out.id_color = in.id_color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.id_color;
}
