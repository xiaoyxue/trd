// Parametric hello-triangle. Base positions/colors are generated from the
// vertex index (no vertex buffer); each vertex is scaled, rotated, then
// translated by the per-frame params uniform.

struct Params {
    center: vec2<f32>,
    size: vec2<f32>,
    theta: f32,
};

@group(0) @binding(0) var<uniform> params: Params;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>(0.5, -0.5),
    );
    var colors = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );

    let base = positions[index] * params.size;
    let c = cos(params.theta);
    let s = sin(params.theta);
    let rotated = vec2<f32>(
        c * base.x - s * base.y,
        s * base.x + c * base.y,
    );
    let p = params.center + rotated;

    var out: VsOut;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.color = colors[index];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
