// The minimal reference shader (see `render/triangle_renderer.rs`): a single
// gradient triangle. The vertex stage passes NDC positions straight through and
// forwards the per-vertex color; the fragment stage multiplies it by a uniform
// tint bound at group 0.

struct Tint {
    color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> tint: Tint;

struct VsIn {
    @location(0) position: vec2<f32>,
    @location(1) color: vec3<f32>,
};

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0) * tint.color;
}
