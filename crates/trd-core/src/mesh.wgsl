// Indexed mesh path. Positions/colors come from vertex buffers; each position is
// transformed by the per-frame clip transform `P·V·M`.

struct Params {
    transform: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.position = params.transform * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
