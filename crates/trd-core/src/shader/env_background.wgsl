const PI: f32 = 3.14159265358979323846;

struct BackgroundUniform {
    inverse_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    // rotation, exposure, blur, tonemap mode
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: BackgroundUniform;
@group(1) @binding(0) var env_tex: texture_2d<f32>;
@group(1) @binding(1) var env_samp: sampler;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) direction: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let clip = positions[index];
    let world = u.inverse_view_proj * vec4<f32>(clip, 1.0, 1.0);
    var out: VsOut;
    out.clip_position = vec4<f32>(clip, 1.0, 1.0);
    out.direction = world.xyz / world.w - u.camera_pos.xyz;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = normalize(in.direction);
    let theta = acos(d.y);
    var phi = atan2(d.x, d.z) + PI;
    let two_pi = 2.0 * PI;
    let rotate = PI - u.params.x;
    phi = (phi + rotate) - floor((phi + rotate) / two_pi) * two_pi;
    let uv = vec2<f32>(
        clamp(phi / two_pi, 0.0, 1.0),
        clamp(theta / PI, 0.0, 1.0),
    );
    let max_level = f32(textureNumLevels(env_tex) - 1u);
    let linear = textureSampleLevel(env_tex, env_samp, uv, u.params.z * max_level).rgb * u.params.y;
    var mapped: vec3<f32>;
    if (u.params.w > 0.5) {
        mapped = (linear * (2.51 * linear + 0.03)) /
            (linear * (2.43 * linear + 0.59) + 0.14);
    } else {
        mapped = linear / (vec3<f32>(1.0) + linear);
    }
    return vec4<f32>(mapped, 1.0);
}
