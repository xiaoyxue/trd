// Disney principled BRDF mesh path (physically-based shading).
//
// A WGSL port of the reference `ref/DisneyPBR/shader.frag` Disney BRDF (Burley
// 2012), wired into trd's instanced mesh pipeline. Unlike `textured.wgsl` (which
// just samples the albedo flat), this path lights the mesh with a small virtual
// light rig plus an optional equirectangular HDR **environment map** reflection,
// so metallic materials (e.g. the coke can) read as shiny reflective metal.
//
// Bind groups:
//   group 0 = PbrUniform (camera P·V, camera world pos, Disney material params,
//             the light rig, and env/exposure controls) — vertex + fragment.
//   group 1 = albedo texture + sampler (the mesh's base color, sampled at UV).
//   group 2 = environment map (equirect HDR) + sampler.
//
// The albedo is stored `Rgba8UnormSrgb`, so `textureSample` returns **linear**
// base color; the Disney math therefore treats baseColor as already-linear (no
// `mon2lin` gamma step, unlike the GL reference whose sampler returned raw sRGB).
// The color target is `Rgba8UnormSrgb`, so the shader outputs linear radiance and
// the target encodes sRGB on write.

const PI: f32 = 3.14159265358979323846;
const MAX_LIGHTS: u32 = 4u;

struct PbrUniform {
    view_proj: mat4x4<f32>,
    // xyz = camera world position, w unused.
    camera_pos: vec4<f32>,
    // metallic, subsurface, specular, roughness
    mat0: vec4<f32>,
    // specularTint, anisotropic, sheen, sheenTint
    mat1: vec4<f32>,
    // clearcoat, clearcoatGloss, env_intensity, exposure
    mat2: vec4<f32>,
    // baseColorTint.rgb, ambient
    mat3: vec4<f32>,
    // num_dir_lights, num_point_lights, use_env, light_scale
    counts: vec4<f32>,
    // tonemap mode (0 = reinhard, 1 = aces), reserved, reserved, reserved
    mat4: vec4<f32>,
    // xyz = direction the light travels, w = intensity
    dir_lights: array<vec4<f32>, MAX_LIGHTS>,
    // xyz = world position, w = intensity
    point_lights: array<vec4<f32>, MAX_LIGHTS>,
};

@group(0) @binding(0) var<uniform> u: PbrUniform;
@group(1) @binding(0) var albedo_tex: texture_2d<f32>;
@group(1) @binding(1) var albedo_samp: sampler;
@group(2) @binding(0) var env_tex: texture_2d<f32>;
@group(2) @binding(1) var env_samp: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Per-instance model matrix, one column per attribute (column-major).
    @location(3) model_col0: vec4<f32>,
    @location(4) model_col1: vec4<f32>,
    @location(5) model_col2: vec4<f32>,
    @location(6) model_col3: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.model_col0, in.model_col1, in.model_col2, in.model_col3);
    let world = model * vec4<f32>(in.position, 1.0);
    // Normal matrix: the model's upper-left 3×3. Correct for the rotation +
    // uniform-scale + translation transforms trd builds (preview scale-to-fit is
    // uniform), so a plain 3×3 multiply preserves direction (renormalized in fs).
    let m3 = mat3x3<f32>(in.model_col0.xyz, in.model_col1.xyz, in.model_col2.xyz);

    var out: VsOut;
    out.clip_position = u.view_proj * world;
    out.world_position = world.xyz;
    out.world_normal = m3 * in.normal;
    out.uv = in.uv;
    return out;
}

// --- Disney BRDF (ported from ref/DisneyPBR/shader.frag) ----------------------

fn sqr(x: f32) -> f32 { return x * x; }

fn schlick_fresnel(u_in: f32) -> f32 {
    let m = clamp(1.0 - u_in, 0.0, 1.0);
    let m2 = m * m;
    return m2 * m2 * m; // pow(m, 5)
}

fn gtr1(n_dot_h: f32, a: f32) -> f32 {
    if (a >= 1.0) { return 1.0 / PI; }
    let a2 = a * a;
    let t = 1.0 + (a2 - 1.0) * n_dot_h * n_dot_h;
    return (a2 - 1.0) / (PI * log(a2) * t);
}

fn gtr2_aniso(n_dot_h: f32, h_dot_x: f32, h_dot_y: f32, ax: f32, ay: f32) -> f32 {
    return 1.0 / (PI * ax * ay * sqr(sqr(h_dot_x / ax) + sqr(h_dot_y / ay) + n_dot_h * n_dot_h));
}

fn smith_g_ggx(n_dot_v: f32, alpha_g: f32) -> f32 {
    let a = alpha_g * alpha_g;
    let b = n_dot_v * n_dot_v;
    return 1.0 / (n_dot_v + sqrt(a + b - a * b));
}

fn smith_g_ggx_aniso(n_dot_v: f32, v_dot_x: f32, v_dot_y: f32, ax: f32, ay: f32) -> f32 {
    return 1.0 / (n_dot_v + sqrt(sqr(v_dot_x * ax) + sqr(v_dot_y * ay) + sqr(n_dot_v)));
}

// Evaluate the Disney BRDF for light direction L, view direction V, normal N,
// tangent X, bitangent Y and (already-linear) baseColor.
fn disney_brdf(l: vec3<f32>, v: vec3<f32>, n: vec3<f32>, x: vec3<f32>, y: vec3<f32>, base_color: vec3<f32>) -> vec3<f32> {
    let metallic = u.mat0.x;
    let subsurface = u.mat0.y;
    let specular = u.mat0.z;
    let roughness = u.mat0.w;
    let specular_tint = u.mat1.x;
    let anisotropic = u.mat1.y;
    let sheen = u.mat1.z;
    let sheen_tint = u.mat1.w;
    let clearcoat = u.mat2.x;
    let clearcoat_gloss = u.mat2.y;

    let n_dot_l = clamp(dot(n, l), 0.0001, 0.9999);
    let n_dot_v = clamp(dot(n, v), 0.0001, 0.9999);

    let h = normalize(l + v);
    let n_dot_h = clamp(dot(n, h), 0.0001, 0.9999);
    let l_dot_h = clamp(dot(l, h), 0.0001, 0.9999);

    let cdlin = base_color; // already linear (albedo sampled from an sRGB texture)
    let cdlum = 0.3 * cdlin.x + 0.6 * cdlin.y + 0.1 * cdlin.z; // luminance approx.

    let ctint = select(vec3<f32>(1.0), cdlin / cdlum, cdlum > 0.0); // hue+sat
    let cspec0 = mix(specular * 0.08 * mix(vec3<f32>(1.0), ctint, specular_tint), cdlin, metallic);
    let csheen = mix(vec3<f32>(1.0), ctint, sheen_tint);

    // Diffuse fresnel (1 at normal incidence -> .5 at grazing) + retro-reflection.
    let fl = schlick_fresnel(n_dot_l);
    let fv = schlick_fresnel(n_dot_v);
    let fd90 = 0.5 + 2.0 * l_dot_h * l_dot_h * roughness;
    let fd = mix(1.0, fd90, fl) * mix(1.0, fd90, fv);

    // Hanrahan-Krueger subsurface approximation.
    let fss90 = l_dot_h * l_dot_h * roughness;
    let fss = mix(1.0, fss90, fl) * mix(1.0, fss90, fv);
    let ss = 1.25 * (fss * (1.0 / (n_dot_l + n_dot_v) - 0.5) + 0.5);

    // Specular (anisotropic GGX).
    let aspect = sqrt(1.0 - anisotropic * 0.9);
    let ax = max(0.001, sqr(roughness) / aspect);
    let ay = max(0.001, sqr(roughness) * aspect);
    let ds = gtr2_aniso(n_dot_h, dot(h, x), dot(h, y), ax, ay);
    let fh = schlick_fresnel(l_dot_h);
    let fs = mix(cspec0, vec3<f32>(1.0), fh);
    var gs = smith_g_ggx_aniso(n_dot_l, dot(l, x), dot(l, y), ax, ay);
    gs = gs * smith_g_ggx_aniso(n_dot_v, dot(v, x), dot(v, y), ax, ay);

    // Sheen.
    let fsheen = fh * sheen * csheen;

    // Clearcoat (ior 1.5 -> F0 = 0.04).
    let dr = gtr1(n_dot_h, mix(0.1, 0.001, clearcoat_gloss));
    let fr = mix(0.04, 1.0, fh);
    let gr = smith_g_ggx(n_dot_l, 0.25) * smith_g_ggx(n_dot_v, 0.25);

    return ((1.0 / PI) * mix(fd, ss, subsurface) * cdlin + fsheen) * (1.0 - metallic)
        + gs * fs * ds
        + 0.25 * clearcoat * gr * fr * dr;
}

// Equirectangular environment-map lookup (Y is the polar axis), matching the
// reference SampleEnvironmentMap: theta = acos(D.y), phi = atan2(D.x, D.z).
fn sample_environment(dir: vec3<f32>) -> vec3<f32> {
    let d = normalize(dir);
    let theta = acos(clamp(d.y, -1.0, 1.0));
    var phi = atan2(d.x, d.z) + PI;
    let two_pi = 2.0 * PI;
    // Wrap into [0, 2PI) after the reference's PI rotation.
    phi = (phi + PI) - floor((phi + PI) / two_pi) * two_pi;
    let uv = vec2<f32>(clamp(phi / two_pi, 0.0, 1.0), clamp(theta / PI, 0.0, 1.0));
    return textureSampleLevel(env_tex, env_samp, uv, 0.0).rgb;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let metallic = u.mat0.x;
    let roughness = u.mat0.w;
    let env_intensity = u.mat2.z;
    let exposure = u.mat2.w;
    let ambient = u.mat3.w;
    let light_scale = u.counts.w;
    let use_env = u.counts.z > 0.5;
    let tonemap_mode = u.mat4.x;

    let base_color = textureSample(albedo_tex, albedo_samp, in.uv).rgb * u.mat3.rgb;

    let n = normalize(in.world_normal);
    let v = normalize(u.camera_pos.xyz - in.world_position);
    // Arbitrary tangent frame for the (isotropic-by-default) anisotropy terms.
    let x = normalize(cross(n, vec3<f32>(0.0, 0.9999, 0.0001)));
    let y = normalize(cross(n, x));

    // Ambient fill so shadowed regions are not pure black.
    var color = ambient * base_color;

    let num_dir = u32(u.counts.x);
    for (var i: u32 = 0u; i < num_dir && i < MAX_LIGHTS; i = i + 1u) {
        let l = normalize(-u.dir_lights[i].xyz);
        color = color + light_scale * u.dir_lights[i].w * disney_brdf(l, v, n, x, y, base_color);
    }

    let num_point = u32(u.counts.y);
    for (var i: u32 = 0u; i < num_point && i < MAX_LIGHTS; i = i + 1u) {
        let to_light = u.point_lights[i].xyz - in.world_position;
        let l = normalize(to_light);
        let dist = length(to_light);
        let falloff = 1.0 / (0.01 + dist * dist);
        color = color + light_scale * u.point_lights[i].w * falloff * disney_brdf(l, v, n, x, y, base_color);
    }

    // Environment reflection: a metallic surface reflects the surrounding HDR
    // probe (tinted by its base color), a dielectric only a faint fresnel rim.
    if (use_env) {
        let r = reflect(-v, n);
        let env = sample_environment(r) * env_intensity;
        let n_dot_v = clamp(dot(n, v), 0.0, 1.0);
        let f0 = mix(vec3<f32>(0.04), base_color, metallic);
        // Roughness-aware Fresnel (Sébastien Lagarde) so rough metal reflects less.
        let fres = f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(1.0 - n_dot_v, 5.0);
        color = color + env * fres;
    }

    // Exposure + tone map. `tonemap_mode` selects the curve: 0 = per-channel
    // Reinhard `x/(1+x)` (the default, byte-identical to trd's historical
    // pipeline), 1 = ACES filmic (Narkowicz RRT+ODT fit,
    // ref/ToneMapping/tonemap.frag) — a softer highlight roll-off that retains
    // hue/saturation on bright albedo. `exposure` scales the linear radiance
    // first (the ACES adapted_lum); the result stays linear for the sRGB target.
    let exposed = color * exposure;
    var mapped: vec3<f32>;
    if (tonemap_mode > 0.5) {
        let aa: f32 = 2.51;
        let bb: f32 = 0.03;
        let cc: f32 = 2.43;
        let dd: f32 = 0.59;
        let ee: f32 = 0.14;
        mapped = (exposed * (aa * exposed + bb)) / (exposed * (cc * exposed + dd) + ee);
    } else {
        mapped = exposed / (vec3<f32>(1.0) + exposed);
    }
    return vec4<f32>(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
