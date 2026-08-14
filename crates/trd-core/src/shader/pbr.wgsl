// glTF PBR mesh path built on the Disney principled BRDF.
//
// A WGSL port of the reference `ref/DisneyPBR/shader.frag` Disney BRDF (Burley
// 2012), wired into trd's instanced mesh pipeline. Unlike `textured.wgsl` (which
// just samples the albedo flat), this path lights the mesh with a small virtual
// light rig plus an optional equirectangular HDR **environment map** reflection,
// so metallic materials (e.g. the coke can) read as shiny reflective metal.
//
// Bind groups:
//   group 0 = binding 0: PbrSceneUniform — camera P·V, camera world pos and the
//             light rig, written ONCE per frame; binding 1: PbrUniform — this
//             mesh's Disney material + env/exposure controls, selected by a
//             dynamic offset. Split by frequency of change (#182); both stay in
//             group 0 because the portable WebGPU baseline allows only four
//             groups and this shader already uses all four.
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

// What the whole frame shares: written once per frame, never per mesh.
struct PbrSceneUniform {
    view_proj: mat4x4<f32>,
    // xyz = camera world position, w = use_env (1 = a probe is bound).
    camera_pos: vec4<f32>,
    // num_dir_lights, num_point_lights, ambient, light_scale
    light_params: vec4<f32>,
    // env gain, env yaw (radians), reserved, reserved — the yaw is the single
    // source of truth for both the reflections and the sky behind them (#182).
    env_params: vec4<f32>,
    // xyz = direction the light travels, w = intensity
    dir_lights: array<vec4<f32>, MAX_LIGHTS>,
    // xyz = world position, w = intensity
    point_lights: array<vec4<f32>, MAX_LIGHTS>,
};

// What one mesh contributes: selected by a dynamic offset per draw.
struct PbrUniform {
    // metallic, subsurface, specular, roughness
    mat0: vec4<f32>,
    // specularTint, anisotropic, sheen, sheenTint
    mat1: vec4<f32>,
    // clearcoat, clearcoatGloss, env_intensity, exposure
    mat2: vec4<f32>,
    // baseColorTint.rgb, debug view
    mat3: vec4<f32>,
    // tonemap mode (0 = reinhard, 1 = aces), reserved, has normal map, has mr map
    mat4: vec4<f32>,
};

@group(0) @binding(0) var<uniform> scene: PbrSceneUniform;
@group(0) @binding(1) var<uniform> u: PbrUniform;
@group(1) @binding(0) var albedo_tex: texture_2d<f32>;
@group(1) @binding(1) var albedo_samp: sampler;
@group(2) @binding(0) var env_tex: texture_2d<f32>;
@group(2) @binding(1) var env_samp: sampler;
@group(2) @binding(2) var brdf_lut: texture_2d<f32>;
@group(2) @binding(3) var brdf_samp: sampler;
@group(2) @binding(4) var irradiance_tex: texture_2d<f32>;
@group(3) @binding(0) var metallic_roughness_tex: texture_2d<f32>;
@group(3) @binding(1) var normal_tex: texture_2d<f32>;
@group(3) @binding(2) var material_samp: sampler;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Per-instance model matrix, one column per attribute (column-major).
    @location(3) model_col0: vec4<f32>,
    @location(4) model_col1: vec4<f32>,
    @location(5) model_col2: vec4<f32>,
    @location(6) model_col3: vec4<f32>,
    @location(7) tangent: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) world_bitangent: vec3<f32>,
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
    out.clip_position = scene.view_proj * world;
    out.world_position = world.xyz;
    out.world_normal = m3 * in.normal;
    out.uv = in.uv;
    out.world_tangent = m3 * in.tangent.xyz;
    out.world_bitangent = cross(out.world_normal, out.world_tangent) * in.tangent.w;
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
fn disney_brdf(l: vec3<f32>, v: vec3<f32>, n: vec3<f32>, x: vec3<f32>, y: vec3<f32>, base_color: vec3<f32>, metallic: f32, roughness: f32) -> vec3<f32> {
    let subsurface = u.mat0.y;
    let specular = u.mat0.z;
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
fn environment_uv(direction: vec3<f32>) -> vec2<f32> {
    let d = normalize(direction);
    let theta = acos(d.y);
    var phi = atan2(d.x, d.z) + PI;
    let two_pi = 2.0 * PI;
    let rotate = PI - scene.env_params.y;
    phi = (phi + rotate) - floor((phi + rotate) / two_pi) * two_pi;
    return vec2<f32>(clamp(phi / two_pi, 0.0, 1.0), clamp(theta / PI, 0.0, 1.0));
}

fn prefilter_environment(reflection: vec3<f32>, roughness: f32) -> vec3<f32> {
    let max_level = f32(textureNumLevels(env_tex) - 1u);
    return textureSampleLevel(
        env_tex,
        env_samp,
        environment_uv(reflection),
        roughness * max_level,
    ).rgb;
}

fn diffuse_environment(normal: vec3<f32>) -> vec3<f32> {
    return textureSampleLevel(irradiance_tex, env_samp, environment_uv(normal), 0.0).rgb;
}

// Data-map previews should preserve their stored numeric value on an sRGB
// target. Decode that display value here; the target re-encodes it on write.
fn data_preview(value: vec3<f32>) -> vec3<f32> {
    let low = value / 12.92;
    let high = pow((value + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(high, low, value <= vec3<f32>(0.04045));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let debug_view = u.mat3.w;
    var factors = vec4<f32>(1.0);
    if (u.mat4.w > 0.5) {
        factors = textureSample(metallic_roughness_tex, material_samp, in.uv);
    }
    let metallic = u.mat0.x * factors.b;
    let roughness = max(0.02, u.mat0.w * factors.g);
    // Per-object gain × the scene-wide probe gain (#182).
    let env_intensity = u.mat2.z * scene.env_params.x;
    let exposure = u.mat2.w;
    let ambient = scene.light_params.z;
    let light_scale = scene.light_params.w;
    let use_env = scene.camera_pos.w > 0.5;
    let tonemap_mode = u.mat4.x;

    let base_color = textureSample(albedo_tex, albedo_samp, in.uv).rgb * u.mat3.rgb;

    let geometry_normal = normalize(in.world_normal);
    let tangent = normalize(in.world_tangent - geometry_normal * dot(geometry_normal, in.world_tangent));
    let bitangent = normalize(in.world_bitangent);
    var n = geometry_normal;
    var normal_sample = vec3<f32>(0.5, 0.5, 1.0);
    if (u.mat4.z > 0.5) {
        normal_sample = textureSample(normal_tex, material_samp, in.uv).xyz;
        let mapped_normal = normal_sample * 2.0 - 1.0;
        n = normalize(mat3x3<f32>(tangent, bitangent, geometry_normal) * mapped_normal);
    }
    if (debug_view > 1.5 && debug_view < 2.5) {
        return vec4<f32>(data_preview(vec3<f32>(roughness)), 1.0);
    }
    if (debug_view > 2.5 && debug_view < 3.5) {
        return vec4<f32>(data_preview(vec3<f32>(metallic)), 1.0);
    }
    if (debug_view > 3.5) {
        return vec4<f32>(data_preview(normal_sample), 1.0);
    }
    let v = normalize(scene.camera_pos.xyz - in.world_position);
    // Arbitrary tangent frame for the (isotropic-by-default) anisotropy terms.
    let x = normalize(cross(n, vec3<f32>(0.0, 0.9999, 0.0001)));
    let y = normalize(cross(n, x));

    // Ambient fill so shadowed regions are not pure black.
    var color = ambient * base_color;
    if (use_env) {
        let irradiance = diffuse_environment(n);
        color = color + irradiance * base_color * (1.0 - metallic) * env_intensity;
    }

    let num_dir = u32(scene.light_params.x);
    for (var i: u32 = 0u; i < num_dir && i < MAX_LIGHTS; i = i + 1u) {
        let l = normalize(-scene.dir_lights[i].xyz);
        color = color + light_scale * scene.dir_lights[i].w * disney_brdf(l, v, n, x, y, base_color, metallic, roughness);
    }

    let num_point = u32(scene.light_params.y);
    for (var i: u32 = 0u; i < num_point && i < MAX_LIGHTS; i = i + 1u) {
        let to_light = scene.point_lights[i].xyz - in.world_position;
        let l = normalize(to_light);
        let dist = length(to_light);
        let falloff = 1.0 / (0.01 + dist * dist);
        color = color + light_scale * scene.point_lights[i].w * falloff * disney_brdf(l, v, n, x, y, base_color, metallic, roughness);
    }

    // Environment reflection: a metallic surface reflects the surrounding HDR
    // probe (tinted by its base color), a dielectric only a faint fresnel rim.
    if (use_env) {
        let r = reflect(-v, n);
        let env = prefilter_environment(r, roughness) * env_intensity;
        let n_dot_v = clamp(dot(n, v), 0.0, 1.0);
        let f0 = mix(vec3<f32>(0.04), base_color, metallic);
        let brdf = textureSampleLevel(brdf_lut, brdf_samp, vec2<f32>(n_dot_v, roughness), 0.0).rg;
        color = color + env * (f0 * brdf.x + brdf.y);
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
