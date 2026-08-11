// Screen-space gizmo lines. Each model-space segment is repeated across six
// vertices; this stage projects both endpoints and expands them into a pixel-width
// quad. The fragment stage evaluates an analytic rectangle distance for AA even
// when the render pass is single-sampled.

const AA_RADIUS_PX: f32 = 1.0;

struct Params {
    view_proj: mat4x4<f32>,
    // xy = viewport pixels, zw = 2 / viewport pixels.
    viewport: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;

struct VsIn {
    @location(0) start: vec3<f32>,
    @location(1) end: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) model_col0: vec4<f32>,
    @location(4) model_col1: vec4<f32>,
    @location(5) model_col2: vec4<f32>,
    @location(6) model_col3: vec4<f32>,
    // x = endpoint (0/1), y = side (-1/+1), z = full width in pixels.
    @location(7) extrusion: vec3<f32>,
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec3<f32>,
    // Pixel coordinates relative to the unexpanded segment: x along, y across.
    @location(1) @interpolate(linear) line_position: vec2<f32>,
    // x = segment length in pixels, y = requested half-width.
    @location(2) @interpolate(flat) rectangle: vec2<f32>,
};

fn safe_reciprocal_w(w: f32) -> f32 {
    let sign = select(1.0, -1.0, w < 0.0);
    return sign / max(abs(w), 1e-5);
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.model_col0, in.model_col1, in.model_col2, in.model_col3);
    let start_clip = params.view_proj * model * vec4<f32>(in.start, 1.0);
    let end_clip = params.view_proj * model * vec4<f32>(in.end, 1.0);
    let start_ndc = start_clip.xy * safe_reciprocal_w(start_clip.w);
    let end_ndc = end_clip.xy * safe_reciprocal_w(end_clip.w);
    let pixel_delta = (end_ndc - start_ndc) * params.viewport.xy * 0.5;
    let segment_length = length(pixel_delta);
    let tangent = select(
        vec2<f32>(1.0, 0.0),
        pixel_delta / max(segment_length, 1e-5),
        segment_length > 1e-5,
    );
    let normal = vec2<f32>(-tangent.y, tangent.x);
    let at_end = in.extrusion.x > 0.5;
    let endpoint_clip = select(start_clip, end_clip, at_end);
    let cap_offset = select(-AA_RADIUS_PX, AA_RADIUS_PX, at_end);
    let half_width = in.extrusion.z * 0.5;
    let outer_half_width = half_width + AA_RADIUS_PX;
    let pixel_offset =
        tangent * cap_offset + normal * in.extrusion.y * outer_half_width;

    var out: VsOut;
    out.position = vec4<f32>(
        endpoint_clip.xy + pixel_offset * params.viewport.zw * endpoint_clip.w,
        endpoint_clip.z,
        endpoint_clip.w,
    );
    out.color = in.color;
    out.line_position = vec2<f32>(
        select(-AA_RADIUS_PX, segment_length + AA_RADIUS_PX, at_end),
        in.extrusion.y * outer_half_width,
    );
    out.rectangle = vec2<f32>(segment_length, half_width);
    return out;
}

fn rectangle_distance(position: vec2<f32>, rectangle: vec2<f32>) -> f32 {
    let half_size = vec2<f32>(rectangle.x * 0.5, rectangle.y);
    let centered = vec2<f32>(position.x - half_size.x, position.y);
    let delta = abs(centered) - half_size;
    return length(max(delta, vec2<f32>(0.0))) + min(max(delta.x, delta.y), 0.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let distance = rectangle_distance(in.line_position, in.rectangle);
    let alpha = 1.0 - smoothstep(-AA_RADIUS_PX, AA_RADIUS_PX, distance);
    return vec4<f32>(in.color, alpha);
}
