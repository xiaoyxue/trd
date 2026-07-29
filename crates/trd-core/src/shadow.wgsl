// Contact / blob **grounding shadow** (#110 follow-up). A soft dark radial blob
// laid on a placed mesh's reconstructed ground plane so the mesh reads as
// *sitting on* the surface rather than floating over the composited video plate.
//
// The vertex path matches `mesh.wgsl` (per-instance model · the camera `P·V`
// uniform); the model places a unit XY quad on the plane under the mesh. The
// fragment feathers a soft alpha from the quad's local XY radius — darkest at the
// centre, fading to 0 at the rim — and the pipeline alpha-blends it over the
// background frame plane (drawn before the opaque content mesh, depth-write off,
// so the mesh composites on top and the surrounding rim darkens the court).

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
    // The vertex's local model-space XY (∈ [-1, 1]²), the radial coordinate the
    // fragment feathers the shadow from.
    @location(0) local: vec2<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.model_col0, in.model_col1, in.model_col2, in.model_col3);
    var out: VsOut;
    out.position = params.view_proj * model * vec4<f32>(in.position, 1.0);
    out.local = in.position.xy;
    return out;
}

// Peak darkening across the blob core, feathering to 0 by the quad rim (radius
// 1) so the shadow has no hard edge. Kept subtle ("a little") — the mesh grounds
// without a heavy black disc. `smoothstep(1.0, 0.4, d)` holds full strength for
// the inner 40% (under the mesh footprint) then feathers out, so a soft
// darkening still reads on the court *around* the contact instead of being
// entirely occluded by the mesh.
const STRENGTH: f32 = 0.6;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = length(in.local);
    let a = smoothstep(1.0, 0.4, d) * STRENGTH;
    return vec4<f32>(0.0, 0.0, 0.0, a);
}
