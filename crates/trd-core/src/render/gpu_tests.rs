//! GPU-gated render tests (`#[ignore]`, native only) plus their shared
//! readback/device helpers and OBJ fixtures.

use super::*;
use super::{
    Background, DrawSelection, DrawableObject, EnvironmentBackground, FrameFit, RenderMode,
    ResolvedDraw, Scene,
};
use crate::math::{Matrix4, Point3, Vector3};
use crate::{MeshId, MeshResourceError, MeshTable, MeshTableIndex};
use glam::{Mat4, Vec3};

/// Test convenience constructor: a [`Renderer`] over a single mesh with an
/// identity base model (vertices drawn in their own coordinates). Replaces the
/// former single-mesh `SceneRenderer::new`/`with_base_model` production helpers,
/// which only the GPU tests used.
/// An unusable mesh set is **reported**, not panicked on (#235 R8): every
/// streaming front-end learns its meshes from the wire, so "that stream carried
/// no meshes" is an input error a shell should surface — not an abort. The three
/// former `assert!`s are one typed variant now.
#[test]
#[ignore = "requires a GPU adapter"]
fn renderer_rejects_an_unusable_mesh_set_without_panicking() {
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mesh = Mesh::hello_triangle();

    let empty = Renderer::new(test_gpu(), format, &[], &[]);
    assert!(
        matches!(empty, Err(crate::RenderError::InvalidMeshSet { .. })),
        "an empty mesh set is an error, not a panic"
    );

    let mismatched = Renderer::new(test_gpu(), format, std::slice::from_ref(&mesh), &[]);
    assert!(
        matches!(mismatched, Err(crate::RenderError::InvalidMeshSet { .. })),
        "meshes and base models pair one-to-one"
    );

    let zero_samples = Renderer::with_sample_count(
        test_gpu(),
        format,
        std::slice::from_ref(&mesh),
        &[Matrix4::IDENTITY],
        0,
    );
    assert!(
        matches!(zero_samples, Err(crate::RenderError::InvalidMeshSet { .. })),
        "sample_count 0 is not a pass configuration"
    );

    // ...and the valid combination still builds.
    assert!(Renderer::with_sample_count(
        test_gpu(),
        format,
        std::slice::from_ref(&mesh),
        &[Matrix4::IDENTITY],
        1,
    )
    .is_ok());
}

/// The [`Camera`] a set of wire params resolves to for a `width`x`height`
/// target — the tests' stand-in for what a front-end does at the decode
/// boundary (#203).
fn camera_of(params: FrameParams, width: u32, height: u32) -> crate::Camera {
    params
        .to_camera(Viewport { width, height })
        .expect("test params are a valid camera form")
}

fn single(format: wgpu::TextureFormat, mesh: &Mesh) -> Renderer {
    Renderer::new(
        test_gpu(),
        format,
        std::slice::from_ref(mesh),
        &[Matrix4::IDENTITY],
    )
    .expect("one mesh with one base model is a valid mesh set")
}

fn initial_mesh_id(renderer: &Renderer, row: u32) -> MeshId {
    renderer
        .mesh_table()
        .id(MeshTableIndex::new(row))
        .unwrap_or_else(|| panic!("renderer has no initial mesh row {row}"))
}

fn render_with_readback(
    gpu: &GpuContext,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    encode: impl FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView),
) -> Vec<u8> {
    let (device, queue) = (&gpu.device, &gpu.queue);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trd render test target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let unpadded = width * 4;
    let padded_bytes_per_row =
        unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("trd render test readback"),
        size: u64::from(padded_bytes_per_row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("trd render test encoder"),
    });
    encode(&mut encoder, &view);
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU poll failed");
    rx.recv()
        .expect("map_async callback dropped")
        .expect("GPU readback failed");

    let pixels = {
        let mapped = slice.get_mapped_range().expect("buffer mapped after poll");
        crate::protocol::tightly_pack_rgba(&mapped, width, height, padded_bytes_per_row)
            .expect("GPU row unpack failed")
    };
    staging.unmap();
    pixels
}

/// One wgpu device + queue, created once and shared by **every** GPU test in
/// this binary.
///
/// Each test used to build its own `Instance` + adapter + device. Under the
/// NVIDIA proprietary Vulkan driver (e.g. via nixGL on Linux), many threads
/// creating/destroying devices at once deadlock in the driver's internal
/// (priority-inheritance) locks, so the default parallel `cargo test` run hung
/// at 0% GPU — it only completed at `--test-threads=1`/`2`. wgpu's `Device` and
/// `Queue` are cheap `Send + Sync + Clone` handles, so we create a single device
/// once (serialized by `OnceLock`) and hand out clones. Concurrent rendering on
/// one device is fully supported, so the tests run at the default parallelism
/// again without tripping the driver's device-creation deadlock.
fn test_gpu() -> std::sync::Arc<GpuContext> {
    static SHARED: std::sync::OnceLock<std::sync::Arc<GpuContext>> = std::sync::OnceLock::new();
    SHARED
        .get_or_init(|| pollster::block_on(create_test_device()))
        .clone()
}

async fn create_test_device() -> std::sync::Arc<GpuContext> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("GPU adapter required");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("trd mesh continuity test device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("request_device failed");
    std::sync::Arc::new(GpuContext {
        adapter,
        device,
        queue,
    })
}

/// A [`TextureTarget`] must report the format a renderer's pipelines are built
/// for, and its own size, after a resize.
///
/// For a texture target that format is [`TEXTURE_TARGET_FORMAT`]; the surface
/// case is the interesting one and is covered by [`SurfaceTarget`]'s sRGB-view
/// logic (a surface's preferred format is often *non*-sRGB, so it is rendered
/// through an sRGB view to stay byte-identical with the headless CLI). Pinning
/// the texture side here keeps the property accessors honest now that no trait
/// spans the two (#203).
#[test]
#[ignore = "requires a GPU adapter"]
fn texture_target_reports_its_render_format_and_size() {
    let (width, height) = (32, 24);
    let renderer = single(TEXTURE_TARGET_FORMAT, &Mesh::hello_triangle());
    let mut target = renderer
        .create_texture_target(width, height)
        .expect("texture target");

    assert_eq!(target.view_format(), TEXTURE_TARGET_FORMAT);
    assert_eq!(target.viewport(), Viewport { width, height });

    renderer
        .resize_texture_target(&mut target, 64, 48)
        .expect("resize");
    assert_eq!(
        target.viewport(),
        Viewport {
            width: 64,
            height: 48
        }
    );
    assert_eq!(target.size(), (64, 48));
    assert_eq!(target.view_format(), TEXTURE_TARGET_FORMAT);
}

/// `draw_layers` + `read_pixels` must equal `render_layers` (#203).
///
/// The split exists so drawing can stay synchronous — only the readback awaits —
/// so what has to hold is that separating them changes nothing about the pixels
/// or the submission order.
#[test]
#[ignore = "requires a GPU adapter"]
fn draw_then_read_pixels_matches_render_layers() {
    let (width, height) = (32, 24);
    let mut renderer = single(TEXTURE_TARGET_FORMAT, &Mesh::hello_triangle());
    let target = renderer
        .create_texture_target(width, height)
        .expect("target builds");

    let camera = camera_of(FrameParams::IDENTITY, width, height);
    let scene: Scene = [DrawableObject::mesh(
        initial_mesh_id(&renderer, 0),
        Matrix4::IDENTITY,
        RenderMode::Filled,
    )]
    .into_iter()
    .collect();
    let layers = [SceneLayer::new(camera, &scene)];

    let fused = pollster::block_on(renderer.render_layers(&layers, &target)).expect("fused render");

    renderer.draw_layers(&layers, &target).expect("split draw");
    let split = pollster::block_on(renderer.read_pixels(&target)).expect("split readback");

    assert_eq!(
        split, fused,
        "splitting draw from readback changed the pixels"
    );
    assert!(
        fused.chunks_exact(4).any(|px| px[..3] != [0, 0, 0]),
        "the test scene drew nothing, so the comparison is vacuous"
    );
}

/// A shell that shares trd's device must be able to **sample** the rendered
/// target, and get exactly the bytes `read_pixels` would have handed back.
///
/// This is the sRGB trap, pinned. `TEXTURE_TARGET_FORMAT` is `Rgba8UnormSrgb`,
/// which the GPU *linearizes on sample*, while the stored texels are already
/// gamma-encoded — so a view in the texture's own format would gamma-correct a
/// second time and wash the image out (this was hit for real). `create_view`
/// therefore returns an `Rgba8Unorm` view, and the blit below goes
/// `Rgba8Unorm` → `Rgba8Unorm` so a mismatch can only come from the sampled
/// interpretation, not from the copy.
///
/// It also pins the two capabilities the view depends on: `TEXTURE_BINDING`
/// usage (or building the blit bind group is a validation error) and the
/// `Rgba8Unorm` entry in `view_formats` (or `create_view` is).
#[test]
#[ignore = "requires a GPU adapter"]
fn the_target_view_samples_the_bytes_read_pixels_returns() {
    let (width, height) = (32, 24);
    let mut renderer = single(TEXTURE_TARGET_FORMAT, &Mesh::hello_triangle());
    let source = renderer
        .create_texture_target(width, height)
        .expect("source target");
    let sampled = renderer
        .create_texture_target(width, height)
        .expect("destination target");

    let scene: Scene = [DrawableObject::mesh(
        initial_mesh_id(&renderer, 0),
        Matrix4::IDENTITY,
        RenderMode::Filled,
    )]
    .into_iter()
    .collect();
    let expected =
        pollster::block_on(renderer.render_params(FrameParams::IDENTITY, &scene, &source))
            .expect("renders");

    let gpu = renderer.gpu().clone();
    let blitter = wgpu::util::TextureBlitter::new(&gpu.device, wgpu::TextureFormat::Rgba8Unorm);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("trd target view sampling test"),
        });
    blitter.copy(
        &gpu.device,
        &mut encoder,
        &source.create_view(),
        &sampled.create_view(),
    );
    gpu.queue.submit([encoder.finish()]);

    let actual = pollster::block_on(renderer.read_pixels(&sampled)).expect("sampled readback");

    assert!(
        expected.chunks_exact(4).any(|px| px[..3] != [0, 0, 0]),
        "the test scene drew nothing, so the comparison is vacuous"
    );
    assert_eq!(
        actual, expected,
        "sampling the target's view changed its texels — the gamma space is wrong"
    );
}

/// `GpuContext::adopt` must produce a context that renders exactly like one
/// `request` built — that is the whole claim a shell relies on when it hands
/// trd its UI toolkit's device instead of letting trd open a second one.
#[test]
#[ignore = "requires a GPU adapter"]
fn an_adopted_context_renders_like_a_requested_one() {
    let (width, height) = (48, 32);
    let meshes = [Mesh::hello_triangle()];

    let requested = test_gpu();
    let (mut owned, owned_target) = Renderer::with_gpu(requested.clone(), width, height, &meshes)
        .expect("the harness accepts an existing device");
    let owned_scene: Scene = [DrawableObject::mesh(
        initial_mesh_id(&owned, 0),
        Matrix4::IDENTITY,
        RenderMode::Filled,
    )]
    .into_iter()
    .collect();
    let expected =
        pollster::block_on(owned.render_params(FrameParams::IDENTITY, &owned_scene, &owned_target))
            .expect("renders");

    // The trio a front-end hands over — `eframe`'s `wgpu_render_state` exposes
    // exactly these three, already cloned out of its own render state.
    let adopted = GpuContext::adopt(
        requested.adapter.clone(),
        requested.device.clone(),
        requested.queue.clone(),
    );
    assert_eq!(adopted.adapter_facts(), requested.adapter_facts());

    let (mut shared, shared_target) = Renderer::with_gpu(adopted, width, height, &meshes)
        .expect("an adopted context builds a renderer");
    let shared_scene: Scene = [DrawableObject::mesh(
        initial_mesh_id(&shared, 0),
        Matrix4::IDENTITY,
        RenderMode::Filled,
    )]
    .into_iter()
    .collect();
    let actual = pollster::block_on(shared.render_params(
        FrameParams::IDENTITY,
        &shared_scene,
        &shared_target,
    ))
    .expect("renders");

    assert_eq!(
        actual, expected,
        "an adopted device rendered a different frame than a requested one"
    );
}

/// `render_layers` must reproduce the fixed-arity passes exactly, and composite.
///
/// The video editor builds a frame from three passes through two different camera
/// calibrations. `render_layers` generalises that to N layers (#180), and #203
/// deleted the fixed-arity `render_two_pass`/`render_three_pass` wrappers once
/// this test had shown the loop clears, orders and submits identically.
///
/// What is pinned now: one layer is *exactly* what [`Renderer::render`] does to a
/// [`RenderTarget::texture`] — which also exercises the one match dispatcher on
/// its texture arm, including its `Ok(None)` "nothing to present" answer — and
/// additional layers actually composite rather than overwrite.
#[test]
#[ignore = "requires a GPU adapter"]
fn render_layers_composites_and_matches_render_for_one_layer() {
    let (width, height) = (64, 48);
    let mut renderer = single(TEXTURE_TARGET_FORMAT, &Mesh::hello_triangle());
    let target = renderer
        .create_texture_target(width, height)
        .expect("target builds");

    let at = |x: f32| {
        DrawableObject::mesh(
            initial_mesh_id(&renderer, 0),
            Matrix4::from_translation(Vector3::new(x, 0.0, 0.0)),
            RenderMode::Filled,
        )
    };
    let background: Scene = [at(-0.4)].into_iter().collect();
    let foreground: Scene = [at(0.0)].into_iter().collect();
    let overlay: Scene = [DrawableObject::coordinate_axes(Matrix4::IDENTITY)]
        .into_iter()
        .collect();

    let camera = camera_of(FrameParams::IDENTITY, width, height);

    // One layer must be exactly what `render` does to a texture target.
    let one_layer = pollster::block_on(
        renderer.render_layers(&[SceneLayer::new(camera, &foreground)], &target),
    )
    .expect("one layer renders");

    let mut dispatched = RenderTarget::texture(target);
    assert_eq!(
        renderer
            .render(camera, &foreground, &mut dispatched)
            .expect("the texture arm never fails to present"),
        None,
        "a texture target has no surface to repair"
    );
    let single_pass = pollster::block_on(
        renderer.read_pixels(dispatched.as_texture().expect("a texture target")),
    )
    .expect("single pass reads back");
    assert_eq!(one_layer, single_pass, "one layer != render");
    assert_eq!(one_layer.len(), (width * height * 4) as usize);
    assert!(
        one_layer.chunks_exact(4).any(|px| px[..3] != [0, 0, 0]),
        "a single layer drew nothing"
    );

    let target = dispatched.as_texture().expect("a texture target");
    let three_layers = pollster::block_on(renderer.render_layers(
        &[
            SceneLayer::new(camera, &background),
            SceneLayer::new(camera, &foreground),
            SceneLayer::new(camera, &overlay),
        ],
        target,
    ))
    .expect("three layers render");
    assert_ne!(
        three_layers, one_layer,
        "layering did not composite anything extra"
    );
}

/// The placement quad's highlight wash actually reaches the framebuffer, as a
/// **translucent green** — not merely as a batched draw nobody records.
///
/// `QuadFill` is the only primitive whose whole job is a colour, so "the scene
/// contains it" says nothing: a missing pipeline arm, a wrong layer or an
/// unblended target would each leave the picture identical while every CPU-side
/// assertion still passed. This renders it over a known background and reads the
/// pixels back.
#[test]
#[ignore = "requires a GPU adapter"]
fn the_quad_fill_washes_the_target_translucent_green() {
    let (width, height) = (64, 48);
    let mut renderer = single(TEXTURE_TARGET_FORMAT, &Mesh::hello_triangle());
    let target = renderer
        .create_texture_target(width, height)
        .expect("target builds");
    let camera = camera_of(FrameParams::IDENTITY, width, height);

    // The unit XY quad at identity covers the whole clip square, so every pixel
    // is inside it.
    let empty = Scene::new();
    let washed: Scene = [DrawableObject::quad_fill(Matrix4::IDENTITY)]
        .into_iter()
        .collect();

    let before =
        pollster::block_on(renderer.render_layers(&[SceneLayer::new(camera, &empty)], &target))
            .expect("the empty scene renders");
    let after =
        pollster::block_on(renderer.render_layers(&[SceneLayer::new(camera, &washed)], &target))
            .expect("the wash renders");

    assert_ne!(before, after, "the wash changed nothing on screen");
    let centre = ((height / 2) * width + width / 2) as usize * 4;
    let (base, lit) = (&before[centre..centre + 4], &after[centre..centre + 4]);
    assert!(
        lit[1] > base[1],
        "the wash must add green: {base:?} -> {lit:?}"
    );
    assert!(
        lit[0] <= base[0] + 1 && lit[2] <= base[2] + 1,
        "only green, and it must be translucent rather than opaque: {base:?} -> {lit:?}"
    );
}

/// The harness must render correctly into a **caller-owned** target that is
/// resized independently of the renderer.
///
/// `Renderer<T: RenderTarget>` used to bundle the render target as a field, with
/// `with_target`/`viewport`/`resize`/`into_target` forwarding to it — needed
/// because a live-surface shell had to reconfigure or hand back its swapchain
/// *through* the renderer. #203 removed the generic and the target field
/// entirely: the target is a plain call argument, and resizing it is a
/// [`Renderer`] method taking the target rather than a method *on* the target.
/// This exercises that shape end-to-end: build once, render, resize the target
/// out from under the renderer, render again at the new size.
#[test]
#[ignore = "requires a GPU adapter"]
fn render_params_after_resizing_a_caller_owned_target() {
    let mut renderer = single(TEXTURE_TARGET_FORMAT, &Mesh::hello_triangle());
    let scene: Scene = vec![DrawableObject::mesh(
        initial_mesh_id(&renderer, 0),
        Matrix4::IDENTITY,
        RenderMode::Filled,
    )]
    .into();
    let mut target = renderer
        .create_texture_target(40, 24)
        .expect("target builds");
    assert_eq!(
        target.viewport(),
        Viewport {
            width: 40,
            height: 24
        }
    );

    let small = pollster::block_on(renderer.render_params(FrameParams::IDENTITY, &scene, &target))
        .expect("renders at the initial size");
    assert_eq!(small.len(), 40 * 24 * 4);

    // The shell resizes its own target through the renderer; the renderer holds
    // no target of its own to keep in sync (#203).
    renderer
        .resize_texture_target(&mut target, 64, 32)
        .expect("resizes");
    assert_eq!(
        target.viewport(),
        Viewport {
            width: 64,
            height: 32
        }
    );
    let large = pollster::block_on(renderer.render_params(FrameParams::IDENTITY, &scene, &target))
        .expect("renders at the new size");
    assert_eq!(large.len(), 64 * 32 * 4);
    assert!(
        large.chunks_exact(4).any(|px| px[..3] != [0, 0, 0]),
        "the resized target drew nothing"
    );
}

/// [`Renderer::with_gpu`] must produce the same frame as [`Renderer::with_meshes`].
///
/// The streaming browser front-end owns its device before it owns its meshes, so
/// it builds the harness on an existing [`GpuContext`] instead of letting the
/// harness request one (#180). That second constructor is only safe if it is a
/// pure "bring your own device" variant — same auto-fit preview transforms, same
/// texture target, same pixels — so this renders one scene through both and
/// compares them byte for byte.
#[test]
#[ignore = "requires a GPU adapter"]
fn with_gpu_renders_identically_to_with_meshes() {
    use crate::Renderer;

    let (width, height) = (64, 64);
    let meshes = [Mesh::hello_triangle()];

    let (mut owned, owned_target) =
        pollster::block_on(Renderer::with_meshes(width, height, &meshes))
            .expect("the harness requests its own device");
    let owned_scene: Scene = vec![DrawableObject::mesh(
        initial_mesh_id(&owned, 0),
        Matrix4::IDENTITY,
        RenderMode::Filled,
    )]
    .into();
    let expected =
        pollster::block_on(owned.render_params(FrameParams::IDENTITY, &owned_scene, &owned_target))
            .expect("renders");

    let (mut borrowed, borrowed_target) = Renderer::with_gpu(test_gpu(), width, height, &meshes)
        .expect("the harness accepts an existing device");
    let borrowed_scene: Scene = vec![DrawableObject::mesh(
        initial_mesh_id(&borrowed, 0),
        Matrix4::IDENTITY,
        RenderMode::Filled,
    )]
    .into();
    let actual = pollster::block_on(borrowed.render_params(
        FrameParams::IDENTITY,
        &borrowed_scene,
        &borrowed_target,
    ))
    .expect("renders");

    assert_eq!(borrowed.mesh_count(), 1);
    assert_eq!(actual.len(), (width * height * 4) as usize);
    assert_eq!(
        actual, expected,
        "with_gpu and with_meshes disagree about the same scene"
    );
}

/// A freshly constructed renderer must be able to draw **without any setter
/// call at all**: no texture, no material maps, no environment probe.
///
/// This is the regression net for moving uploads out of `encode` (#180). Those
/// fallback bind groups used to be created lazily inside `encode` ("the only
/// place a GPU queue is available"); now that the renderer owns the queue they
/// are built in the constructors instead. Had that been missed, the very first
/// frame would panic in `bind_group()` — but only for a caller that sets
/// nothing, which every other test happens to avoid.
/// The per-mesh PBR slots are only re-uploaded when a setter has changed one
/// (#235 R5) — so the test that matters is that a change still **lands**: a
/// material edited between two frames must show, or the skip would be a stale
/// cache rather than an optimization.
#[test]
#[ignore = "requires a GPU adapter"]
fn a_material_change_between_frames_still_reaches_the_slots() {
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (48, 48);
    let mut renderer = single(format, &Mesh::hello_triangle());
    let scene: Scene = vec![DrawableObject::mesh(
        initial_mesh_id(&renderer, 0),
        Matrix4::IDENTITY,
        RenderMode::Shaded,
    )]
    .into();

    let frame = |renderer: &mut Renderer| {
        render_with_readback(&gpu, format, width, height, |encoder, view| {
            renderer
                .encode(
                    encoder,
                    view,
                    camera_of(FrameParams::IDENTITY, width, height),
                    &scene,
                )
                .expect("the scene is resident");
        })
    };

    renderer
        .set_disney_material(
            crate::MeshTarget::All,
            crate::DisneyMaterial {
                base_color: [0.9, 0.05, 0.05],
                metallic: 0.0,
                roughness: 0.5,
                ..Default::default()
            },
        )
        .expect("all meshes are resident");
    let red = frame(&mut renderer);
    // A second frame with nothing changed: the slots are skipped, and the image
    // must be identical — the skip is invisible.
    let red_again = frame(&mut renderer);
    assert_eq!(red, red_again, "an unchanged scene renders identically");

    renderer
        .set_disney_material(
            crate::MeshTarget::All,
            crate::DisneyMaterial {
                base_color: [0.05, 0.05, 0.9],
                metallic: 0.0,
                roughness: 0.5,
                ..Default::default()
            },
        )
        .expect("all meshes are resident");
    let blue = frame(&mut renderer);
    assert_ne!(
        red, blue,
        "a material set between frames must reach the GPU slots"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn first_frame_renders_with_no_setters_called() {
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (48, 48);
    let mut renderer = single(format, &Mesh::hello_triangle());

    // Every draw kind that reads a lazily-bound resource: Textured samples the
    // albedo, Pbr samples albedo + material maps + the environment probe.
    for mode in [RenderMode::Textured, RenderMode::Shaded] {
        let scene: Scene = vec![DrawableObject::mesh(
            initial_mesh_id(&renderer, 0),
            Matrix4::IDENTITY,
            mode,
        )]
        .into();
        let pixels = render_with_readback(&gpu, format, width, height, |encoder, view| {
            renderer
                .encode(
                    encoder,
                    view,
                    camera_of(FrameParams::IDENTITY, width, height),
                    &scene,
                )
                .expect("the scene is resident");
        });
        assert_eq!(pixels.len(), (width * height * 4) as usize);
        assert!(
            pixels
                .chunks_exact(4)
                .any(|px| px[3] > 0 && px[..3] != [0, 0, 0]),
            "{mode:?} drew nothing on the first frame with no setters called"
        );
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn mesh_renderer_draws_multiple_instances() {
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (64, 64);
    let mut mesh = single(format, &Mesh::hello_triangle());
    let mesh_id = initial_mesh_id(&mesh, 0);

    // One centered instance vs. two instances translated to opposite sides.
    let single: Scene = [DrawableObject::mesh(
        mesh_id,
        Matrix4::IDENTITY,
        RenderMode::Filled,
    )]
    .into_iter()
    .collect();
    let single_px = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            camera_of(FrameParams::IDENTITY, width, height),
            &single,
        )
        .expect("the scene is resident");
    });

    let two: Scene = [
        DrawableObject::mesh(
            mesh_id,
            Matrix4::from_translation(Vector3::new(-0.4, 0.0, 0.0)),
            RenderMode::Filled,
        ),
        DrawableObject::mesh(
            mesh_id,
            Matrix4::from_translation(Vector3::new(0.4, 0.0, 0.0)),
            RenderMode::Filled,
        ),
    ]
    .into_iter()
    .collect();
    let two_px = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(e, v, camera_of(FrameParams::IDENTITY, width, height), &two)
            .expect("the scene is resident");
    });

    assert_ne!(
        single_px, two_px,
        "two translated instances must differ from one centered instance"
    );

    // The two-instance frame must have colored pixels in both the left and
    // right thirds of the image (one triangle each).
    let has_color_in = |xs: std::ops::Range<u32>| {
        xs.into_iter().any(|x| {
            (0..height).any(|y| {
                let i = ((y * width + x) * 4) as usize;
                two_px[i] > 0 || two_px[i + 1] > 0 || two_px[i + 2] > 0
            })
        })
    };
    assert!(has_color_in(0..width / 3), "left instance is missing");
    assert!(
        has_color_in(2 * width / 3..width),
        "right instance is missing"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn mesh_renderer_depth_buffer_occludes_far_behind_near() {
    // Two full-screen quads fully overlapping in screen space: a RED quad
    // nearer the camera (NDC z=0.25) and a GREEN quad farther (z=0.75). The
    // scene submits RED first (mesh 0) and GREEN last (mesh 1), so *without* a
    // depth buffer the later-drawn GREEN would overwrite RED (submission-order
    // painter's algorithm). With the depth buffer the nearer RED must win —
    // proving solid meshes z-occlude instead of last-draw-wins (the bug that
    // let textured meshes sample back faces over the front, #20).
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (64, 64);

    let quad = |rgb: [f32; 3]| Mesh {
        vertices: vec![
            Vertex {
                position: [-1.0, 1.0, 0.0],
                color: rgb,
                uv: [0.0, 0.0],
            },
            Vertex {
                position: [1.0, 1.0, 0.0],
                color: rgb,
                uv: [0.0, 0.0],
            },
            Vertex {
                position: [-1.0, -1.0, 0.0],
                color: rgb,
                uv: [0.0, 0.0],
            },
            Vertex {
                position: [1.0, -1.0, 0.0],
                color: rgb,
                uv: [0.0, 0.0],
            },
        ],
        indices: vec![0, 2, 3, 0, 3, 1],
        shading: None,
    };
    // mesh 0 = red (drawn first), mesh 1 = green (drawn last).
    let mut mesh = Renderer::new(
        test_gpu(),
        format,
        &[quad([1.0, 0.0, 0.0]), quad([0.0, 1.0, 0.0])],
        &[Matrix4::IDENTITY, Matrix4::IDENTITY],
    )
    .expect("two meshes with two base models is a valid mesh set");
    let red_mesh = initial_mesh_id(&mesh, 0);
    let green_mesh = initial_mesh_id(&mesh, 1);
    let scene: Scene = [
        DrawableObject::mesh(
            red_mesh,
            Matrix4::from_translation(Vector3::new(0.0, 0.0, 0.25)),
            RenderMode::Filled,
        ),
        DrawableObject::mesh(
            green_mesh,
            Matrix4::from_translation(Vector3::new(0.0, 0.0, 0.75)),
            RenderMode::Filled,
        ),
    ]
    .into_iter()
    .collect();
    let px = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            camera_of(FrameParams::IDENTITY, width, height),
            &scene,
        )
        .expect("the scene is resident");
    });
    let c = ((height / 2 * width + width / 2) * 4) as usize;
    let (r, g, b) = (px[c], px[c + 1], px[c + 2]);
    assert!(
        r > 200 && g < 70 && b < 70,
        "nearer red quad must occlude the farther green quad drawn after it, \
             got rgb=({r},{g},{b})"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn mesh_renderer_wireframe_lights_edges_only() {
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (64, 64);
    let mut mesh = single(format, &Mesh::hello_triangle());
    let mesh_id = initial_mesh_id(&mesh, 0);
    let model = Matrix4::IDENTITY;
    let filled_scene: Scene = [DrawableObject::mesh(mesh_id, model, RenderMode::Filled)]
        .into_iter()
        .collect();
    let wire_scene: Scene = [DrawableObject::mesh(mesh_id, model, RenderMode::Wireframe)]
        .into_iter()
        .collect();

    let filled = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            camera_of(FrameParams::IDENTITY, width, height),
            &filled_scene,
        )
        .expect("the scene is resident");
    });

    let wire = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            camera_of(FrameParams::IDENTITY, width, height),
            &wire_scene,
        )
        .expect("the scene is resident");
    });

    assert_ne!(filled, wire, "wireframe must differ from filled");

    let lit = |px: &[u8]| -> usize {
        (0..(width * height) as usize)
            .filter(|i| {
                let b = i * 4;
                px[b] > 0 || px[b + 1] > 0 || px[b + 2] > 0
            })
            .count()
    };
    let (filled_lit, wire_lit) = (lit(&filled), lit(&wire));
    assert!(wire_lit > 0, "wireframe must light its edges");
    assert!(
        wire_lit < filled_lit,
        "wireframe ({wire_lit}) must light fewer pixels than filled ({filled_lit})"
    );

    // The triangle's centroid is interior: filled there, background in
    // wireframe (no edge crosses the center of mass).
    let centroid = {
        let v = &Mesh::hello_triangle().vertices;
        let cx = (v[0].position[0] + v[1].position[0] + v[2].position[0]) / 3.0;
        let cy = (v[0].position[1] + v[1].position[1] + v[2].position[1]) / 3.0;
        // NDC (clip, y-up) -> pixel (y-down).
        let px = ((cx * 0.5 + 0.5) * width as f32).round() as u32;
        let py = ((1.0 - (cy * 0.5 + 0.5)) * height as f32).round() as u32;
        ((py.min(height - 1) * width + px.min(width - 1)) * 4) as usize
    };
    assert!(
        filled[centroid] > 0 || filled[centroid + 1] > 0 || filled[centroid + 2] > 0,
        "filled centroid must be lit"
    );
    assert_eq!(
        (wire[centroid], wire[centroid + 1], wire[centroid + 2]),
        (0, 0, 0),
        "wireframe centroid must be background"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn mesh_renderer_textured_samples_bound_texture() {
    // A full-screen quad mapped 1:1 to a 2×2 checker texture (#20): white,
    // red / green, blue (top-left origin). Each screen quadrant must show the
    // matching texel color, proving the textured pipeline samples the bound
    // texture at the interpolated vertex UVs with the correct orientation.
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (64, 64);

    // Quad in NDC: uv(0,0) at the top-left corner (NDC (-1,+1)), so the
    // texture's top-left texel lands in the framebuffer's top-left quadrant.
    let quad = Mesh {
        vertices: vec![
            Vertex {
                position: [-1.0, 1.0, 0.0],
                color: [0.0; 3],
                uv: [0.0, 0.0],
            }, // top-left
            Vertex {
                position: [1.0, 1.0, 0.0],
                color: [0.0; 3],
                uv: [1.0, 0.0],
            }, // top-right
            Vertex {
                position: [-1.0, -1.0, 0.0],
                color: [0.0; 3],
                uv: [0.0, 1.0],
            }, // bottom-left
            Vertex {
                position: [1.0, -1.0, 0.0],
                color: [0.0; 3],
                uv: [1.0, 1.0],
            }, // bottom-right
        ],
        indices: vec![0, 2, 3, 0, 3, 1],
        shading: None,
    };

    // 2×2 checker, row-major top-left origin: white, red / green, blue.
    let checker = crate::texture::ImageTexture::from_rgba(
        2,
        2,
        vec![
            255, 255, 255, 255, 255, 0, 0, 255, // white, red
            0, 255, 0, 255, 0, 0, 255, 255, // green, blue
        ],
    )
    .unwrap();

    let mut mesh = single(format, &quad);
    mesh.set_texture(&checker)
        .expect("the initial mesh is resident");
    let scene: Scene = [DrawableObject::mesh(
        initial_mesh_id(&mesh, 0),
        Matrix4::IDENTITY,
        RenderMode::Textured,
    )]
    .into_iter()
    .collect();
    let pixels = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            camera_of(FrameParams::IDENTITY, width, height),
            &scene,
        )
        .expect("the scene is resident");
    });

    // Read a pixel at the center of each screen quadrant (well away from the
    // uv=0.5 seams, so any bilinear bleed is negligible).
    let at = |x: u32, y: u32| -> [u8; 3] {
        let i = ((y * width + x) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2]]
    };
    let tl = at(width / 4, height / 4);
    let tr = at(3 * width / 4, height / 4);
    let bl = at(width / 4, 3 * height / 4);
    let br = at(3 * width / 4, 3 * height / 4);

    let dominant =
        |c: [u8; 3], hi: [bool; 3]| (0..3).all(|k| if hi[k] { c[k] > 200 } else { c[k] < 70 });
    assert!(
        dominant(tl, [true, true, true]),
        "top-left must be white, got {tl:?}"
    );
    assert!(
        dominant(tr, [true, false, false]),
        "top-right must be red, got {tr:?}"
    );
    assert!(
        dominant(bl, [false, true, false]),
        "bottom-left must be green, got {bl:?}"
    );
    assert!(
        dominant(br, [false, false, true]),
        "bottom-right must be blue, got {br:?}"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn mesh_renderer_aabb_overlay_draws_green_box() {
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (64, 64);
    let mut mesh = single(format, &Mesh::hello_triangle());
    let mesh_id = initial_mesh_id(&mesh, 0);
    let model = Matrix4::IDENTITY;
    let plain_scene: Scene = [DrawableObject::mesh(mesh_id, model, RenderMode::Filled)]
        .into_iter()
        .collect();
    let box_scene: Scene = [
        DrawableObject::mesh(mesh_id, model, RenderMode::Filled),
        DrawableObject::aabb_box(mesh_id, model),
    ]
    .into_iter()
    .collect();

    let plain = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            camera_of(FrameParams::IDENTITY, width, height),
            &plain_scene,
        )
        .expect("the scene is resident");
    });

    let with_box = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            camera_of(FrameParams::IDENTITY, width, height),
            &box_scene,
        )
        .expect("the scene is resident");
    });

    assert_ne!(plain, with_box, "AABB overlay must change the image");

    // The overlay must light pure-green pixels (R≈0, G>0, B≈0) that are not
    // present without it — the box drawn in AABB_COLOR = [0, 1, 0].
    let pure_green = |px: &[u8]| -> usize {
        (0..(width * height) as usize)
            .filter(|i| {
                let b = i * 4;
                px[b] == 0 && px[b + 1] > 0 && px[b + 2] == 0
            })
            .count()
    };
    assert_eq!(
        pure_green(&plain),
        0,
        "no green box expected without the overlay"
    );
    assert!(
        pure_green(&with_box) > 0,
        "AABB overlay must light green box pixels"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn mesh_renderer_axes_overlay_draws_rgb_gizmo() {
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (64, 64);
    let mut mesh = single(format, &Mesh::hello_triangle());
    let mesh_id = initial_mesh_id(&mesh, 0);
    let model = Matrix4::IDENTITY;
    let plain_scene: Scene = [DrawableObject::mesh(mesh_id, model, RenderMode::Filled)]
        .into_iter()
        .collect();
    let axes_scene: Scene = [
        DrawableObject::mesh(mesh_id, model, RenderMode::Filled),
        DrawableObject::coordinate_axes(Matrix4::IDENTITY),
    ]
    .into_iter()
    .collect();

    let plain = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            camera_of(FrameParams::IDENTITY, width, height),
            &plain_scene,
        )
        .expect("the scene is resident");
    });

    let with_axes = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            camera_of(FrameParams::IDENTITY, width, height),
            &axes_scene,
        )
        .expect("the scene is resident");
    });

    assert_ne!(plain, with_axes, "axes overlay must change the image");

    // Under the identity camera the +X axis draws a pure-red horizontal line
    // and the +Y axis a pure-green vertical line from the center; both must
    // add colored pixels beyond whatever the filled triangle already lit.
    let count = |px: &[u8], pred: fn(u8, u8, u8) -> bool| -> usize {
        (0..(width * height) as usize)
            .filter(|i| {
                let b = i * 4;
                pred(px[b], px[b + 1], px[b + 2])
            })
            .count()
    };
    let pure_red = |r: u8, g: u8, b: u8| r > 0 && g == 0 && b == 0;
    let pure_green = |r: u8, g: u8, b: u8| r == 0 && g > 0 && b == 0;

    assert!(
        count(&with_axes, pure_red) > count(&plain, pure_red),
        "X axis must add pure-red pixels"
    );
    assert!(
        count(&with_axes, pure_green) > count(&plain, pure_green),
        "Y axis must add pure-green pixels"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn gizmo_lines_and_arrowheads_stay_smooth_without_msaa() {
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (128, 128);
    let mesh_data = Mesh::hello_triangle();
    let mut renderer = Renderer::with_sample_count(
        test_gpu(),
        format,
        std::slice::from_ref(&mesh_data),
        &[Matrix4::IDENTITY],
        1,
    )
    .expect("one mesh, one base model, sample_count 1");
    let model = Matrix4::from_scale(Vector3::new(0.5, 0.5, 0.5));
    let scene: Scene = [DrawableObject::coordinate_axes(model)]
        .into_iter()
        .collect();

    let pixels = render_with_readback(&gpu, format, width, height, |e, v| {
        renderer
            .encode(
                e,
                v,
                camera_of(FrameParams::IDENTITY, width, height),
                &scene,
            )
            .expect("the scene has no mesh resources");
    });
    let red = |x: u32, y: u32| {
        let i = ((y * width + x) * 4) as usize;
        pixels[i]
    };

    // Halfway along +X, the 3px shaft plus its analytic feather covers multiple
    // rows and includes partial-alpha edge pixels even in this single-sample pass.
    let shaft_x = 80;
    let shaft_rows = (56..72)
        .filter(|&y| red(shaft_x, y) > 0)
        .collect::<Vec<_>>();
    assert!(
        shaft_rows.len() >= 4,
        "expanded shaft should cover multiple rows, got {shaft_rows:?}"
    );
    assert!(
        shaft_rows
            .iter()
            .any(|&y| (1..=254).contains(&red(shaft_x, y))),
        "shaft edge should contain analytically anti-aliased pixels"
    );

    // Beyond the shortened +X shaft, a red pixel off the centerline proves the
    // cone geometry continues to the original axis tip.
    let arrow_wing_lit =
        (107u32..111).any(|x| (58u32..70).any(|y| y.abs_diff(height / 2) >= 2 && red(x, y) > 0));
    assert!(
        arrow_wing_lit,
        "the +X arrowhead should light pixels outside the shaft"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn scene_composes_all_drawable_kinds_together() {
    // #41: every primitive is a `DrawableObject`, and every front-end submits
    // the same heterogeneous `Scene` for draw-kind batching. A scene mixing a
    // filled mesh, a wireframe mesh, an AABB box, and the axes gizmo must
    // render all of them at once — the filled mesh alone lights fewer pixels
    // than the full composed scene, and the green box + RGB axes appear.
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (64, 64);
    let mut mesh = single(format, &Mesh::hello_triangle());
    let mesh_id = initial_mesh_id(&mesh, 0);
    let model = Matrix4::IDENTITY;

    let filled_only: Scene = [DrawableObject::mesh(mesh_id, model, RenderMode::Filled)]
        .into_iter()
        .collect();
    let composed: Scene = [
        DrawableObject::mesh(mesh_id, model, RenderMode::Filled),
        DrawableObject::mesh(mesh_id, model, RenderMode::Wireframe),
        DrawableObject::aabb_box(mesh_id, model),
        DrawableObject::coordinate_axes(Matrix4::IDENTITY),
    ]
    .into_iter()
    .collect();

    let filled_px = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            camera_of(FrameParams::IDENTITY, width, height),
            &filled_only,
        )
        .expect("the scene is resident");
    });
    let composed_px = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            camera_of(FrameParams::IDENTITY, width, height),
            &composed,
        )
        .expect("the scene is resident");
    });

    assert_ne!(
        filled_px, composed_px,
        "the composed scene must differ from the filled mesh alone"
    );

    // The AABB box (pure green) and the axes gizmo (pure red +X line) are
    // both present only in the composed scene.
    let count = |px: &[u8], pred: fn(u8, u8, u8) -> bool| -> usize {
        (0..(width * height) as usize)
            .filter(|i| {
                let b = i * 4;
                pred(px[b], px[b + 1], px[b + 2])
            })
            .count()
    };
    let pure_green = |r: u8, g: u8, b: u8| r == 0 && g > 0 && b == 0;
    let pure_red = |r: u8, g: u8, b: u8| r > 0 && g == 0 && b == 0;
    assert!(
        count(&composed_px, pure_green) > 0,
        "AABB box must light green pixels in the composed scene"
    );
    assert!(
        count(&composed_px, pure_red) > count(&filled_px, pure_red),
        "axes gizmo must add pure-red pixels in the composed scene"
    );
}

/// A unit quad centered at the origin in the z=0 plane, spanning
/// `[-0.5, 0.5]²`. Used to render a *loaded* mesh (not the baked triangle).
const QUAD_OBJ: &str = "\
v -0.5 -0.5 0.0
v 0.5 -0.5 0.0
v 0.5 0.5 0.0
v -0.5 0.5 0.0
f 1 2 3 4
";

// A unit cube centered at the origin (±0.5) — a mesh with real depth extent,
// used by the dolly-turntable scenario test to make near/far framing matter.
const CUBE_OBJ: &str = "\
v -0.5 -0.5 -0.5
v 0.5 -0.5 -0.5
v 0.5 0.5 -0.5
v -0.5 0.5 -0.5
v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v 0.5 0.5 0.5
v -0.5 0.5 0.5
f 1 2 3 4
f 5 6 7 8
f 1 5 8 4
f 2 6 7 3
f 4 3 7 8
f 1 2 6 5
";

#[test]
#[ignore = "requires a GPU adapter"]
fn environment_background_draws_bound_probe() {
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (32, 32);
    let mut renderer = single(format, &Mesh::hello_triangle());
    renderer.set_env_map(EnvMapData::from_rgba32f(
        2,
        1,
        vec![4.0, 0.1, 0.1, 1.0, 4.0, 0.1, 0.1, 1.0],
        2048,
    ));
    let scene = Scene::new().with_background(Background {
        environment: Some(EnvironmentBackground {
            exposure: 1.0,
            blur: 0.0,
            tonemap: Tonemap::Reinhard,
        }),
        frame: None,
    });
    let pixels = render_with_readback(&gpu, format, width, height, |encoder, view| {
        renderer
            .encode(
                encoder,
                view,
                camera_of(FrameParams::IDENTITY, width, height),
                &scene,
            )
            .expect("the scene has no mesh resources");
    });
    assert!(
        pixels.chunks_exact(4).any(|pixel| pixel[0] > 128),
        "environment background should produce visible probe color"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn mesh_renderer_renders_loaded_quad_filled_with_correct_coverage() {
    // #37/#41: a mesh loaded from OBJ (not the baked triangle) renders filled
    // via `draw_indexed` as a `Primitive::Mesh`. Under the identity camera
    // the unit quad spans NDC [-0.5, 0.5]², i.e. the central quarter of the
    // frame — so the center is lit, the corners are dark, and coverage ≈ 25%.
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (64, 64);
    let quad = Mesh::from_obj(QUAD_OBJ).expect("quad OBJ parses");
    let mut mesh = single(format, &quad);
    let mesh_id = initial_mesh_id(&mesh, 0);

    let scene: Scene = [DrawableObject::mesh(
        mesh_id,
        Matrix4::IDENTITY,
        RenderMode::Filled,
    )]
    .into_iter()
    .collect();
    let px = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            camera_of(FrameParams::IDENTITY, width, height),
            &scene,
        )
        .expect("the scene is resident");
    });

    let (w, h) = (width as usize, height as usize);
    let covered = |x: usize, y: usize| -> bool {
        let b = (y * w + x) * 4;
        px[b] > 0 || px[b + 1] > 0 || px[b + 2] > 0
    };
    assert!(covered(w / 2, h / 2), "quad center must be covered");
    assert!(!covered(1, 1), "top-left corner must be outside the quad");
    assert!(
        !covered(w - 2, 1),
        "top-right corner must be outside the quad"
    );
    assert!(
        !covered(1, h - 2),
        "bottom-left corner must be outside the quad"
    );
    assert!(
        !covered(w - 2, h - 2),
        "bottom-right corner must be outside the quad"
    );

    let covered_count = (0..w * h)
        .filter(|i| {
            let b = i * 4;
            px[b] > 0 || px[b + 1] > 0 || px[b + 2] > 0
        })
        .count();
    let frac = covered_count as f32 / (w * h) as f32;
    assert!(
        (0.18..=0.32).contains(&frac),
        "quad coverage {frac} is not ≈ the central quarter (0.25)"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn cg_and_cv_cameras_render_matching_output() {
    // #43/#49: a CG-authored camera (eye/target/up/fovy) and its CV-lowered
    // equivalent (pose = world-from-camera, K = intrinsics) describe the *same*
    // camera, so rendering the same `Scene` under each yields matching pixels.
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (96, 96);
    let viewport = Viewport { width, height };
    let quad = Mesh::from_obj(QUAD_OBJ).expect("quad OBJ parses");
    let mut mesh = single(format, &quad);
    let mesh_id = initial_mesh_id(&mesh, 0);

    let scene: Scene = [DrawableObject::mesh(
        mesh_id,
        Matrix4::IDENTITY,
        RenderMode::Filled,
    )]
    .into_iter()
    .collect();

    // An off-axis camera so orientation actually matters.
    let eye_arr = [0.6f32, 0.4, 1.4];
    let target_arr = [0.0f32, 0.0, 0.0];
    let up_arr = [0.0f32, 1.0, 0.0];
    let fovy = crate::DEFAULT_FOV_Y;

    let cg = FrameParams {
        eye: Some(eye_arr),
        target: Some(target_arr),
        up: Some(up_arr),
        fovy: Some(fovy),
        ..FrameParams::IDENTITY
    };

    // Lower the same camera to CV form (K + pose) via the camera API.
    let cam = crate::Camera::look_at(
        Point3::new(eye_arr[0], eye_arr[1], eye_arr[2]),
        Point3::new(target_arr[0], target_arr[1], target_arr[2]),
        Vector3::new(up_arr[0], up_arr[1], up_arr[2]),
        fovy,
        viewport,
    );
    let cv = FrameParams {
        pose: Some(cam.to_pose().matrix().to_cols_array()),
        k: Some(cam.to_intrinsics()),
        ..FrameParams::IDENTITY
    };

    let cg_px = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(e, v, cg.to_camera(viewport).unwrap(), &scene)
            .expect("the scene is resident");
    });
    let cv_px = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(e, v, cv.to_camera(viewport).unwrap(), &scene)
            .expect("the scene is resident");
    });

    // Both must actually show the quad (non-trivial coverage).
    let lit = |px: &[u8]| {
        px.chunks_exact(4)
            .filter(|c| c[0] > 0 || c[1] > 0 || c[2] > 0)
            .count()
    };
    assert!(lit(&cg_px) > 200, "CG camera must render the quad");
    assert!(lit(&cv_px) > 200, "CV camera must render the quad");

    // ...and their outputs must match within a tiny tolerance (a few edge
    // pixels may differ by rounding in the K⇄projection round-trip).
    let differing = cg_px
        .chunks_exact(4)
        .zip(cv_px.chunks_exact(4))
        .filter(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .any(|(x, y)| (i16::from(*x) - i16::from(*y)).abs() > 2)
        })
        .count();
    let frac = differing as f32 / (width * height) as f32;
    assert!(
        frac < 0.01,
        "CG and CV renders differ in {differing} px (fraction {frac})"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn dolly_turntable_bird_eye_cg_cv_wireframe_stays_framed() {
    // #49 scenario end-to-end: a fixed 45° bird's-eye camera dollies in and
    // out while a mesh spins about +Y, rendered as a **wireframe**. At every
    // (dolly distance, spin angle) this asserts the three defining behaviors
    // of the slice:
    //   (a) the CG-authored camera (eye/target/up/fovy) and its CV-lowered
    //       equivalent (pose + K) render matching pixels;
    //   (b) near/far fit: the spinning mesh stays fully framed — visible
    //       wireframe, empty frame border (nothing clipped at any distance);
    //   (c) the dolly actually reframes: dollying in covers more pixels than
    //       dollying out.
    // (A cube stands in for the bunny for a fast, deterministic GPU test; the
    // same scenario is exercised on the real bunny by examples/bunny_dolly.py
    // + `render.sh --wireframe`.)
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (128, 128);
    let viewport = Viewport { width, height };
    let cube = Mesh::from_obj(CUBE_OBJ).expect("cube OBJ parses");
    let mut mesh = single(format, &cube);
    let mesh_id = initial_mesh_id(&mesh, 0);

    let fovy = crate::DEFAULT_FOV_Y; // 45°
                                     // Fixed bird's-eye view direction: 45° elevation, 35° azimuth (unit).
    let elev = 45f32.to_radians();
    let azim = 35f32.to_radians();
    let view_dir =
        Vec3::new(elev.cos() * azim.sin(), elev.sin(), elev.cos() * azim.cos()).normalize();
    let target = Point3::new(0.0, 0.0, 0.0);
    let up = Vector3::Y;

    // Dolly-in (near) → mid → dolly-out (far).
    let distances = [3.5f32, 4.75, 6.0];
    // Turntable spin angles about +Y.
    let angles = [0.0f32, std::f32::consts::FRAC_PI_2, 2.4];

    let lit = |px: &[u8]| -> usize {
        px.chunks_exact(4)
            .filter(|c| c[0] > 0 || c[1] > 0 || c[2] > 0)
            .count()
    };
    // Lit pixels in the outer 2-px ring — must stay 0 (mesh never clipped).
    let border_lit = |px: &[u8]| -> usize {
        let w = width as usize;
        let h = height as usize;
        let mut n = 0;
        for y in 0..h {
            for x in 0..w {
                if x < 2 || x >= w - 2 || y < 2 || y >= h - 2 {
                    let b = (y * w + x) * 4;
                    if px[b] > 0 || px[b + 1] > 0 || px[b + 2] > 0 {
                        n += 1;
                    }
                }
            }
        }
        n
    };

    let mut lit_at_zero_spin = Vec::new();
    for &dist in &distances {
        let eye_arr = [view_dir.x * dist, view_dir.y * dist, view_dir.z * dist];
        let eye = Point3::new(eye_arr[0], eye_arr[1], eye_arr[2]);
        // Lower the same camera to CV form (K + pose) once per distance.
        let cam = crate::Camera::look_at(eye, target, up, fovy, viewport);
        let pose = cam.to_pose().matrix().to_cols_array();
        let k = cam.to_intrinsics();

        for &theta in &angles {
            let scene: Scene = [DrawableObject::mesh(
                mesh_id,
                Matrix4::from_glam(Mat4::from_rotation_y(theta)),
                RenderMode::Wireframe,
            )]
            .into_iter()
            .collect();

            let cg = FrameParams {
                eye: Some(eye_arr),
                target: Some([0.0, 0.0, 0.0]),
                up: Some([0.0, 1.0, 0.0]),
                fovy: Some(fovy),
                ..FrameParams::IDENTITY
            };
            let cv = FrameParams {
                pose: Some(pose),
                k: Some(k),
                ..FrameParams::IDENTITY
            };

            let cg_px = render_with_readback(&gpu, format, width, height, |e, v| {
                mesh.encode(e, v, cg.to_camera(viewport).unwrap(), &scene)
                    .expect("the scene is resident");
            });
            let cv_px = render_with_readback(&gpu, format, width, height, |e, v| {
                mesh.encode(e, v, cv.to_camera(viewport).unwrap(), &scene)
                    .expect("the scene is resident");
            });

            let cg_lit = lit(&cg_px);
            // (b) near/far fit: visible wireframe, but framed — never fills
            // the frame and never touches the border (nothing clipped).
            assert!(
                cg_lit > 20,
                "dist {dist} theta {theta}: wireframe must be visible (near/far fit)"
            );
            assert!(
                (cg_lit as f32) < 0.5 * (width * height) as f32,
                "dist {dist} theta {theta}: mesh must stay framed, not overflow ({cg_lit} px)"
            );
            assert_eq!(
                border_lit(&cg_px),
                0,
                "dist {dist} theta {theta}: mesh must not touch the frame border (stays framed)"
            );

            // (a) CG and CV forms render matching pixels (few edge pixels may
            // differ by rounding in the K⇄projection round-trip).
            let differing = cg_px
                .chunks_exact(4)
                .zip(cv_px.chunks_exact(4))
                .filter(|(a, b)| {
                    a.iter()
                        .zip(b.iter())
                        .any(|(x, y)| (i16::from(*x) - i16::from(*y)).abs() > 2)
                })
                .count();
            let frac = differing as f32 / (width * height) as f32;
            assert!(
                frac < 0.02,
                "dist {dist} theta {theta}: CG vs CV differ in {differing} px ({frac})"
            );

            if theta == angles[0] {
                lit_at_zero_spin.push((dist, cg_lit));
            }
        }
    }

    // (c) the dolly reframes the mesh: closer distance ⇒ larger footprint.
    for pair in lit_at_zero_spin.windows(2) {
        let (near_d, near_lit) = pair[0];
        let (far_d, far_lit) = pair[1];
        assert!(
                near_lit > far_lit,
                "dolly-in ({near_d}, {near_lit}px) must cover more than dolly-out ({far_d}, {far_lit}px)"
            );
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn frame_plane_composites_background_under_scene() {
    // The scene's background frame plane fills the frame from a reused 2×2 texture (upload +
    // fullscreen sample + top-left `v=0` orientation), and a solid mesh drawn
    // in the same scene z-composites ON TOP of it (depth-write-off plane).
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (64, 64);

    // A green fullscreen quad at z=0.5 (in front of the plane's cleared depth
    // 1.0), used to prove the mesh occludes the background.
    let quad = Mesh {
        vertices: vec![
            Vertex {
                position: [-1.0, 1.0, 0.5],
                color: [0.0, 1.0, 0.0],
                uv: [0.0, 0.0],
            },
            Vertex {
                position: [1.0, 1.0, 0.5],
                color: [0.0, 1.0, 0.0],
                uv: [1.0, 0.0],
            },
            Vertex {
                position: [-1.0, -1.0, 0.5],
                color: [0.0, 1.0, 0.0],
                uv: [0.0, 1.0],
            },
            Vertex {
                position: [1.0, -1.0, 0.5],
                color: [0.0, 1.0, 0.0],
                uv: [1.0, 1.0],
            },
        ],
        indices: vec![0, 2, 3, 0, 3, 1],
        shading: None,
    };
    let mut mesh = single(format, &quad);
    let mesh_id = initial_mesh_id(&mesh, 0);

    // 2×2 background, row-major top-left origin: white, red / green, blue.
    assert!(!mesh.has_frame_texture());
    mesh.update_frame_texture_rgba(
        &[
            255, 255, 255, 255, 255, 0, 0, 255, // white, red
            0, 255, 0, 255, 0, 0, 255, 255, // green, blue
        ],
        2,
        2,
    );
    assert!(mesh.has_frame_texture());

    // (a) Plane only: each screen quadrant shows the matching background texel.
    // The frame plane is a background *setting* now, so the "plane only" scene
    // has no objects at all (#204).
    let plane_only = Scene::new().with_background(Background {
        environment: None,
        frame: Some(FrameFit::Stretch),
    });
    let bg = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            camera_of(FrameParams::IDENTITY, width, height),
            &plane_only,
        )
        .expect("the scene has no mesh resources");
    });
    let at = |px: &[u8], x: u32, y: u32| -> [u8; 3] {
        let i = ((y * width + x) * 4) as usize;
        [px[i], px[i + 1], px[i + 2]]
    };
    let dominant =
        |c: [u8; 3], hi: [bool; 3]| (0..3).all(|k| if hi[k] { c[k] > 200 } else { c[k] < 70 });
    assert!(
        dominant(at(&bg, width / 4, height / 4), [true, true, true]),
        "background top-left must be white, got {:?}",
        at(&bg, width / 4, height / 4)
    );
    assert!(
        dominant(at(&bg, 3 * width / 4, height / 4), [true, false, false]),
        "background top-right must be red, got {:?}",
        at(&bg, 3 * width / 4, height / 4)
    );
    assert!(
        dominant(at(&bg, width / 4, 3 * height / 4), [false, true, false]),
        "background bottom-left must be green, got {:?}",
        at(&bg, width / 4, 3 * height / 4)
    );
    assert!(
        dominant(at(&bg, 3 * width / 4, 3 * height / 4), [false, false, true]),
        "background bottom-right must be blue, got {:?}",
        at(&bg, 3 * width / 4, 3 * height / 4)
    );

    // (b) Plane + solid green mesh (the plane is the scene's background, the mesh
    // its one object): the whole frame is green, proving the mesh composites over
    // the plane.
    let composited = Scene::from(vec![DrawableObject::mesh(
        mesh_id,
        Matrix4::IDENTITY,
        RenderMode::Filled,
    )])
    .with_background(Background {
        environment: None,
        frame: Some(FrameFit::Stretch),
    });
    let over = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            camera_of(FrameParams::IDENTITY, width, height),
            &composited,
        )
        .expect("the scene is resident");
    });
    let foreground: Scene = [DrawableObject::mesh(
        mesh_id,
        Matrix4::IDENTITY,
        RenderMode::Filled,
    )]
    .into_iter()
    .collect();
    let two_pass = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            camera_of(FrameParams::IDENTITY, width, height),
            &plane_only,
        )
        .expect("the scene has no mesh resources");
        mesh.encode_overlay(
            e,
            v,
            camera_of(FrameParams::IDENTITY, width, height),
            &foreground,
        )
        .expect("the foreground is resident");
    });
    assert_eq!(
        two_pass, over,
        "two-pass composite must match one-pass output"
    );
    for &(x, y) in &[
        (width / 4, height / 4),
        (3 * width / 4, height / 4),
        (width / 4, 3 * height / 4),
        (3 * width / 4, 3 * height / 4),
    ] {
        assert!(
            dominant(at(&over, x, y), [false, true, false]),
            "mesh must occlude the background at ({x},{y}), got {:?}",
            at(&over, x, y)
        );
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn triangle_renderer_draws_gradient_triangle() {
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (64, 64);
    let renderer = TriangleRenderer::new(&gpu.device, format);

    let px = render_with_readback(&gpu, format, width, height, |e, v| {
        renderer.encode(e, v);
    });
    let at = |x: u32, y: u32| -> [u8; 3] {
        let i = ((y * width + x) * 4) as usize;
        [px[i], px[i + 1], px[i + 2]]
    };

    // The triangle covers screen center; a top corner is outside it (background).
    assert_ne!(
        at(width / 2, height / 2),
        [0, 0, 0],
        "center must be inside the triangle"
    );
    assert_eq!(
        at(0, 0),
        [0, 0, 0],
        "top-left corner must be background (black)"
    );

    // Per-vertex colors interpolate across the face: a point near the red apex is
    // redder than a point near the blue bottom-right corner, and vice versa —
    // proving a gradient rather than a flat fill.
    let near_apex = at(width / 2, 22);
    let near_blue = at(42, 46);
    assert!(
        near_apex[0] > near_blue[0] && near_blue[2] > near_apex[2],
        "gradient expected: near-apex {near_apex:?} redder, near-blue {near_blue:?} bluer",
    );
}

/// #141: the object-id ("color index") picking pass resolves *which* object a
/// pixel hit. Two quads are placed side by side (a gap between them) under the
/// identity camera; clicking the left one returns object index 0, the right one
/// index 1, and the gap / a corner returns `None` (background). Exercises
/// `Renderer::pick` — staging, the id pass and the one-texel read-back — end-to-end
/// on a real GPU, including the lazily allocated `PickTarget` it owns (#235 R4).
#[test]
#[ignore = "requires a GPU adapter"]
fn picking_resolves_object_ids_and_background() {
    let quad = Mesh::from_obj(QUAD_OBJ).expect("quad OBJ parses");
    // The main-pass format is irrelevant to picking (its pipeline is PICK_FORMAT).
    let mut renderer = single(wgpu::TextureFormat::Rgba8UnormSrgb, &quad);

    let (width, height) = (64u32, 64u32);

    // Two objects: a 0.35-scaled quad at NDC x = -0.5 (object 0) and +0.5
    // (object 1), leaving the center a background gap.
    let model = |x: f32| {
        Matrix4::from_glam(
            Mat4::from_translation(Vec3::new(x, 0.0, 0.0)) * Mat4::from_scale(Vec3::splat(0.35)),
        )
    };
    let mesh_id = initial_mesh_id(&renderer, 0);
    let draws = [
        ResolvedDraw {
            mesh_id,
            model: model(-0.5),
            selection: DrawSelection::INHERIT,
        },
        ResolvedDraw {
            mesh_id,
            model: model(0.5),
            selection: DrawSelection::INHERIT,
        },
    ];

    let mut pick = |x: u32, y: u32| {
        pollster::block_on(renderer.pick(
            camera_of(FrameParams::IDENTITY, width, height),
            &draws,
            x,
            y,
            Viewport { width, height },
        ))
        .expect("the draws are resident")
    };

    // NDC x=-0.5 → pixel ≈ 16; x=+0.5 → ≈ 48; center gap at 32; corner at (2,2).
    assert_eq!(pick(16, 32), Some(0), "left object → index 0");
    assert_eq!(pick(48, 32), Some(1), "right object → index 1");
    assert_eq!(pick(32, 32), None, "center gap → background");
    assert_eq!(pick(2, 2), None, "corner → background");
    // Out-of-bounds coordinates are safely rejected.
    assert_eq!(pick(width, 0), None, "x == width is out of bounds");

    let shadow_then_mesh = [
        ResolvedDraw {
            mesh_id,
            model: Matrix4::IDENTITY,
            selection: DrawSelection::Shadow,
        },
        ResolvedDraw {
            mesh_id,
            model: model(-0.5),
            selection: DrawSelection::INHERIT,
        },
    ];
    assert_eq!(
        pollster::block_on(renderer.pick(
            camera_of(FrameParams::IDENTITY, width, height),
            &shadow_then_mesh,
            16,
            32,
            Viewport { width, height },
        ))
        .expect("the mesh draw is resident"),
        Some(1),
        "skipping a shadow must preserve the original draw-list index"
    );
}

/// The pick target tracks the viewport: after a resize both of its attachments
/// are rebuilt at the new size and picking keeps resolving the same objects —
/// the path a window drag takes, and the one place `PickTarget`'s two
/// `ViewportAttachment`s are re-`ensure`d rather than first allocated (#363).
#[test]
#[ignore = "requires a GPU adapter"]
fn picking_tracks_a_resized_viewport() {
    let quad = Mesh::from_obj(QUAD_OBJ).expect("quad OBJ parses");
    let mut renderer = single(wgpu::TextureFormat::Rgba8UnormSrgb, &quad);

    // Same two objects as above: 0.35-scaled quads at NDC x = ∓0.5, center gap.
    let model = |x: f32| {
        Matrix4::from_glam(
            Mat4::from_translation(Vec3::new(x, 0.0, 0.0)) * Mat4::from_scale(Vec3::splat(0.35)),
        )
    };
    let mesh_id = initial_mesh_id(&renderer, 0);
    let draws = [
        ResolvedDraw {
            mesh_id,
            model: model(-0.5),
            selection: DrawSelection::INHERIT,
        },
        ResolvedDraw {
            mesh_id,
            model: model(0.5),
            selection: DrawSelection::INHERIT,
        },
    ];

    // Square viewports throughout, so a pixel keeps its NDC meaning across the
    // resize: x = ∓0.5 lands a quarter and three quarters across.
    let pick_at = |renderer: &mut Renderer, size: u32, x: u32, y: u32| {
        pollster::block_on(renderer.pick(
            camera_of(FrameParams::IDENTITY, size, size),
            &draws,
            x,
            y,
            Viewport {
                width: size,
                height: size,
            },
        ))
        .expect("the draws are resident")
    };

    assert_eq!(
        renderer.pick_target_size(),
        None,
        "nothing is allocated until the first pick"
    );
    assert_eq!(pick_at(&mut renderer, 64, 16, 32), Some(0));
    assert_eq!(renderer.pick_target_size(), Some((64, 64)));

    // Grown: id and depth are recreated together, the read-back buffer is not.
    assert_eq!(pick_at(&mut renderer, 128, 32, 64), Some(0), "left, grown");
    assert_eq!(pick_at(&mut renderer, 128, 96, 64), Some(1), "right, grown");
    assert_eq!(pick_at(&mut renderer, 128, 64, 64), None, "gap, grown");
    assert_eq!(renderer.pick_target_size(), Some((128, 128)));

    // Shrunk back, and the same clicks still resolve.
    assert_eq!(pick_at(&mut renderer, 64, 48, 32), Some(1), "right, shrunk");
    assert_eq!(renderer.pick_target_size(), Some((64, 64)));
}

/// `MeshTarget` decides *which* meshes an appearance edit reaches, and a
/// foreign identity must fail without writing another mesh.
#[test]
#[ignore = "requires a GPU adapter"]
fn mesh_target_selects_which_meshes_an_appearance_edit_reaches() {
    let meshes = [Mesh::hello_triangle(), Mesh::hello_triangle()];
    let mut renderer = Renderer::new(
        test_gpu(),
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &meshes,
        &[Matrix4::IDENTITY; 2],
    )
    .expect("two meshes with two base models is a valid mesh set");
    let first = initial_mesh_id(&renderer, 0);
    let second = initial_mesh_id(&renderer, 1);

    let red = crate::MeshAppearance {
        material: crate::DisneyMaterial {
            base_color: [1.0, 0.0, 0.0],
            ..Default::default()
        },
        ..Default::default()
    };
    renderer
        .set_appearance(crate::MeshTarget::All, red.clone())
        .expect("all meshes are resident");
    assert_eq!(
        renderer.mesh_appearance(first),
        Ok(&red),
        "All reaches mesh 0"
    );
    assert_eq!(
        renderer.mesh_appearance(second),
        Ok(&red),
        "All reaches mesh 1"
    );

    let blue = crate::DisneyMaterial {
        base_color: [0.0, 0.0, 1.0],
        ..Default::default()
    };
    renderer
        .set_disney_material(crate::MeshTarget::One(second), blue.clone())
        .expect("mesh 1 is resident");
    assert_eq!(
        renderer.mesh_appearance(first).map(|a| &a.material),
        Ok(&red.material),
        "One(1) leaves mesh 0 alone"
    );
    assert_eq!(
        renderer.mesh_appearance(second).map(|a| &a.material),
        Ok(&blue),
        "One(1) reaches mesh 1"
    );

    let foreign = MeshTable::new(vec![Mesh::hello_triangle()])
        .expect("registration succeeds")
        .id(MeshTableIndex::new(0))
        .expect("one registered mesh");
    let error = renderer
        .set_disney_material(
            crate::MeshTarget::One(foreign),
            crate::DisneyMaterial {
                base_color: [0.0, 1.0, 0.0],
                ..Default::default()
            },
        )
        .expect_err("a foreign identity is not resident");
    assert_eq!(error, MeshResourceError::NotResident { mesh: foreign });
    assert_eq!(
        renderer.mesh_appearance(first).map(|a| &a.material),
        Ok(&red.material)
    );
    assert_eq!(
        renderer.mesh_appearance(second).map(|a| &a.material),
        Ok(&blue)
    );
    assert_eq!(
        renderer.mesh_appearance(foreign),
        Err(MeshResourceError::NotResident { mesh: foreign })
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn explicit_mesh_table_uploads_preserve_the_registered_identities() {
    let table = MeshTable::new(vec![Mesh::hello_triangle()]).expect("registration succeeds");
    let registered = table
        .id(MeshTableIndex::new(0))
        .expect("one registered mesh");
    let explicit = Renderer::with_mesh_table(
        test_gpu(),
        TEXTURE_TARGET_FORMAT,
        table.clone(),
        &[Matrix4::IDENTITY],
        1,
    )
    .expect("the explicit table uploads");
    let auto_fit = Renderer::auto_fit_table(test_gpu(), TEXTURE_TARGET_FORMAT, table.clone())
        .expect("the auto-fit table uploads");

    for renderer in [&explicit, &auto_fit] {
        assert_eq!(
            renderer.mesh_table().ids().collect::<Vec<_>>(),
            table.ids().collect::<Vec<_>>()
        );
        assert!(
            renderer.mesh_appearance(registered).is_ok(),
            "the registered identity must be resident after upload"
        );
    }
}

/// A mesh added at runtime is drawable, and adding it leaves the meshes already
/// uploaded exactly as they were (#353).
///
/// The interesting half is the PBR slot array. A slot is selected by a **dynamic
/// offset validated against the slot buffer**, so before `add_mesh` grew that
/// buffer, binding slot 1 of a one-slot array was a wgpu error rather than a
/// mis-render — which is why both draws below are `Shaded`: a `Filled` draw
/// never binds the group and would pass either way.
#[test]
#[ignore = "requires a GPU adapter"]
fn add_mesh_grows_the_pbr_slots_and_keeps_existing_appearance() {
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (64, 64);
    let mut renderer = single(format, &Mesh::hello_triangle());
    let initial = initial_mesh_id(&renderer, 0);

    // Mesh 0 gets a distinctive appearance *before* the add, so the assertion
    // afterwards is that reallocating the slot buffer did not lose it.
    let red = crate::MeshAppearance {
        material: crate::DisneyMaterial {
            base_color: [1.0, 0.0, 0.0],
            ..Default::default()
        },
        ..Default::default()
    };
    renderer
        .set_appearance(crate::MeshTarget::One(initial), red.clone())
        .expect("the initial mesh is resident");
    assert_eq!(renderer.mesh_count(), 1, "one mesh to start");

    let one: Scene = [DrawableObject::mesh(
        initial,
        Matrix4::from_translation(Vector3::new(-0.4, 0.0, 0.0)),
        RenderMode::Shaded,
    )]
    .into_iter()
    .collect();
    let before = render_with_readback(&gpu, format, width, height, |e, v| {
        renderer
            .encode(e, v, camera_of(FrameParams::IDENTITY, width, height), &one)
            .expect("the scene is resident");
    });

    let added = renderer
        .add_mesh(&Mesh::hello_triangle())
        .expect("the mesh uploads");
    assert_ne!(added, initial, "runtime uploads mint fresh identities");
    assert_eq!(
        renderer.mesh_count(),
        2,
        "and the store grew by exactly one"
    );
    assert_eq!(
        renderer.mesh_appearance(initial),
        Ok(&red),
        "the existing mesh keeps its appearance across the slot reallocation"
    );
    assert_eq!(
        renderer.mesh_appearance(added),
        Ok(&crate::MeshAppearance::default()),
        "and the new mesh starts from the default"
    );

    // Drawing mesh 1 binds slot 1: the frame renders, and it differs from the
    // one-mesh frame because the second triangle is there.
    let two: Scene = [
        DrawableObject::mesh(
            initial,
            Matrix4::from_translation(Vector3::new(-0.4, 0.0, 0.0)),
            RenderMode::Shaded,
        ),
        DrawableObject::mesh(
            added,
            Matrix4::from_translation(Vector3::new(0.4, 0.0, 0.0)),
            RenderMode::Shaded,
        ),
    ]
    .into_iter()
    .collect();
    let after = render_with_readback(&gpu, format, width, height, |e, v| {
        renderer
            .encode(e, v, camera_of(FrameParams::IDENTITY, width, height), &two)
            .expect("the scene is resident");
    });

    assert_ne!(
        before, after,
        "the mesh added at runtime must reach the frame"
    );
    let lit_in = |px: &[u8], xs: std::ops::Range<u32>| {
        xs.into_iter().any(|x| {
            (0..height).any(|y| {
                let i = ((y * width + x) * 4) as usize;
                px[i] > 0 || px[i + 1] > 0 || px[i + 2] > 0
            })
        })
    };
    assert!(
        !lit_in(&before, 2 * width / 3..width),
        "nothing was drawn on the right before the add"
    );
    assert!(
        lit_in(&after, 2 * width / 3..width),
        "the added mesh draws on the right"
    );
}

/// A removed mesh frees its slot, keeps every other identity valid, and the
/// next upload reuses only the private slot, never the public identity (#353).
///
/// Compacting the store instead would renumber meshes after the hole, silently
/// repointing any scene that holds an id.
#[test]
#[ignore = "requires a GPU adapter"]
fn remove_mesh_frees_the_slot_without_renumbering_the_survivors() {
    let meshes = [
        Mesh::hello_triangle(),
        Mesh::hello_triangle(),
        Mesh::hello_triangle(),
    ];
    let mut renderer = Renderer::new(
        test_gpu(),
        wgpu::TextureFormat::Rgba8UnormSrgb,
        &meshes,
        &[Matrix4::IDENTITY; 3],
    )
    .expect("three meshes with three base models is a valid mesh set");
    let removed = initial_mesh_id(&renderer, 0);
    let middle = initial_mesh_id(&renderer, 1);
    let survivor = initial_mesh_id(&renderer, 2);

    let blue = crate::DisneyMaterial {
        base_color: [0.0, 0.0, 1.0],
        ..Default::default()
    };
    renderer
        .set_disney_material(crate::MeshTarget::One(survivor), blue.clone())
        .expect("the survivor is resident");

    renderer
        .remove_mesh(removed)
        .expect("the initial mesh is resident");
    assert_eq!(
        renderer.remove_mesh(removed),
        Err(MeshResourceError::NotResident { mesh: removed }),
        "repeated removal is explicit"
    );
    assert_eq!(
        renderer.mesh_appearance(removed),
        Err(MeshResourceError::NotResident { mesh: removed }),
        "the removed identity is stale"
    );
    assert_eq!(
        renderer.mesh_appearance(survivor).map(|a| &a.material),
        Ok(&blue),
        "the survivor keeps its identity and material"
    );
    assert_eq!(renderer.mesh_count(), 2, "only live resources are counted");
    assert_eq!(
        renderer.meshes.slot_count(),
        3,
        "deletion does not collapse the private allocation span"
    );

    // The next upload reuses the hole rather than growing past it.
    let replacement = renderer
        .add_mesh(&Mesh::hello_triangle())
        .expect("the replacement uploads");
    assert_ne!(
        replacement, removed,
        "a recycled slot gets a fresh identity"
    );
    assert_eq!(renderer.mesh_count(), 3, "the live count returns to three");
    assert_eq!(
        renderer.meshes.slot_count(),
        3,
        "the private span stays flat"
    );
    assert_eq!(
        renderer.mesh_appearance(replacement),
        Ok(&crate::MeshAppearance::default()),
        "and the reused slot starts clean rather than inheriting the old mesh"
    );
    let scene: Scene = [
        DrawableObject::mesh(
            replacement,
            Matrix4::from_translation(Vector3::new(-0.4, 0.0, 0.0)),
            RenderMode::Shaded,
        ),
        DrawableObject::mesh(
            survivor,
            Matrix4::from_translation(Vector3::new(0.4, 0.0, 0.0)),
            RenderMode::Shaded,
        ),
    ]
    .into_iter()
    .collect();
    let pixels = render_with_readback(
        &test_gpu(),
        TEXTURE_TARGET_FORMAT,
        64,
        64,
        |encoder, view| {
            renderer
                .encode(
                    encoder,
                    view,
                    camera_of(FrameParams::IDENTITY, 64, 64),
                    &scene,
                )
                .expect("the replacement and survivor are resident");
        },
    );
    let side_is_lit = |xs: std::ops::Range<u32>| {
        xs.into_iter().any(|x| {
            (0..64).any(|y| {
                let offset = ((y * 64 + x) * 4) as usize;
                pixels[offset..offset + 3] != [0, 0, 0]
            })
        })
    };
    assert!(side_is_lit(0..64 / 3), "the replacement must draw");
    assert!(side_is_lit(2 * 64 / 3..64), "the survivor must still draw");

    renderer
        .remove_mesh(middle)
        .expect("the middle mesh is resident");
    assert_eq!(renderer.mesh_count(), 2);
    assert_eq!(renderer.meshes.slot_count(), 3);
}

fn assert_not_resident(error: RenderError, mesh: MeshId) {
    assert!(
        matches!(
            &error,
            RenderError::MeshResource(MeshResourceError::NotResident { mesh: actual })
                if *actual == mesh
        ),
        "expected NotResident for {mesh:?}, got {error:?}"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn stale_and_foreign_mesh_ids_are_rejected_by_scene_pick_setters_and_delete() {
    let meshes = [Mesh::hello_triangle(), Mesh::hello_triangle()];
    let mut renderer = Renderer::new(
        test_gpu(),
        TEXTURE_TARGET_FORMAT,
        &meshes,
        &[Matrix4::IDENTITY; 2],
    )
    .expect("two meshes build");
    let stale = initial_mesh_id(&renderer, 0);
    let survivor = initial_mesh_id(&renderer, 1);
    renderer.remove_mesh(stale).expect("the mesh is resident");

    let foreign = MeshTable::new(vec![Mesh::hello_triangle()])
        .expect("registration succeeds")
        .id(MeshTableIndex::new(0))
        .expect("one registered mesh");
    let target = renderer
        .create_texture_target(32, 32)
        .expect("target builds");
    let camera = camera_of(FrameParams::IDENTITY, 32, 32);
    let texture = crate::ImageTexture::from_rgba(1, 1, vec![255, 255, 255, 255])
        .expect("one texel is a texture");

    for invalid in [stale, foreign] {
        for object in [
            DrawableObject::mesh(invalid, Matrix4::IDENTITY, RenderMode::Filled),
            DrawableObject::aabb_box(invalid, Matrix4::IDENTITY),
        ] {
            let scene: Scene = [object].into_iter().collect();
            let error = renderer
                .draw_layers(&[SceneLayer::new(camera, &scene)], &target)
                .expect_err("a nonresident scene must fail");
            assert_not_resident(error, invalid);
        }

        let draws = [ResolvedDraw {
            mesh_id: invalid,
            model: Matrix4::IDENTITY,
            selection: DrawSelection::INHERIT,
        }];
        let error = pollster::block_on(renderer.pick(
            camera,
            &draws,
            16,
            16,
            Viewport {
                width: 32,
                height: 32,
            },
        ))
        .expect_err("a nonresident pick must fail");
        assert_not_resident(error, invalid);

        assert_eq!(
            renderer.set_mesh_texture(invalid, &texture),
            Err(MeshResourceError::NotResident { mesh: invalid })
        );
        assert_eq!(
            renderer.set_disney_material(
                crate::MeshTarget::One(invalid),
                crate::DisneyMaterial::default(),
            ),
            Err(MeshResourceError::NotResident { mesh: invalid })
        );
        assert_eq!(
            renderer.remove_mesh(invalid),
            Err(MeshResourceError::NotResident { mesh: invalid })
        );
    }

    assert_eq!(renderer.mesh_count(), 1);
    assert!(renderer.mesh_appearance(survivor).is_ok());
}

#[test]
#[ignore = "requires a GPU adapter"]
fn set_texture_keeps_targeting_the_stale_initial_row_zero_after_replacement() {
    let mut renderer = single(TEXTURE_TARGET_FORMAT, &Mesh::hello_triangle());
    let original = initial_mesh_id(&renderer, 0);
    renderer
        .remove_mesh(original)
        .expect("the initial mesh is resident");
    let replacement = renderer
        .add_mesh(&Mesh::hello_triangle())
        .expect("the replacement uploads");
    let texture = crate::ImageTexture::from_rgba(1, 1, vec![255, 255, 255, 255])
        .expect("one texel is a texture");

    assert_eq!(
        renderer.set_texture(&texture),
        Err(MeshResourceError::NotResident { mesh: original }),
        "the convenience setter must not retarget the replacement in slot zero"
    );
    renderer
        .set_mesh_texture(replacement, &texture)
        .expect("the replacement remains explicitly addressable");
}

#[test]
#[ignore = "requires a GPU adapter"]
fn repeated_hole_reuse_keeps_live_count_flat_and_uploads_each_new_pbr_appearance() {
    let gpu = test_gpu();
    let format = TEXTURE_TARGET_FORMAT;
    let (width, height) = (48, 48);
    let mut renderer = single(format, &Mesh::hello_triangle());
    let mut current = initial_mesh_id(&renderer, 0);
    let mut previous_pixels: Option<Vec<u8>> = None;

    for color in [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]] {
        renderer
            .remove_mesh(current)
            .expect("the current occupant is resident");
        let stale = current;
        current = renderer
            .add_mesh(&Mesh::hello_triangle())
            .expect("the hole accepts a replacement");
        assert_ne!(current, stale);
        assert_eq!(renderer.mesh_count(), 1, "only the replacement is live");
        assert_eq!(
            renderer.meshes.slot_count(),
            1,
            "reusing the hole must not grow the private allocation span"
        );
        assert_eq!(
            renderer.mesh_appearance(stale),
            Err(MeshResourceError::NotResident { mesh: stale })
        );

        let material = crate::DisneyMaterial {
            base_color: color,
            metallic: 0.0,
            roughness: 0.5,
            ..Default::default()
        };
        renderer
            .set_disney_material(crate::MeshTarget::One(current), material.clone())
            .expect("the replacement is resident");
        assert_eq!(
            renderer.mesh_appearance(current).map(|a| &a.material),
            Ok(&material)
        );

        let scene: Scene = [DrawableObject::mesh(
            current,
            Matrix4::IDENTITY,
            RenderMode::Shaded,
        )]
        .into_iter()
        .collect();
        let pixels = render_with_readback(&gpu, format, width, height, |encoder, view| {
            renderer
                .encode(
                    encoder,
                    view,
                    camera_of(FrameParams::IDENTITY, width, height),
                    &scene,
                )
                .expect("the replacement scene is resident");
        });
        if let Some(previous) = &previous_pixels {
            assert_ne!(
                previous.as_slice(),
                pixels.as_slice(),
                "a replacement's changed PBR appearance must reach its reused slot"
            );
        }
        previous_pixels = Some(pixels);
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn wireframe_overlay_order_follows_reused_private_slots_not_identity_age() {
    let colored = |color: [f32; 3]| {
        let mut mesh = Mesh::hello_triangle();
        for vertex in &mut mesh.vertices {
            vertex.color = color;
        }
        mesh
    };
    let discarded = colored([1.0, 0.0, 0.0]);
    let green = colored([0.0, 1.0, 0.0]);
    let blue = colored([0.0, 0.0, 1.0]);
    let gpu = test_gpu();
    let format = TEXTURE_TARGET_FORMAT;
    let (width, height) = (64, 64);
    let mut renderer = Renderer::auto_fit(gpu.clone(), format, &[discarded, green])
        .expect("two meshes build with the same preview transform policy as runtime additions");
    let first_slot = initial_mesh_id(&renderer, 0);
    let later_slot = initial_mesh_id(&renderer, 1);
    renderer
        .remove_mesh(first_slot)
        .expect("the first slot is occupied");
    let replacement = renderer.add_mesh(&blue).expect("the first slot is reused");
    assert!(
        replacement > later_slot,
        "the replacement must be newer than the surviving identity"
    );

    let render = |renderer: &mut Renderer, ids: &[MeshId]| {
        let scene: Scene = ids
            .iter()
            .copied()
            .map(|id| DrawableObject::mesh(id, Matrix4::IDENTITY, RenderMode::Wireframe))
            .collect();
        render_with_readback(&gpu, format, width, height, |encoder, view| {
            renderer
                .encode(
                    encoder,
                    view,
                    camera_of(FrameParams::IDENTITY, width, height),
                    &scene,
                )
                .expect("the wireframe scene is resident");
        })
    };
    let green_only = render(&mut renderer, &[later_slot]);
    let blue_only = render(&mut renderer, &[replacement]);
    assert_ne!(
        green_only, blue_only,
        "the two overlays need distinct colors"
    );

    let combined = render(&mut renderer, &[later_slot, replacement]);
    assert_eq!(
        combined, green_only,
        "slot 0's newer replacement must draw before slot 1's older identity"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn draw_layers_validates_the_whole_list_before_any_layer_writes() {
    let meshes = [Mesh::hello_triangle(), Mesh::hello_triangle()];
    let mut renderer = Renderer::new(
        test_gpu(),
        TEXTURE_TARGET_FORMAT,
        &meshes,
        &[Matrix4::IDENTITY; 2],
    )
    .expect("two meshes build");
    let stale = initial_mesh_id(&renderer, 0);
    let valid = initial_mesh_id(&renderer, 1);
    renderer
        .remove_mesh(stale)
        .expect("the first mesh is resident");

    let (width, height) = (48, 48);
    let target = renderer
        .create_texture_target(width, height)
        .expect("target builds");
    let camera = camera_of(FrameParams::IDENTITY, width, height);
    let empty = Scene::new();
    let valid_scene: Scene = [DrawableObject::mesh(
        valid,
        Matrix4::IDENTITY,
        RenderMode::Filled,
    )]
    .into_iter()
    .collect();
    let invalid_scene: Scene = [DrawableObject::mesh(
        stale,
        Matrix4::IDENTITY,
        RenderMode::Filled,
    )]
    .into_iter()
    .collect();

    let baseline =
        pollster::block_on(renderer.render_layers(&[SceneLayer::new(camera, &empty)], &target))
            .expect("the empty scene clears the target");
    let would_write = pollster::block_on(
        renderer.render_layers(&[SceneLayer::new(camera, &valid_scene)], &target),
    )
    .expect("the valid scene renders");
    assert_ne!(
        would_write, baseline,
        "the first layer would change the target"
    );
    let baseline =
        pollster::block_on(renderer.render_layers(&[SceneLayer::new(camera, &empty)], &target))
            .expect("the target resets to the baseline");

    let error = renderer
        .draw_layers(
            &[
                SceneLayer::new(camera, &valid_scene),
                SceneLayer::new(camera, &invalid_scene),
            ],
            &target,
        )
        .expect_err("the later invalid layer rejects the whole list");
    assert_not_resident(error, stale);
    let after = pollster::block_on(renderer.read_pixels(&target)).expect("target reads back");
    assert_eq!(
        after, baseline,
        "validation must happen before the first layer clears or writes"
    );
}

/// Removing a mesh releases its GPU memory **at the call**, not at the next
/// frame (#353).
///
/// Dropping a wgpu resource does not free it: wgpu reclaims while servicing a
/// submission, so before `remove_mesh` flushed, a deleted mesh kept its memory
/// until something else rendered — ~445 MiB for a real GLB, which is what
/// "delete freed nothing" looked like from the outside. The allocator report is
/// the only honest witness here: `nvidia-smi` reports per-process memory as
/// `[N/A]` on WDDM.
#[test]
#[ignore = "requires a GPU adapter"]
fn removing_a_mesh_releases_its_memory_immediately() {
    // A **private** device, not the shared `test_gpu()`: the allocator report is
    // device-global, and the rest of the suite runs concurrently on that one, so
    // a neighbour's allocation would perturb the totals and — worse — a
    // neighbour's submission could service this mesh's release, letting the test
    // pass with the flush removed.
    let gpu = pollster::block_on(create_test_device());
    let backend = gpu.adapter_facts().backend;
    assert!(
        gpu.device.generate_allocator_report().is_some() || backend == "Gl",
        "the {backend} backend should report allocations; without a report this \
         test cannot pin anything and must not quietly pass"
    );
    if gpu.device.generate_allocator_report().is_none() {
        // GL keeps no allocator of its own to report on.
        return;
    }
    let mut renderer = Renderer::new(
        gpu.clone(),
        wgpu::TextureFormat::Rgba8UnormSrgb,
        std::slice::from_ref(&Mesh::hello_triangle()),
        &[Matrix4::IDENTITY],
    )
    .expect("one mesh with one base model is a valid mesh set");

    // ~64k vertices and a 512² map: small enough to stay quick, far larger than
    // the allocator's own noise.
    let big = Mesh {
        vertices: (0..64_000)
            .map(|i| {
                let f = i as f32 * 0.001;
                Vertex {
                    position: [f.sin(), f.cos(), 0.0],
                    color: [1.0, 1.0, 1.0],
                    uv: [0.0, 0.0],
                }
            })
            .collect(),
        indices: (0..64_000).collect(),
        shading: None,
    };
    let texture = crate::ImageTexture::from_rgba(512, 512, vec![200u8; 512 * 512 * 4])
        .expect("a 512² texture builds");

    let live = |gpu: &GpuContext| {
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        gpu.device
            .generate_allocator_report()
            .map_or(0, |r| r.total_allocated_bytes)
    };
    let baseline = live(&gpu);

    let id = renderer.add_mesh(&big).expect("the mesh uploads");
    renderer
        .set_mesh_texture(id, &texture)
        .expect("the albedo uploads");
    renderer
        .set_mesh_metallic_roughness_texture(id, &texture)
        .expect("the metallic-roughness map uploads");
    renderer
        .set_mesh_normal_texture(id, &texture)
        .expect("the normal map uploads");
    let loaded = live(&gpu);
    assert!(
        loaded > baseline,
        "the mesh and its maps must show up as allocated ({baseline} -> {loaded} bytes)"
    );

    renderer.remove_mesh(id).expect("the mesh is resident");
    let freed = live(&gpu);
    assert!(
        freed < loaded,
        "removing must release, not defer: still {freed} bytes allocated of {loaded}"
    );
    // Back to roughly the baseline: allow slack for the grown slot array, which
    // is a high-water mark by design.
    let slack = (loaded - baseline) / 10;
    assert!(
        freed <= baseline + slack,
        "removing must return almost everything: {freed} bytes vs a {baseline}-byte baseline"
    );
}
