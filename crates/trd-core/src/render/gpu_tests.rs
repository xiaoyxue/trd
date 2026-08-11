//! GPU-gated render tests (`#[ignore]`, native only) plus their shared
//! readback/device helpers and OBJ fixtures.

use super::*;
use crate::math::{Matrix4, Point3, Vector3};
use glam::{Mat4, Vec3};

/// Test convenience constructor: a [`SceneRenderer`] over a single mesh with an
/// identity base model (vertices drawn in their own coordinates). Replaces the
/// former single-mesh `SceneRenderer::new`/`with_base_model` production helpers,
/// which only the GPU tests used.
#[cfg(not(target_arch = "wasm32"))]
fn single(format: wgpu::TextureFormat, mesh: &Mesh) -> SceneRenderer {
    SceneRenderer::new(
        test_gpu(),
        format,
        std::slice::from_ref(mesh),
        &[Matrix4::IDENTITY],
    )
}

#[cfg(not(target_arch = "wasm32"))]
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
        crate::tightly_pack_rgba(&mapped, width, height, padded_bytes_per_row)
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
#[cfg(not(target_arch = "wasm32"))]
fn test_gpu() -> std::sync::Arc<GpuContext> {
    static SHARED: std::sync::OnceLock<std::sync::Arc<GpuContext>> = std::sync::OnceLock::new();
    SHARED
        .get_or_init(|| pollster::block_on(create_test_device()))
        .clone()
}

#[cfg(not(target_arch = "wasm32"))]
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

/// `RenderTarget` must report the format a renderer's pipelines are built for.
///
/// For [`OffscreenTarget`] that is [`OFFSCREEN_FORMAT`]; the on-screen case is
/// the interesting one and is covered by `OnscreenTarget`'s own sRGB-view logic
/// (a surface's preferred format is often *non*-sRGB, so the target renders
/// through an sRGB view to stay byte-identical with the headless CLI). Pinning
/// the offscreen side here keeps the trait honest about which of the two a
/// caller gets.
#[test]
#[ignore = "requires a GPU adapter"]
#[cfg(not(target_arch = "wasm32"))]
fn offscreen_target_reports_its_render_format_and_size() {
    let gpu = test_gpu();
    let (width, height) = (32, 24);
    let mut target = OffscreenTarget::new(&gpu.device, width, height).expect("offscreen target");

    assert_eq!(target.view_format(), OFFSCREEN_FORMAT);
    assert_eq!(target.viewport(), Viewport { width, height });

    RenderTarget::resize(&mut target, &gpu, 64, 48).expect("resize");
    assert_eq!(
        target.viewport(),
        Viewport {
            width: 64,
            height: 48
        }
    );
    assert_eq!(target.view_format(), OFFSCREEN_FORMAT);
}

/// [`Renderer::with_gpu`] must produce the same frame as [`Renderer::with_meshes`].
///
/// The streaming browser front-end owns its device before it owns its meshes, so
/// it builds the harness on an existing [`GpuContext`] instead of letting the
/// harness request one (#180). That second constructor is only safe if it is a
/// pure "bring your own device" variant — same auto-fit preview transforms, same
/// offscreen target, same pixels — so this renders one scene through both and
/// compares them byte for byte.
#[test]
#[ignore = "requires a GPU adapter"]
#[cfg(not(target_arch = "wasm32"))]
fn with_gpu_renders_identically_to_with_meshes() {
    use crate::Renderer;

    let (width, height) = (64, 64);
    let meshes = [Mesh::hello_triangle()];
    let scene = vec![DrawableObject::Mesh {
        mesh_id: 0,
        model: Matrix4::IDENTITY.to_cols_array(),
        mode: RenderMode::Filled,
    }];

    let mut owned = pollster::block_on(Renderer::with_meshes(width, height, &meshes))
        .expect("the harness requests its own device");
    let expected =
        pollster::block_on(owned.render_scene(FrameParams::IDENTITY, &scene)).expect("renders");

    let mut borrowed = Renderer::with_gpu(test_gpu(), width, height, &meshes)
        .expect("the harness accepts an existing device");
    let actual =
        pollster::block_on(borrowed.render_scene(FrameParams::IDENTITY, &scene)).expect("renders");

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
#[test]
#[ignore = "requires a GPU adapter"]
#[cfg(not(target_arch = "wasm32"))]
fn first_frame_renders_with_no_setters_called() {
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (48, 48);
    let mut renderer = single(format, &Mesh::hello_triangle());

    // Every draw kind that reads a lazily-bound resource: Textured samples the
    // albedo, Pbr samples albedo + material maps + the environment probe.
    for mode in [RenderMode::Textured, RenderMode::Pbr] {
        let scene = vec![DrawableObject::Mesh {
            mesh_id: 0,
            model: Matrix4::IDENTITY.to_cols_array(),
            mode,
        }];
        let pixels = render_with_readback(&gpu, format, width, height, |encoder, view| {
            renderer.encode(
                encoder,
                view,
                FrameParams::IDENTITY,
                &scene,
                Viewport { width, height },
            );
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
#[cfg(not(target_arch = "wasm32"))]
fn mesh_renderer_draws_multiple_instances() {
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (64, 64);
    let mut mesh = single(format, &Mesh::hello_triangle());

    // One centered instance vs. two instances translated to opposite sides.
    let single = [DrawableObject::Mesh {
        mesh_id: 0,
        model: Matrix4::IDENTITY.to_cols_array(),
        mode: RenderMode::Filled,
    }];
    let single_px = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            FrameParams::IDENTITY,
            &single,
            Viewport { width, height },
        );
    });

    let two = [
        DrawableObject::Mesh {
            mesh_id: 0,
            model: Matrix4::from_translation(Vector3::new(-0.4, 0.0, 0.0)).to_cols_array(),
            mode: RenderMode::Filled,
        },
        DrawableObject::Mesh {
            mesh_id: 0,
            model: Matrix4::from_translation(Vector3::new(0.4, 0.0, 0.0)).to_cols_array(),
            mode: RenderMode::Filled,
        },
    ];
    let two_px = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            FrameParams::IDENTITY,
            &two,
            Viewport { width, height },
        );
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
#[cfg(not(target_arch = "wasm32"))]
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
    let mut mesh = SceneRenderer::new(
        test_gpu(),
        format,
        &[quad([1.0, 0.0, 0.0]), quad([0.0, 1.0, 0.0])],
        &[Matrix4::IDENTITY, Matrix4::IDENTITY],
    );
    let scene = [
        DrawableObject::Mesh {
            mesh_id: 0,
            model: Matrix4::from_translation(Vector3::new(0.0, 0.0, 0.25)).to_cols_array(),
            mode: RenderMode::Filled,
        },
        DrawableObject::Mesh {
            mesh_id: 1,
            model: Matrix4::from_translation(Vector3::new(0.0, 0.0, 0.75)).to_cols_array(),
            mode: RenderMode::Filled,
        },
    ];
    let px = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            FrameParams::IDENTITY,
            &scene,
            Viewport { width, height },
        );
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
#[cfg(not(target_arch = "wasm32"))]
fn mesh_renderer_wireframe_lights_edges_only() {
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (64, 64);
    let mut mesh = single(format, &Mesh::hello_triangle());
    let model = Matrix4::IDENTITY.to_cols_array();
    let filled_scene = [DrawableObject::Mesh {
        mesh_id: 0,
        model,
        mode: RenderMode::Filled,
    }];
    let wire_scene = [DrawableObject::Mesh {
        mesh_id: 0,
        model,
        mode: RenderMode::Wireframe,
    }];

    let filled = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            FrameParams::IDENTITY,
            &filled_scene,
            Viewport { width, height },
        );
    });

    let wire = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            FrameParams::IDENTITY,
            &wire_scene,
            Viewport { width, height },
        );
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
#[cfg(not(target_arch = "wasm32"))]
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
    mesh.set_texture(&checker);
    let scene = [DrawableObject::Mesh {
        mesh_id: 0,
        model: Matrix4::IDENTITY.to_cols_array(),
        mode: RenderMode::Textured,
    }];
    let pixels = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            FrameParams::IDENTITY,
            &scene,
            Viewport { width, height },
        );
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
#[cfg(not(target_arch = "wasm32"))]
fn mesh_renderer_aabb_overlay_draws_green_box() {
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (64, 64);
    let mut mesh = single(format, &Mesh::hello_triangle());
    let model = Matrix4::IDENTITY.to_cols_array();
    let plain_scene = [DrawableObject::Mesh {
        mesh_id: 0,
        model,
        mode: RenderMode::Filled,
    }];
    let box_scene = [
        DrawableObject::Mesh {
            mesh_id: 0,
            model,
            mode: RenderMode::Filled,
        },
        DrawableObject::AabbBox { mesh_id: 0, model },
    ];

    let plain = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            FrameParams::IDENTITY,
            &plain_scene,
            Viewport { width, height },
        );
    });

    let with_box = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            FrameParams::IDENTITY,
            &box_scene,
            Viewport { width, height },
        );
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
#[cfg(not(target_arch = "wasm32"))]
fn mesh_renderer_axes_overlay_draws_rgb_gizmo() {
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (64, 64);
    let mut mesh = single(format, &Mesh::hello_triangle());
    let model = Matrix4::IDENTITY.to_cols_array();
    let plain_scene = [DrawableObject::Mesh {
        mesh_id: 0,
        model,
        mode: RenderMode::Filled,
    }];
    let axes_scene = [
        DrawableObject::Mesh {
            mesh_id: 0,
            model,
            mode: RenderMode::Filled,
        },
        DrawableObject::CoordinateAxes {
            model: Matrix4::IDENTITY.to_cols_array(),
        },
    ];

    let plain = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            FrameParams::IDENTITY,
            &plain_scene,
            Viewport { width, height },
        );
    });

    let with_axes = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            FrameParams::IDENTITY,
            &axes_scene,
            Viewport { width, height },
        );
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
#[cfg(not(target_arch = "wasm32"))]
fn gizmo_lines_and_arrowheads_stay_smooth_without_msaa() {
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (128, 128);
    let mesh_data = Mesh::hello_triangle();
    let mut renderer = SceneRenderer::with_sample_count(
        test_gpu(),
        format,
        std::slice::from_ref(&mesh_data),
        &[Matrix4::IDENTITY],
        1,
    );
    let model = Matrix4::from_scale(Vector3::new(0.5, 0.5, 0.5)).to_cols_array();
    let scene = [DrawableObject::CoordinateAxes { model }];

    let pixels = render_with_readback(&gpu, format, width, height, |e, v| {
        renderer.encode(
            e,
            v,
            FrameParams::IDENTITY,
            &scene,
            Viewport { width, height },
        );
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
#[cfg(not(target_arch = "wasm32"))]
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
    let model = Matrix4::IDENTITY.to_cols_array();

    let filled_only = [DrawableObject::Mesh {
        mesh_id: 0,
        model,
        mode: RenderMode::Filled,
    }];
    let composed = [
        DrawableObject::Mesh {
            mesh_id: 0,
            model,
            mode: RenderMode::Filled,
        },
        DrawableObject::Mesh {
            mesh_id: 0,
            model,
            mode: RenderMode::Wireframe,
        },
        DrawableObject::AabbBox { mesh_id: 0, model },
        DrawableObject::CoordinateAxes {
            model: Matrix4::IDENTITY.to_cols_array(),
        },
    ];

    let filled_px = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            FrameParams::IDENTITY,
            &filled_only,
            Viewport { width, height },
        );
    });
    let composed_px = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            FrameParams::IDENTITY,
            &composed,
            Viewport { width, height },
        );
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
#[cfg(not(target_arch = "wasm32"))]
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
#[cfg(not(target_arch = "wasm32"))]
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
    let scene = [DrawableObject::EnvironmentBackground {
        rotation: 0.0,
        exposure: 1.0,
        blur: 0.0,
        tonemap: Tonemap::Reinhard,
    }];
    let pixels = render_with_readback(&gpu, format, width, height, |encoder, view| {
        renderer.encode(
            encoder,
            view,
            FrameParams::IDENTITY,
            &scene,
            Viewport { width, height },
        );
    });
    assert!(
        pixels.chunks_exact(4).any(|pixel| pixel[0] > 128),
        "environment background should produce visible probe color"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
#[cfg(not(target_arch = "wasm32"))]
fn mesh_renderer_renders_loaded_quad_filled_with_correct_coverage() {
    // #37/#41: a mesh loaded from OBJ (not the baked triangle) renders filled
    // via `draw_indexed` as a `DrawableObject::Mesh`. Under the identity camera
    // the unit quad spans NDC [-0.5, 0.5]², i.e. the central quarter of the
    // frame — so the center is lit, the corners are dark, and coverage ≈ 25%.
    let gpu = test_gpu();
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (width, height) = (64, 64);
    let quad = Mesh::from_obj(QUAD_OBJ).expect("quad OBJ parses");
    let mut mesh = single(format, &quad);

    let scene = [DrawableObject::Mesh {
        mesh_id: 0,
        model: Matrix4::IDENTITY.to_cols_array(),
        mode: RenderMode::Filled,
    }];
    let px = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            FrameParams::IDENTITY,
            &scene,
            Viewport { width, height },
        );
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
#[cfg(not(target_arch = "wasm32"))]
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

    let scene = [DrawableObject::Mesh {
        mesh_id: 0,
        model: Matrix4::IDENTITY.to_cols_array(),
        mode: RenderMode::Filled,
    }];

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
        mesh.encode(e, v, cg, &scene, viewport);
    });
    let cv_px = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(e, v, cv, &scene, viewport);
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
#[cfg(not(target_arch = "wasm32"))]
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
            let scene = [DrawableObject::Mesh {
                mesh_id: 0,
                model: Mat4::from_rotation_y(theta).to_cols_array(),
                mode: RenderMode::Wireframe,
            }];

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
                mesh.encode(e, v, cg, &scene, viewport);
            });
            let cv_px = render_with_readback(&gpu, format, width, height, |e, v| {
                mesh.encode(e, v, cv, &scene, viewport);
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
#[cfg(not(target_arch = "wasm32"))]
fn frame_plane_composites_background_under_scene() {
    // A FramePlane fills the background from a reused 2×2 texture (upload +
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
    let plane_only = [DrawableObject::FramePlane {
        fit: FrameFit::Stretch,
    }];
    let bg = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            FrameParams::IDENTITY,
            &plane_only,
            Viewport { width, height },
        );
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

    // (b) Plane + solid green mesh (built first in the scene, drawn on top):
    // the whole frame is green, proving the mesh composites over the plane.
    let composited = [
        DrawableObject::FramePlane {
            fit: FrameFit::Stretch,
        },
        DrawableObject::Mesh {
            mesh_id: 0,
            model: Matrix4::IDENTITY.to_cols_array(),
            mode: RenderMode::Filled,
        },
    ];
    let over = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            FrameParams::IDENTITY,
            &composited,
            Viewport { width, height },
        );
    });
    let foreground = [DrawableObject::Mesh {
        mesh_id: 0,
        model: Matrix4::IDENTITY.to_cols_array(),
        mode: RenderMode::Filled,
    }];
    let two_pass = render_with_readback(&gpu, format, width, height, |e, v| {
        mesh.encode(
            e,
            v,
            FrameParams::IDENTITY,
            &plane_only,
            Viewport { width, height },
        );
        mesh.encode_overlay(
            e,
            v,
            FrameParams::IDENTITY,
            &foreground,
            Viewport { width, height },
        );
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
#[cfg(not(target_arch = "wasm32"))]
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
/// `SceneRenderer::encode_picking` + `PickTarget` end-to-end on a real GPU.
#[test]
#[ignore = "requires a GPU adapter"]
#[cfg(not(target_arch = "wasm32"))]
fn picking_resolves_object_ids_and_background() {
    let gpu = test_gpu();
    let quad = Mesh::from_obj(QUAD_OBJ).expect("quad OBJ parses");
    // The main-pass format is irrelevant to picking (its pipeline is PICK_FORMAT).
    let mut renderer = single(wgpu::TextureFormat::Rgba8UnormSrgb, &quad);

    let (width, height) = (64u32, 64u32);
    let target = PickTarget::new(&gpu.device, width, height);

    // Two objects: a 0.35-scaled quad at NDC x = -0.5 (object 0) and +0.5
    // (object 1), leaving the center a background gap.
    let model = |x: f32| {
        (Mat4::from_translation(Vec3::new(x, 0.0, 0.0)) * Mat4::from_scale(Vec3::splat(0.35)))
            .to_cols_array()
    };
    let draws = [
        Draw {
            mesh_id: 0,
            model: model(-0.5),
            mode: None,
        },
        Draw {
            mesh_id: 0,
            model: model(0.5),
            mode: None,
        },
    ];

    let mut pick = |x: u32, y: u32| {
        pollster::block_on(target.pick(&gpu, &mut renderer, FrameParams::IDENTITY, &draws, x, y))
    };

    // NDC x=-0.5 → pixel ≈ 16; x=+0.5 → ≈ 48; center gap at 32; corner at (2,2).
    assert_eq!(pick(16, 32), Some(0), "left object → index 0");
    assert_eq!(pick(48, 32), Some(1), "right object → index 1");
    assert_eq!(pick(32, 32), None, "center gap → background");
    assert_eq!(pick(2, 2), None, "corner → background");
    // Out-of-bounds coordinates are safely rejected.
    assert_eq!(pick(width, 0), None, "x == width is out of bounds");
}
