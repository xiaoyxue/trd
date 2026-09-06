//! Measures what loading and deleting a model actually costs in GPU memory.
//!
//! `nvidia-smi` reports per-process memory as `[N/A]` on WDDM and an OS-level
//! total is too noisy to read a single delete out of, so this asks wgpu's
//! allocator directly. It prints the two numbers that differ, which is the whole
//! point of the exercise:
//!
//! * **allocated** — bytes live in resources. This is what a delete must return.
//! * **reserved** — bytes the allocator holds from the driver, including the
//!   free space inside its blocks. This is what an OS-level tool shows, and it
//!   stays high on purpose: the blocks are reused by the next load.
//!
//! Run on a GPU box: `cargo run --release -p trd-core --example vram_probe`

fn report(device: &wgpu::Device, label: &str) -> (u64, u64) {
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let Some(r) = device.generate_allocator_report() else {
        println!("{label:<26} (this backend reports no allocations)");
        return (0, 0);
    };
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    println!(
        "{label:<26} allocated {:8.1} MiB   reserved {:8.1} MiB   ({} allocs, {} blocks)",
        mib(r.total_allocated_bytes),
        mib(r.total_reserved_bytes),
        r.allocations.len(),
        r.blocks.len(),
    );
    (r.total_allocated_bytes, r.total_reserved_bytes)
}

/// A mesh with `n` vertices — the scale of a real GLB, not a test triangle.
fn fat_mesh(n: u32) -> trd_core::Mesh {
    trd_core::Mesh {
        vertices: (0..n)
            .map(|i| {
                let f = i as f32 * 0.001;
                trd_core::Vertex {
                    position: [f.sin(), f.cos(), f * 0.01],
                    color: [1.0, 1.0, 1.0],
                    uv: [0.0, 0.0],
                }
            })
            .collect(),
        indices: (0..n).collect(),
        shading: None,
    }
}

fn main() {
    let (mut renderer, _target) =
        pollster::block_on(trd_core::Renderer::with_meshes(256, 256, &[fat_mesh(3)]))
            .expect("a renderer builds");
    let device = renderer.gpu().device.clone();
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);

    let (base, base_reserved) = report(&device, "baseline (1 tiny mesh)");

    let big = fat_mesh(2_000_000);
    let texture = trd_core::ImageTexture::from_rgba(2048, 2048, vec![200u8; 2048 * 2048 * 4])
        .expect("a 2048² texture builds");
    let load = |renderer: &mut trd_core::Renderer| {
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
        id
    };

    let id = load(&mut renderer);
    let (loaded, loaded_reserved) = report(&device, "after add_mesh + 3 maps");
    renderer.remove_mesh(id).expect("the mesh is resident");
    let (freed, freed_reserved) = report(&device, "after remove_mesh");

    let d = |after: u64, before: u64| (after as f64 - before as f64) / (1024.0 * 1024.0);
    println!(
        "\nload   {:+8.1} MiB allocated / {:+8.1} reserved\n\
         delete {:+8.1} MiB allocated / {:+8.1} reserved\n\
         net    {:+8.1} MiB allocated / {:+8.1} reserved",
        d(loaded, base),
        d(loaded_reserved, base_reserved),
        d(freed, loaded),
        d(freed_reserved, loaded_reserved),
        d(freed, base),
        d(freed_reserved, base_reserved),
    );

    // Repeat: the question a one-shot measurement cannot answer is whether the
    // retained `reserved` is a cache (flat) or a leak (climbing).
    println!("\nrepeating the load/delete cycle — `reserved` must plateau, not climb:");
    for cycle in 1..=5 {
        let id = load(&mut renderer);
        renderer.remove_mesh(id).expect("the mesh is resident");
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let r = device
            .generate_allocator_report()
            .expect("the report is available");
        println!(
            "  cycle {cycle}: allocated {:8.1} MiB   reserved {:8.1} MiB",
            mib(r.total_allocated_bytes),
            mib(r.total_reserved_bytes),
        );
    }
}
