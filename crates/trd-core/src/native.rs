//! Native-only headless rendering: render the triangle to an offscreen texture
//! and read it back to a PNG file. Not compiled for the wasm target.

use std::path::Path;

use crate::render::render_triangle;

/// Errors that can occur while rendering headlessly to an image file.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// No GPU adapter could satisfy the request.
    #[error("no suitable GPU adapter found: {0}")]
    RequestAdapter(#[from] wgpu::RequestAdapterError),
    /// The GPU device could not be created.
    #[error("failed to create GPU device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    /// Polling the device to drive the readback failed.
    #[error("failed to poll GPU device: {0}")]
    Poll(#[from] wgpu::PollError),
    /// Mapping the readback buffer failed.
    #[error("failed to map readback buffer: {0}")]
    BufferMap(#[from] wgpu::BufferAsyncError),
    /// The requested image dimensions are invalid (must be non-zero).
    #[error("image dimensions must be non-zero, got {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },
    /// The read-back bytes did not form a valid image buffer.
    #[error("read-back bytes did not form a valid {width}x{height} image")]
    ImageBuffer { width: u32, height: u32 },
    /// Encoding or writing the image file failed.
    #[error("failed to encode or write image: {0}")]
    Image(#[from] image::ImageError),
}

const BYTES_PER_PIXEL: u32 = 4;

/// Renders the hello-triangle at `width` x `height` and writes it to `path` as
/// a PNG. Blocks until the GPU work completes.
pub fn render_to_png(width: u32, height: u32, path: &Path) -> Result<(), RenderError> {
    if width == 0 || height == 0 {
        return Err(RenderError::InvalidDimensions { width, height });
    }
    pollster::block_on(render_to_png_async(width, height, path))
}

async fn render_to_png_async(width: u32, height: u32, path: &Path) -> Result<(), RenderError> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;
    let info = adapter.get_info();
    log::info!(
        "using {:?} adapter \"{}\" ({:?}), driver: {}",
        info.backend,
        info.name,
        info.device_type,
        info.driver_info
    );
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("trd device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("trd render target"),
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
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    render_triangle(&device, &queue, &view, format, crate::render::FrameParams::IDENTITY);

    // copy_texture_to_buffer requires each row to be a multiple of
    // COPY_BYTES_PER_ROW_ALIGNMENT (256 bytes); pad and strip the padding after.
    let unpadded_bytes_per_row = width * BYTES_PER_PIXEL;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("trd readback buffer"),
        size: u64::from(padded_bytes_per_row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("trd readback encoder"),
    });
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
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receiver
        .recv()
        .expect("map_async callback was dropped without sending")?;

    let mut pixels = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
    {
        let mapped = slice
            .get_mapped_range()
            .expect("readback buffer should be mapped after successful map_async");
        for row in 0..height as usize {
            let start = row * padded_bytes_per_row as usize;
            let end = start + unpadded_bytes_per_row as usize;
            pixels.extend_from_slice(&mapped[start..end]);
        }
    }
    staging.unmap();

    let image = image::RgbaImage::from_raw(width, height, pixels)
        .ok_or(RenderError::ImageBuffer { width, height })?;
    image.save(path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_dimensions() {
        // The dimension guard returns before any GPU work, so this needs no GPU.
        let path = std::env::temp_dir().join("trd_zero_dims_should_not_exist.png");
        std::fs::remove_file(&path).ok();

        let err = render_to_png(0, 16, &path).expect_err("zero width must be rejected");
        assert!(matches!(
            err,
            RenderError::InvalidDimensions {
                width: 0,
                height: 16
            }
        ));
        assert!(
            !path.exists(),
            "no file should be written for invalid dimensions"
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn renders_non_blank_triangle() {
        // 65 px wide -> 260 bytes/row, padded to 512; exercises row de-padding.
        let width = 65;
        let height = 48;
        let path = std::env::temp_dir().join("trd_test_triangle.png");
        render_to_png(width, height, &path).expect("render should succeed");

        let image = image::open(&path)
            .expect("output png should decode")
            .to_rgba8();
        assert_eq!(image.dimensions(), (width, height));

        // The clear color is black, so a top corner (outside the triangle) is black
        // while the image center (inside the triangle) is not. This also checks that
        // row de-padding kept pixels spatially aligned.
        let corner = image.get_pixel(0, 0);
        assert_eq!(
            [corner.0[0], corner.0[1], corner.0[2]],
            [0, 0, 0],
            "top-left corner should be the black clear color"
        );

        let center = image.get_pixel(width / 2, height / 2);
        let brightness = u32::from(center.0[0]) + u32::from(center.0[1]) + u32::from(center.0[2]);
        assert!(brightness > 0, "center pixel should not be black");

        std::fs::remove_file(&path).ok();
    }
}
