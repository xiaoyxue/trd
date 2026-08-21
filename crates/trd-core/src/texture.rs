//! The **texture abstraction** + its columnar Arrow decode (protocol `0.0.4`, #20).
//!
//! [`Texture`] is the abstraction over a *surface texture*: a function from a
//! surface coordinate (UV, and later position) to sampled data. It is a **trait**
//! rather than a single struct so many *kinds* of texture share one interface —
//! new kinds are added by implementing the trait, not by editing an enum:
//!
//!   * [`ImageTexture`] — an **image map**: RGBA8 pixels, decoded from an Arrow
//!     `fixed_shape_tensor<u8>[H, W, 4]` table (the bunny albedo). This is the
//!     self-describing image tensor trd already *emits* on output
//!     (`protocol/output_session.rs`),
//!     so a texture input is symmetric with a rendered frame.
//!   * [`ConstantTexture`] — a **constant map**: one uniform color (a 1x1 image);
//!     the default when a mesh is drawn textured but no texture stream is bound.
//!   * *future kinds* (implement [`Texture`] the same way): bump / normal maps,
//!     checker / noise procedural maps, etc.
//!
//! **GPU vs CPU.** Every kind yields GPU-uploadable RGBA8 via [`Texture::to_image`]
//! (image maps return their pixels; constant / procedural maps *bake* to a small
//! image), which the renderer uploads to a `wgpu::Texture` and samples in the
//! shader. [`Texture::sample`] is the matching CPU evaluation (nearest lookup for
//! image maps, the constant for constant maps) — used by tests, procedural
//! baking, and non-GPU kinds.
//!
//! **sRGB.** Bytes are treated as sRGB-encoded color and uploaded to an
//! `Rgba8UnormSrgb` texture, so the GPU linearizes texels on sample — matching
//! the existing output path. Decode here is byte-exact / colorspace-agnostic;
//! the sRGB choice lives at upload (`render.rs`).

use arrow::array::{Array, FixedSizeListArray, RecordBatch, UInt8Array};
use arrow::datatypes::DataType;
use thiserror::Error;

/// The single image column of an Arrow texture table.
pub const TEXTURE_COLUMN: &str = "rgba";

/// Arrow canonical-extension metadata keys carried on the `rgba` field.
pub(crate) const EXTENSION_NAME_KEY: &str = "ARROW:extension:name";
pub(crate) const EXTENSION_METADATA_KEY: &str = "ARROW:extension:metadata";
/// The canonical fixed-shape-tensor extension name.
pub(crate) const FIXED_SHAPE_TENSOR: &str = "arrow.fixed_shape_tensor";

/// Raw RGBA8 image data ready to upload to a `wgpu::Texture`: tightly-packed,
/// row-major, `height * width * 4` bytes (`[H, W, 4]`).
#[derive(Debug, Clone, PartialEq)]
pub struct ImageData {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Tightly-packed row-major RGBA8 bytes (`height * width * 4`).
    pub rgba: Vec<u8>,
}

/// The abstraction over a surface texture (#20): maps a surface `uv` coordinate
/// to sampled RGBA data. Implemented by each texture *kind* ([`ImageTexture`],
/// [`ConstantTexture`], and future bump / procedural maps).
///
/// The renderer only ever needs [`to_image`](Texture::to_image) (the bytes to
/// upload); [`sample`](Texture::sample) is the CPU counterpart for tests and
/// non-image kinds. Object-safe, so a bound texture can be held as `&dyn Texture`.
///
/// **The CPU face is deliberate, and it is why this file is at the crate root**
/// (#247 S5, upholding #232). Today only `to_image` has production callers —
/// [`sample`](Texture::sample) and [`kind`](Texture::kind) are exercised by
/// tests — and reducing the trait to "a thing that produces [`ImageData`] for
/// upload" would make it a *GPU-upload* interface, which belongs in `render/`.
/// It is kept as an evaluation of a surface texture because that is what a
/// procedural / bump / noise kind implements, and because it keeps `mesh` and
/// `texture` symmetric: both are decoded assets at the root, each with a GPU
/// face in `render/` (`mesh_store.rs`, `bound_texture.rs`). Deleting `sample`
/// and moving this file are **one** decision, not two — take them together or
/// not at all.
pub trait Texture {
    /// The RGBA color at surface coordinate `uv` (each component in `[0, 1]`).
    /// Nearest lookup with clamp-to-edge for image maps; the constant color for
    /// constant maps; computed for procedural maps. `v = 0` is the first
    /// (top) image row, matching the uploaded texel layout.
    fn sample(&self, uv: [f32; 2]) -> [u8; 4];

    /// The RGBA8 image to upload to a `wgpu::Texture` for GPU sampling.
    fn to_image(&self) -> ImageData;

    /// A short human-readable kind name, for diagnostics/logging.
    fn kind(&self) -> &'static str;
}

/// Errors decoding an Arrow texture table into an [`ImageTexture`].
#[derive(Debug, Error)]
pub enum TextureError {
    /// The texture table has no rows (needs exactly one image row).
    #[error("texture table is empty (needs one row)")]
    Empty,
    /// The required `rgba` image column is absent.
    #[error("texture table is missing required column `{0}`")]
    MissingColumn(&'static str),
    /// A column had an unexpected Arrow type.
    #[error("texture column `{column}` has unexpected type (expected {expected}, got {actual:?})")]
    ColumnType {
        column: &'static str,
        expected: &'static str,
        actual: DataType,
    },
    /// The `rgba` column carries null values.
    #[error("texture column `{0}` contains null values")]
    NullValues(&'static str),
    /// The tensor extension shape is not `[H, W, 4]` (interleaved RGBA).
    #[error("texture tensor shape {shape:?} is not [H, W, 4] (interleaved RGBA)")]
    Shape { shape: Vec<usize> },
    /// The flat byte length disagrees with the `H*W*4` implied by the shape.
    #[error("texture byte length {actual} != {width}x{height}x4 = {expected}")]
    ByteLength {
        actual: usize,
        expected: usize,
        width: u32,
        height: u32,
    },
    /// Zero width/height.
    #[error("texture dimensions {width}x{height} are invalid (must be non-zero)")]
    InvalidDimensions { width: u32, height: u32 },
    /// The `rgba` field lacks the `arrow.fixed_shape_tensor` extension (so its
    /// `[H, W, 4]` shape is not self-describing).
    #[error("texture column `{TEXTURE_COLUMN}` is not a fixed_shape_tensor: {0}")]
    NotTensor(String),
}

/// An **image map**: a decoded RGBA8 image sampled by UV. Pixels are row-major
/// `[H, W, 4]` (see [`ImageData`]); uploads directly to an `Rgba8UnormSrgb`
/// `wgpu::Texture`.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageTexture {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl ImageTexture {
    /// Builds an image map from tightly-packed row-major RGBA8 bytes, validating
    /// non-zero dimensions and `rgba.len() == width * height * 4`.
    pub fn from_rgba(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self, TextureError> {
        if width == 0 || height == 0 {
            return Err(TextureError::InvalidDimensions { width, height });
        }
        let expected = width as usize * height as usize * 4;
        if rgba.len() != expected {
            return Err(TextureError::ByteLength {
                actual: rgba.len(),
                expected,
                width,
                height,
            });
        }
        Ok(Self {
            width,
            height,
            rgba,
        })
    }

    /// Image width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The row-major RGBA8 pixel bytes (`height * width * 4`).
    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }

    /// Decodes the first row of an Arrow **texture table** into an image map.
    ///
    /// Expects a `rgba` column of type `FixedSizeList<UInt8>[H*W*4]` bearing the
    /// `arrow.fixed_shape_tensor` extension with shape `[H, W, 4]`; the height
    /// and width come from that shape (self-describing, like the output schema).
    pub fn from_arrow(batch: &RecordBatch) -> Result<Self, TextureError> {
        if batch.num_rows() == 0 {
            return Err(TextureError::Empty);
        }
        let index = batch
            .schema()
            .index_of(TEXTURE_COLUMN)
            .map_err(|_| TextureError::MissingColumn(TEXTURE_COLUMN))?;
        let field = batch.schema().field(index).clone();
        let column = batch.column(index);

        // Height/width come from the self-describing fixed_shape_tensor extension
        // shape `[H, W, 4]`, read from the field's canonical-extension metadata
        // (arrow exposes `list_size`/rank but not the shape array), mirroring the
        // output image tensor.
        let meta = field.metadata();
        if meta.get(EXTENSION_NAME_KEY).map(String::as_str) != Some(FIXED_SHAPE_TENSOR) {
            return Err(TextureError::NotTensor(format!(
                "column `{TEXTURE_COLUMN}` is not `{FIXED_SHAPE_TENSOR}`"
            )));
        }
        let shape = meta
            .get(EXTENSION_METADATA_KEY)
            .and_then(|json| parse_tensor_shape(json))
            .ok_or_else(|| {
                TextureError::NotTensor("missing/invalid fixed_shape_tensor shape metadata".into())
            })?;
        let (height, width) = match shape.as_slice() {
            [height, width, 4] => (*height, *width),
            _ => return Err(TextureError::Shape { shape }),
        };
        let width = u32::try_from(width).map_err(|_| TextureError::Shape {
            shape: shape.clone(),
        })?;
        let height = u32::try_from(height).map_err(|_| TextureError::Shape { shape })?;

        let list = column
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| TextureError::ColumnType {
                column: TEXTURE_COLUMN,
                expected: "FixedSizeList<UInt8>",
                actual: column.data_type().clone(),
            })?;
        if list.is_null(0) {
            return Err(TextureError::NullValues(TEXTURE_COLUMN));
        }
        let row = list.value(0);
        let bytes =
            row.as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| TextureError::ColumnType {
                    column: TEXTURE_COLUMN,
                    expected: "FixedSizeList<UInt8>",
                    actual: row.data_type().clone(),
                })?;
        if bytes.null_count() > 0 {
            return Err(TextureError::NullValues(TEXTURE_COLUMN));
        }
        Self::from_rgba(width, height, bytes.values().to_vec())
    }
}

impl Texture for ImageTexture {
    fn sample(&self, uv: [f32; 2]) -> [u8; 4] {
        // Clamp-to-edge, nearest-neighbor. `v = 0` maps to the first (top) row.
        let u = uv[0].clamp(0.0, 1.0);
        let v = uv[1].clamp(0.0, 1.0);
        let x = ((u * self.width as f32) as u32).min(self.width - 1);
        let y = ((v * self.height as f32) as u32).min(self.height - 1);
        let i = (y as usize * self.width as usize + x as usize) * 4;
        [
            self.rgba[i],
            self.rgba[i + 1],
            self.rgba[i + 2],
            self.rgba[i + 3],
        ]
    }

    fn to_image(&self) -> ImageData {
        ImageData {
            width: self.width,
            height: self.height,
            rgba: self.rgba.clone(),
        }
    }

    fn kind(&self) -> &'static str {
        "image"
    }
}

/// A **constant map**: one uniform RGBA color at every UV. Bakes to a 1x1 image
/// for GPU sampling. The default albedo when a mesh is drawn textured but no
/// texture stream is bound (so sampling is an identity multiply against white).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantTexture {
    /// The uniform RGBA color.
    pub color: [u8; 4],
}

impl ConstantTexture {
    /// A constant map of `color`.
    pub fn new(color: [u8; 4]) -> Self {
        Self { color }
    }

    /// Opaque white — the identity albedo.
    pub fn white() -> Self {
        Self {
            color: [255, 255, 255, 255],
        }
    }
}

impl Texture for ConstantTexture {
    fn sample(&self, _uv: [f32; 2]) -> [u8; 4] {
        self.color
    }

    fn to_image(&self) -> ImageData {
        ImageData {
            width: 1,
            height: 1,
            rgba: self.color.to_vec(),
        }
    }

    fn kind(&self) -> &'static str {
        "constant"
    }
}

/// Extracts the integer `shape` array from a canonical
/// `arrow.fixed_shape_tensor` extension-metadata JSON string, e.g.
/// `{"shape":[2,3,4],"dim_names":["height","width","channel"]}` -> `[2, 3, 4]`.
/// A tiny purpose-built parser (no JSON dependency) for this fixed,
/// arrow-produced shape; returns `None` if the `shape` array is absent or
/// malformed.
pub(crate) fn parse_tensor_shape(json: &str) -> Option<Vec<usize>> {
    const KEY: &str = "\"shape\"";
    let after_key = &json[json.find(KEY)? + KEY.len()..];
    let open = after_key.find('[')?;
    let close = after_key[open..].find(']')? + open;
    after_key[open + 1..close]
        .split(',')
        .map(|s| s.trim().parse::<usize>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow::datatypes::{Field, Schema};
    use arrow_schema::extension::FixedShapeTensor;

    /// Builds a one-row texture table with a `rgba` fixed_shape_tensor column of
    /// shape `[height, width, 4]` from tightly-packed RGBA bytes.
    fn texture_batch(width: usize, height: usize, rgba: Vec<u8>) -> RecordBatch {
        let list_size = (width * height * 4) as i32;
        let storage = DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::UInt8, false)),
            list_size,
        );
        let extension = FixedShapeTensor::try_new(
            DataType::UInt8,
            vec![height, width, 4],
            Some(vec![
                "height".to_string(),
                "width".to_string(),
                "channel".to_string(),
            ]),
            None,
        )
        .unwrap();
        let field = Field::new(TEXTURE_COLUMN, storage, false).with_extension_type(extension);
        let array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::UInt8, false)),
            list_size,
            Arc::new(UInt8Array::from(rgba)),
            None,
        );
        RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![Arc::new(array)]).unwrap()
    }

    #[test]
    fn image_from_rgba_validates_length() {
        assert!(ImageTexture::from_rgba(2, 2, vec![0; 16]).is_ok());
        assert!(matches!(
            ImageTexture::from_rgba(2, 2, vec![0; 15]),
            Err(TextureError::ByteLength { .. })
        ));
        assert!(matches!(
            ImageTexture::from_rgba(0, 2, vec![]),
            Err(TextureError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn image_from_arrow_round_trips_a_2x2_checker() {
        // 2x2 checker: white, red / green, blue.
        let rgba = vec![
            255, 255, 255, 255, 255, 0, 0, 255, // row 0: white, red
            0, 255, 0, 255, 0, 0, 255, 255, // row 1: green, blue
        ];
        let batch = texture_batch(2, 2, rgba.clone());
        let tex = ImageTexture::from_arrow(&batch).unwrap();
        assert_eq!((tex.width(), tex.height()), (2, 2));
        assert_eq!(tex.rgba(), rgba.as_slice());
        assert_eq!(tex.to_image().rgba, rgba);
        assert_eq!(tex.kind(), "image");
    }

    #[test]
    fn image_from_arrow_reads_non_square_dimensions_from_shape() {
        // 3 wide x 2 tall => shape [2, 3, 4].
        let rgba = vec![7u8; 3 * 2 * 4];
        let batch = texture_batch(3, 2, rgba);
        let tex = ImageTexture::from_arrow(&batch).unwrap();
        assert_eq!((tex.width(), tex.height()), (3, 2));
        assert_eq!(tex.rgba().len(), 3 * 2 * 4);
    }

    #[test]
    fn image_sample_nearest_hits_the_four_texels() {
        // 2x2: white, red / green, blue. Sample near each texel center.
        let rgba = vec![
            255, 255, 255, 255, 255, 0, 0, 255, // row 0: white, red
            0, 255, 0, 255, 0, 0, 255, 255, // row 1: green, blue
        ];
        let tex = ImageTexture::from_rgba(2, 2, rgba).unwrap();
        assert_eq!(tex.sample([0.25, 0.25]), [255, 255, 255, 255]); // top-left white
        assert_eq!(tex.sample([0.75, 0.25]), [255, 0, 0, 255]); // top-right red
        assert_eq!(tex.sample([0.25, 0.75]), [0, 255, 0, 255]); // bottom-left green
        assert_eq!(tex.sample([0.75, 0.75]), [0, 0, 255, 255]); // bottom-right blue
                                                                // Clamp-to-edge past the borders.
        assert_eq!(tex.sample([-1.0, -1.0]), [255, 255, 255, 255]);
        assert_eq!(tex.sample([2.0, 2.0]), [0, 0, 255, 255]);
    }

    #[test]
    fn image_from_arrow_empty_table_errors() {
        let batch = texture_batch(1, 1, vec![0, 0, 0, 0]).slice(0, 0);
        assert!(matches!(
            ImageTexture::from_arrow(&batch),
            Err(TextureError::Empty)
        ));
    }

    #[test]
    fn image_from_arrow_missing_column_errors() {
        let schema = Schema::new(vec![Field::new("not_rgba", DataType::UInt8, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(UInt8Array::from(vec![0u8]))],
        )
        .unwrap();
        assert!(matches!(
            ImageTexture::from_arrow(&batch),
            Err(TextureError::MissingColumn("rgba"))
        ));
    }

    #[test]
    fn constant_map_samples_its_color_and_bakes_1x1() {
        let c = ConstantTexture::new([10, 20, 30, 40]);
        assert_eq!(c.sample([0.3, 0.7]), [10, 20, 30, 40]);
        let img = c.to_image();
        assert_eq!((img.width, img.height), (1, 1));
        assert_eq!(img.rgba, vec![10, 20, 30, 40]);
        assert_eq!(c.kind(), "constant");
        assert_eq!(ConstantTexture::white().color, [255, 255, 255, 255]);
    }

    #[test]
    fn texture_trait_is_object_safe() {
        let textures: Vec<Box<dyn Texture>> = vec![
            Box::new(ConstantTexture::white()),
            Box::new(ImageTexture::from_rgba(1, 1, vec![1, 2, 3, 4]).unwrap()),
        ];
        assert_eq!(textures[0].sample([0.0, 0.0]), [255, 255, 255, 255]);
        assert_eq!(textures[1].sample([0.0, 0.0]), [1, 2, 3, 4]);
    }

    #[test]
    fn parse_tensor_shape_reads_the_shape_array() {
        assert_eq!(
            parse_tensor_shape(r#"{"shape":[2,3,4]}"#),
            Some(vec![2, 3, 4])
        );
        assert_eq!(
            parse_tensor_shape(r#"{"shape": [100, 200, 4], "dim_names": ["h","w","c"]}"#),
            Some(vec![100, 200, 4])
        );
        assert_eq!(parse_tensor_shape(r#"{"dim_names":["h"]}"#), None);
        assert_eq!(parse_tensor_shape(r#"{"shape":[2,x,4]}"#), None);
    }
}
