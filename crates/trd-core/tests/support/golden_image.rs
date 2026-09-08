use std::path::Path;

const CHANNEL_EPS: u8 = 16;
const MAX_DIFF_FRACTION: f64 = 0.02;

pub fn compare_or_update(
    actual: &[u8],
    width: u32,
    height: u32,
    golden: &Path,
    update: bool,
) -> Result<(), String> {
    assert_eq!(actual.len(), (width * height * 4) as usize);
    if update {
        let image = image::RgbaImage::from_raw(width, height, actual.to_vec())
            .expect("validated RGBA length");
        return image
            .save(golden)
            .map_err(|error| format!("write {}: {error}", golden.display()));
    }
    let expected = image::open(golden)
        .map_err(|error| format!("open {}: {error}", golden.display()))?
        .to_rgba8();
    if expected.dimensions() != (width, height) {
        return Err(format!(
            "{} has unexpected dimensions {:?}",
            golden.display(),
            expected.dimensions()
        ));
    }
    let expected = expected.into_raw();
    let total = (width * height) as usize;
    let mut differing = 0;
    let mut maximum = 0;
    for (actual, expected) in actual.chunks_exact(4).zip(expected.chunks_exact(4)) {
        let delta = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        maximum = maximum.max(delta);
        differing += usize::from(delta > CHANNEL_EPS);
    }
    let fraction = differing as f64 / total as f64;
    if fraction > MAX_DIFF_FRACTION {
        return Err(format!(
            "{}: {differing}/{total} pixels differ beyond eps={CHANNEL_EPS} \
             ({:.3}% > {:.3}%; max channel diff {maximum})",
            golden.display(),
            fraction * 100.0,
            MAX_DIFF_FRACTION * 100.0,
        ));
    }
    Ok(())
}
