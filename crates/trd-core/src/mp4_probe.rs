//! Minimal MP4 metadata probe: the browser's missing `ffprobe`.
//!
//! `<video>` exposes `duration`, `videoWidth` and `videoHeight` — but **not the
//! frame rate**, and HTML has no property for it. Without one a browser shell
//! has to number frames on an invented grid, so a 25 fps / 10 s clip shows 300
//! frames instead of 250 and every displayed frame number is fiction. The native
//! shell has no such problem: `ffprobe` reports `r_frame_rate` and `nb_frames`
//! directly.
//!
//! This reads the same facts out of the container so both platforms agree.
//! Parsing happens here rather than in JS because the alternative — `mp4box.js`
//! — is an npm dependency, and this repo installs web dependencies offline
//! through `bun2nix`; adding one for metadata trd can read itself would be a
//! poor trade.
//!
//! Only the boxes needed for a frame count are parsed, from a **`moov` box the
//! caller has already located**: `moov` can sit at either end of a file, and a
//! shell reading a multi-gigabyte local file should seek to it rather than hand
//! the whole thing over.
//!
//! ```text
//! moov
//! └── trak                    (one per stream; the video one is wanted)
//!     ├── mdia
//!     │   ├── mdhd            timescale + duration
//!     │   ├── hdlr            handler type — `vide` identifies the track
//!     │   └── minf/stbl
//!     │       ├── stsz        sample count  = frame count
//!     │       └── stts        sample deltas = frame durations
//!     └── tkhd                track dimensions
//! ```

/// What a container says about its video track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mp4VideoInfo {
    pub width: u32,
    pub height: u32,
    /// Frame rate as an exact rational, e.g. `30000/1001` for 29.97.
    pub fps_num: u32,
    pub fps_den: u32,
    /// Total samples in the track — the real frame count.
    pub frame_count: u32,
    pub duration_us: i64,
}

/// Reads the video track's metadata from a `moov` box.
///
/// `moov` must be the complete box **including** its 8-byte header. Returns
/// `None` when the bytes are not a `moov`, hold no video track, or are truncated
/// — a shell should fall back to its own estimate rather than treat this as
/// fatal, since an unparsed container still plays.
pub fn probe_moov(moov: &[u8]) -> Option<Mp4VideoInfo> {
    let body = box_body(moov, b"moov")?;
    // The first track carrying a `vide` handler wins; audio and subtitle tracks
    // are skipped rather than assumed to come later.
    boxes(body)
        .filter(|(kind, _)| kind == b"trak")
        .find_map(|(_, trak)| probe_trak(trak))
}

fn probe_trak(trak: &[u8]) -> Option<Mp4VideoInfo> {
    let mdia = find_box(trak, b"mdia")?;
    if find_box(mdia, b"hdlr").and_then(handler_type)? != *b"vide" {
        return None;
    }

    let (timescale, track_duration) = find_box(mdia, b"mdhd").and_then(mdhd)?;
    let stbl = find_box(mdia, b"minf").and_then(|minf| find_box(minf, b"stbl"))?;
    let frame_count = find_box(stbl, b"stsz").and_then(stsz_count)?;
    let (sample_count, total_delta) = find_box(stbl, b"stts").and_then(stts_totals)?;
    let (width, height) = find_box(trak, b"tkhd").and_then(tkhd_size)?;

    if frame_count == 0 || timescale == 0 {
        return None;
    }

    // Frame rate as an exact rational rather than a rounded float: `stts` deltas
    // are in timescale units, so `timescale / mean_delta` recovers rates like
    // 30000/1001 exactly. Falls back to the track duration when `stts` is empty.
    let (fps_num, fps_den) = if sample_count > 0 && total_delta > 0 {
        reduce(
            timescale.saturating_mul(sample_count),
            u32::try_from(total_delta).unwrap_or(u32::MAX),
        )
    } else if track_duration > 0 {
        reduce(
            timescale.saturating_mul(frame_count),
            u32::try_from(track_duration).unwrap_or(u32::MAX),
        )
    } else {
        (25, 1)
    };

    let duration_us = if track_duration > 0 {
        (i128::from(track_duration) * 1_000_000 / i128::from(timescale)) as i64
    } else {
        i64::from(frame_count) * 1_000_000 * i64::from(fps_den) / i64::from(fps_num.max(1))
    };

    Some(Mp4VideoInfo {
        width,
        height,
        fps_num: fps_num.max(1),
        fps_den: fps_den.max(1),
        frame_count,
        duration_us,
    })
}

// ---------------------------------------------------------------------------
// Box walking
// ---------------------------------------------------------------------------

/// Yields `(kind, body)` for each box directly inside `data`.
///
/// Stops at the first malformed length rather than propagating an error: a
/// truncated tail still lets the boxes before it be read.
fn boxes(data: &[u8]) -> impl Iterator<Item = ([u8; 4], &[u8])> {
    let mut offset = 0usize;
    std::iter::from_fn(move || {
        let header = data.get(offset..offset + 8)?;
        let size32 = u32::from_be_bytes(header[0..4].try_into().ok()?) as usize;
        let kind: [u8; 4] = header[4..8].try_into().ok()?;
        // `size == 1` means a 64-bit length follows the type; `size == 0` means
        // "to end of file".
        let (size, header_len) = match size32 {
            1 => {
                let ext = data.get(offset + 8..offset + 16)?;
                (u64::from_be_bytes(ext.try_into().ok()?) as usize, 16)
            }
            0 => (data.len() - offset, 8),
            n if n >= 8 => (n, 8),
            _ => return None,
        };
        let end = offset.checked_add(size)?;
        let body = data.get(offset + header_len..end.min(data.len()))?;
        offset = end;
        Some((kind, body))
    })
}

fn find_box<'a>(data: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    boxes(data).find(|(k, _)| k == kind).map(|(_, body)| body)
}

/// The body of `data` when it is a single box of `kind`, header included.
fn box_body<'a>(data: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    let (found, body) = boxes(data).next()?;
    (found == *kind).then_some(body)
}

// ---------------------------------------------------------------------------
// Leaf boxes
// ---------------------------------------------------------------------------

/// `hdlr`: version+flags (4), pre_defined (4), handler type (4).
fn handler_type(hdlr: &[u8]) -> Option<[u8; 4]> {
    hdlr.get(8..12)?.try_into().ok()
}

/// `mdhd` -> `(timescale, duration)`. Version 1 widens the times to 64-bit.
fn mdhd(mdhd: &[u8]) -> Option<(u32, u64)> {
    let version = *mdhd.first()?;
    if version == 1 {
        let timescale = u32::from_be_bytes(mdhd.get(20..24)?.try_into().ok()?);
        let duration = u64::from_be_bytes(mdhd.get(24..32)?.try_into().ok()?);
        Some((timescale, duration))
    } else {
        let timescale = u32::from_be_bytes(mdhd.get(12..16)?.try_into().ok()?);
        let duration = u64::from(u32::from_be_bytes(mdhd.get(16..20)?.try_into().ok()?));
        Some((timescale, duration))
    }
}

/// `stsz` sample count — the frame count, whether or not sizes are uniform.
fn stsz_count(stsz: &[u8]) -> Option<u32> {
    Some(u32::from_be_bytes(stsz.get(8..12)?.try_into().ok()?))
}

/// `stts` -> `(samples, summed deltas)` across every run.
///
/// A constant-frame-rate track has one entry; a variable one has several, and
/// summing both sides yields the mean rate either way.
fn stts_totals(stts: &[u8]) -> Option<(u32, u64)> {
    let entry_count = u32::from_be_bytes(stts.get(4..8)?.try_into().ok()?) as usize;
    let mut samples = 0u32;
    let mut total = 0u64;
    for index in 0..entry_count {
        let at = 8 + index * 8;
        let count = u32::from_be_bytes(stts.get(at..at + 4)?.try_into().ok()?);
        let delta = u32::from_be_bytes(stts.get(at + 4..at + 8)?.try_into().ok()?);
        samples = samples.saturating_add(count);
        total = total.saturating_add(u64::from(count) * u64::from(delta));
    }
    Some((samples, total))
}

/// `tkhd` track dimensions, stored as 16.16 fixed point at the end of the box.
///
/// The offset must clear the `FullBox` header as well as the version-dependent
/// times — miss those 4 bytes and `width` reads the **last matrix element**
/// (`0x40000000`, i.e. 1.0 in 2.30 fixed point), which shifts down to exactly
/// `16384` and looks plausible enough to reach the renderer.
fn tkhd_size(tkhd: &[u8]) -> Option<(u32, u32)> {
    // version + flags, then creation/modification/track_ID/reserved/duration.
    let base = if *tkhd.first()? == 1 { 4 + 32 } else { 4 + 20 };
    // ...reserved(8) + layer/alternate/volume/reserved(8) + matrix(36)
    let at = base + 8 + 8 + 36;
    let width = u32::from_be_bytes(tkhd.get(at..at + 4)?.try_into().ok()?) >> 16;
    let height = u32::from_be_bytes(tkhd.get(at + 4..at + 8)?.try_into().ok()?) >> 16;
    Some((width, height))
}

/// Reduces a rational by its greatest common divisor, so a rate reads as
/// `30000/1001` rather than an unreduced multiple of it.
fn reduce(num: u32, den: u32) -> (u32, u32) {
    let mut a = num;
    let mut b = den;
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    let g = a.max(1);
    (num / g, den / g)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a box: 4-byte big-endian size, 4-byte type, then the payload.
    fn atom(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(payload);
        out
    }

    fn mdhd_v0(timescale: u32, duration: u32) -> Vec<u8> {
        let mut p = vec![0u8; 4]; // version + flags
        p.extend_from_slice(&0u32.to_be_bytes()); // creation
        p.extend_from_slice(&0u32.to_be_bytes()); // modification
        p.extend_from_slice(&timescale.to_be_bytes());
        p.extend_from_slice(&duration.to_be_bytes());
        p.extend_from_slice(&[0; 4]); // language + pre_defined
        atom(b"mdhd", &p)
    }

    /// The unity display matrix, as a real file carries it. The last element is
    /// `0x40000000` (1.0 in 2.30 fixed point), which is what an offset that
    /// misses the `FullBox` header reads as a width of 16384 — so writing the
    /// true matrix rather than zeroes is what gives the size tests teeth.
    fn unity_matrix() -> Vec<u8> {
        [0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000]
            .iter()
            .flat_map(|value| value.to_be_bytes())
            .collect()
    }

    fn tkhd_v0(width: u32, height: u32) -> Vec<u8> {
        let mut p = vec![0u8; 4]; // version + flags
        p.extend_from_slice(&[0; 4]); // creation_time
        p.extend_from_slice(&[0; 4]); // modification_time
        p.extend_from_slice(&[0; 4]); // track_ID
        p.extend_from_slice(&[0; 4]); // reserved
        p.extend_from_slice(&[0; 4]); // duration
        p.extend_from_slice(&[0; 8]); // reserved[2]
        p.extend_from_slice(&[0; 8]); // layer/alternate_group/volume/reserved
        p.extend(unity_matrix());
        p.extend_from_slice(&(width << 16).to_be_bytes());
        p.extend_from_slice(&(height << 16).to_be_bytes());
        atom(b"tkhd", &p)
    }

    fn tkhd_v1(width: u32, height: u32) -> Vec<u8> {
        let mut p = vec![1u8, 0, 0, 0]; // version 1 + flags
        p.extend_from_slice(&[0; 8]); // creation_time (64-bit)
        p.extend_from_slice(&[0; 8]); // modification_time (64-bit)
        p.extend_from_slice(&[0; 4]); // track_ID
        p.extend_from_slice(&[0; 4]); // reserved
        p.extend_from_slice(&[0; 8]); // duration (64-bit)
        p.extend_from_slice(&[0; 8]); // reserved[2]
        p.extend_from_slice(&[0; 8]); // layer/alternate_group/volume/reserved
        p.extend(unity_matrix());
        p.extend_from_slice(&(width << 16).to_be_bytes());
        p.extend_from_slice(&(height << 16).to_be_bytes());
        atom(b"tkhd", &p)
    }

    fn stts_cfr(count: u32, delta: u32) -> Vec<u8> {
        let mut p = vec![0u8; 4];
        p.extend_from_slice(&1u32.to_be_bytes()); // one entry
        p.extend_from_slice(&count.to_be_bytes());
        p.extend_from_slice(&delta.to_be_bytes());
        atom(b"stts", &p)
    }

    fn stsz(count: u32) -> Vec<u8> {
        let mut p = vec![0u8; 4];
        p.extend_from_slice(&0u32.to_be_bytes()); // sample_size 0 => table follows
        p.extend_from_slice(&count.to_be_bytes());
        atom(b"stsz", &p)
    }

    fn hdlr(kind: &[u8; 4]) -> Vec<u8> {
        let mut p = vec![0u8; 8];
        p.extend_from_slice(kind);
        atom(b"hdlr", &p)
    }

    /// The `mdia` subtree of a constant-rate video track.
    fn video_mdia(timescale: u32, delta: u32, frames: u32) -> Vec<u8> {
        video_mdia_with_handler(timescale, delta, frames, b"vide")
    }

    fn video_mdia_with_handler(
        timescale: u32,
        delta: u32,
        frames: u32,
        handler: &[u8; 4],
    ) -> Vec<u8> {
        let mut stbl = stsz(frames);
        stbl.extend(stts_cfr(frames, delta));
        let minf = atom(b"minf", &atom(b"stbl", &stbl));
        let mut mdia = mdhd_v0(timescale, delta * frames);
        mdia.extend(hdlr(handler));
        mdia.extend(minf);
        mdia
    }

    /// `timescale`/`delta` express the frame rate; 12800/512 = 25 fps.
    fn moov(timescale: u32, delta: u32, frames: u32, handler: &[u8; 4]) -> Vec<u8> {
        let mut trak = tkhd_v0(1920, 1080);
        trak.extend(atom(
            b"mdia",
            &video_mdia_with_handler(timescale, delta, frames, handler),
        ));
        atom(b"moov", &atom(b"trak", &trak))
    }

    #[test]
    fn reads_track_size_past_the_display_matrix() {
        // The regression this guards: an offset 4 bytes short lands on the last
        // matrix element and reports 16384 x <real width>, which then stretches
        // the frame plane because the render target takes its aspect from here.
        let info = probe_moov(&moov(12800, 512, 250, b"vide")).expect("probe");
        assert_eq!((info.width, info.height), (1920, 1080));
        assert_ne!(info.width, 16384, "read the matrix instead of the width");
    }

    #[test]
    fn reads_track_size_from_a_version_1_tkhd() {
        let mut trak = tkhd_v1(3840, 2160);
        trak.extend(atom(b"mdia", &video_mdia(12800, 512, 250)));
        let moov = atom(b"moov", &atom(b"trak", &trak));
        let info = probe_moov(&moov).expect("probe");
        assert_eq!((info.width, info.height), (3840, 2160));
    }

    #[test]
    fn reads_frame_count_and_exact_rate() {
        let info = probe_moov(&moov(12800, 512, 250, b"vide")).expect("probe");
        assert_eq!(
            info.frame_count, 250,
            "stsz sample count is the frame count"
        );
        assert_eq!((info.fps_num, info.fps_den), (25, 1));
        assert_eq!((info.width, info.height), (1920, 1080));
        assert_eq!(info.duration_us, 10_000_000);
    }

    /// The whole point of a rational: 29.97 must not round to 30.
    #[test]
    fn ntsc_rate_stays_exact() {
        let info = probe_moov(&moov(30000, 1001, 100, b"vide")).expect("probe");
        assert_eq!((info.fps_num, info.fps_den), (30000, 1001));
    }

    #[test]
    fn ignores_non_video_tracks() {
        assert!(probe_moov(&moov(44100, 1024, 400, b"soun")).is_none());
    }

    #[test]
    fn rejects_other_boxes_and_truncation() {
        assert!(probe_moov(&atom(b"ftyp", b"isom")).is_none());
        let full = moov(12800, 512, 250, b"vide");
        assert!(probe_moov(&full[..full.len() / 2]).is_none());
        assert!(probe_moov(&[]).is_none());
    }
}
