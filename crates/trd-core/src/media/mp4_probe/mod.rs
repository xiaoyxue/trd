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
//! Locating `moov` is deliberately *not* this module's job — a browser shell
//! does it with a few small range reads, a native shell with `ffprobe` — so this
//! stays a pure function over bytes and unit-testable without a file, a network,
//! or a GPU.
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

use super::video::VideoTiming;

/// Reads the video track's metadata from a `moov` box.
///
/// `moov` must be the complete box **including** its 8-byte header. Returns
/// `None` when the bytes are not a `moov`, hold no video track, or are truncated
/// — a shell should fall back to its own estimate rather than treat this as
/// fatal, since an unparsed container still plays.
pub fn probe_moov(moov: &[u8]) -> Option<VideoTiming> {
    let body = box_body(moov, b"moov")?;
    // The first track carrying a `vide` handler wins; audio and subtitle tracks
    // are skipped rather than assumed to come later.
    boxes(body)
        .filter(|(kind, _)| kind == b"trak")
        .find_map(|(_, trak)| probe_trak(trak))
}

fn probe_trak(trak: &[u8]) -> Option<VideoTiming> {
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
    //
    // Both products are formed in `u64`: `timescale * sample_count` leaves `u32`
    // after only ~75 minutes of 60 fps at a 16 kHz timescale, and ~13 at 90 kHz,
    // and a saturating multiply there reports a plausible but wrong rate rather
    // than failing (#314).
    let (fps_num, fps_den) = if sample_count > 0 && total_delta > 0 {
        reduce(u64::from(timescale) * u64::from(sample_count), total_delta)
    } else if track_duration > 0 {
        reduce(
            u64::from(timescale) * u64::from(frame_count),
            track_duration,
        )
    } else {
        (25, 1)
    };

    let duration_us = if track_duration > 0 {
        (i128::from(track_duration) * 1_000_000 / i128::from(timescale)) as i64
    } else {
        i64::from(frame_count) * 1_000_000 * i64::from(fps_den) / i64::from(fps_num.max(1))
    };

    Some(VideoTiming {
        width,
        height,
        fps_num: fps_num.max(1),
        fps_den: fps_den.max(1),
        frame_count,
        duration_us,
        unpresented_tail_samples: unpresented_tail(stbl, track_duration),
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

/// Trailing samples whose presentation time is at or past the track's declared
/// duration — stored by the container, but never shown.
///
/// A recorder that stops mid-interval writes a final sample outside the
/// presentation window; ffmpeg surfaces the same thing natively as
/// `AV_PKT_FLAG_DISCARD`, and no decoder outputs it (#324). Counting it as a
/// frame is what leaves a final timeline index with no picture behind it.
///
/// **`stts` alone is not enough.** It accumulates to *decode* timestamps, and
/// B-frame content reorders: the sample that sits past the end does so in
/// **presentation** order, which is `stts` plus the composition offset `ctts`
/// carries. Reading only `stts` finds nothing on exactly the files this exists
/// for.
///
/// Both tables are run-length encoded and walked together, one sample at a time
/// but with no allocation — the moov is already in memory, and the arithmetic is
/// two additions per sample.
fn unpresented_tail(stbl: &[u8], track_duration: u64) -> Option<u32> {
    if track_duration == 0 {
        return Some(0);
    }
    let stts = find_box(stbl, b"stts")?;
    let deltas = RunTable::new(stts)?;
    // Absent on constant-order content, and that is not an error: no `ctts`
    // means presentation order *is* decode order.
    let mut offsets = find_box(stbl, b"ctts").and_then(RunTable::new);

    let mut dts = 0i64;
    let mut unpresented = 0u32;
    for (count, delta) in deltas.runs() {
        for _ in 0..count {
            let offset = offsets.as_mut().and_then(RunTable::next_value).unwrap_or(0);
            if dts + i64::from(offset) >= track_duration as i64 {
                unpresented = unpresented.saturating_add(1);
            }
            dts += i64::from(delta);
        }
    }
    Some(unpresented)
}

/// A `stts`/`ctts` style run-length table: `[version+flags][entry_count]` then
/// `(count, value)` pairs. `ctts` version 1 stores signed offsets, which is why
/// the value is read as `i32`.
struct RunTable<'a> {
    body: &'a [u8],
    entries: usize,
    entry: usize,
    left: u32,
}

impl<'a> RunTable<'a> {
    fn new(body: &'a [u8]) -> Option<Self> {
        let entries = u32::from_be_bytes(body.get(4..8)?.try_into().ok()?) as usize;
        Some(Self {
            body,
            entries,
            entry: 0,
            left: 0,
        })
    }

    fn pair(&self, index: usize) -> Option<(u32, i32)> {
        let at = 8 + index * 8;
        let count = u32::from_be_bytes(self.body.get(at..at + 4)?.try_into().ok()?);
        let value = i32::from_be_bytes(self.body.get(at + 4..at + 8)?.try_into().ok()?);
        Some((count, value))
    }

    /// Every `(count, value)` run in order.
    fn runs(&self) -> impl Iterator<Item = (u32, i32)> + '_ {
        (0..self.entries).filter_map(|index| self.pair(index))
    }

    /// The next sample's value, unrolling runs as it goes.
    fn next_value(&mut self) -> Option<i32> {
        while self.left == 0 {
            let (count, _) = self.pair(self.entry)?;
            if self.entry >= self.entries {
                return None;
            }
            self.entry += 1;
            self.left = count;
            if count == 0 {
                continue;
            }
        }
        let (_, value) = self.pair(self.entry - 1)?;
        self.left -= 1;
        Some(value)
    }
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
///
/// Takes `u64` because the numerator is `timescale * sample_count`, which is
/// routinely larger than `u32` holds — see the call site. Should the reduced
/// pair still not fit, both sides are scaled by the same divisor rather than
/// clamped: an approximate rate stays usable, a truncated one silently is not.
fn reduce(num: u64, den: u64) -> (u32, u32) {
    let mut a = num;
    let mut b = den;
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    let g = a.max(1);
    let (num, den) = (num / g, den / g);
    let scale = num.max(den).div_ceil(u64::from(u32::MAX)).max(1);
    ((num / scale).max(1) as u32, (den / scale).max(1) as u32)
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

    /// Builds an `stts`/`ctts` payload from `(count, value)` runs.
    fn run_table(runs: &[(u32, i32)]) -> Vec<u8> {
        let mut p = vec![0u8; 4]; // version + flags
        p.extend_from_slice(&(runs.len() as u32).to_be_bytes());
        for (count, value) in runs {
            p.extend_from_slice(&count.to_be_bytes());
            p.extend_from_slice(&value.to_be_bytes());
        }
        p
    }

    /// An `stbl` carrying just the tables this reads.
    fn timing_stbl(stts: &[(u32, i32)], ctts: Option<&[(u32, i32)]>) -> Vec<u8> {
        let mut body = atom(b"stts", &run_table(stts));
        if let Some(ctts) = ctts {
            body.extend(atom(b"ctts", &run_table(ctts)));
        }
        body
    }

    /// #324, with the shape measured on a 217.77 GiB recording: 694 839 pictures
    /// one timescale unit apart, then a sample the container stores but never
    /// presents — its *presentation* time is the declared duration itself, so it
    /// starts exactly where the track ends.
    ///
    /// ffmpeg reports that sample as `AV_PKT_FLAG_DISCARD`; here it is found from
    /// the boxes alone, which is all the browser has.
    #[test]
    fn a_sample_starting_at_the_declared_end_is_never_presented() {
        let stbl = timing_stbl(&[(694_840, 1)], Some(&[(694_839, 0), (1, 1)]));

        assert_eq!(unpresented_tail(&stbl, 694_840), Some(1));
    }

    /// The trap this walked into first: `stts` accumulates to **decode** time, so
    /// on B-frame content the trailing sample looks in-range until `ctts` is
    /// applied. Same table without the composition offset — and the answer is
    /// wrong.
    #[test]
    fn decode_order_alone_cannot_see_the_unpresented_sample() {
        let stbl = timing_stbl(&[(694_840, 1)], None);

        assert_eq!(
            unpresented_tail(&stbl, 694_840),
            Some(0),
            "without ctts every sample's dts is inside the track, which is exactly why ctts is read"
        );
    }

    /// The ordinary case must stay zero, or every file would grow a footnote.
    #[test]
    fn a_track_that_ends_with_its_last_picture_has_no_unpresented_tail() {
        let stbl = timing_stbl(&[(250, 1)], None);

        assert_eq!(unpresented_tail(&stbl, 250), Some(0));
    }

    /// A container that declares no duration says nothing about what it presents,
    /// and guessing would invent a warning out of missing metadata.
    #[test]
    fn an_unknown_duration_reports_no_unpresented_tail() {
        let stbl = timing_stbl(&[(250, 1)], None);

        assert_eq!(unpresented_tail(&stbl, 0), Some(0));
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

    /// A real file's `moov` holds several tracks, and the video one is not
    /// necessarily first — picking "the first `trak`" would read the audio
    /// track's (absent) dimensions and report nothing.
    #[test]
    fn finds_the_video_track_behind_an_audio_track() {
        let mut audio_trak = tkhd_v0(0, 0);
        audio_trak.extend(atom(
            b"mdia",
            &video_mdia_with_handler(44100, 1024, 400, b"soun"),
        ));
        let mut video_trak = tkhd_v0(1280, 720);
        video_trak.extend(atom(b"mdia", &video_mdia(12800, 512, 250)));

        let mut body = atom(b"trak", &audio_trak);
        body.extend(atom(b"trak", &video_trak));
        let info = probe_moov(&atom(b"moov", &body)).expect("probe");

        assert_eq!((info.width, info.height), (1280, 720));
        assert_eq!(info.frame_count, 250);
    }

    /// A variable-rate track has several `stts` runs; the mean must come from
    /// both sides, not from the first entry.
    #[test]
    fn averages_a_variable_frame_rate_across_stts_runs() {
        let mut stts = vec![0u8; 4];
        stts.extend_from_slice(&2u32.to_be_bytes()); // two entries
        for (count, delta) in [(50u32, 512u32), (50, 1024)] {
            stts.extend_from_slice(&count.to_be_bytes());
            stts.extend_from_slice(&delta.to_be_bytes());
        }
        let mut stbl = stsz(100);
        stbl.extend(atom(b"stts", &stts));
        let minf = atom(b"minf", &atom(b"stbl", &stbl));
        let mut mdia = mdhd_v0(12800, 76_800);
        mdia.extend(hdlr(b"vide"));
        mdia.extend(minf);
        let mut trak = tkhd_v0(1920, 1080);
        trak.extend(atom(b"mdia", &mdia));

        let info = probe_moov(&atom(b"moov", &atom(b"trak", &trak))).expect("probe");

        // 100 samples over 76 800 units at 12 800/s = 6 s ⇒ 100/6 fps, reduced.
        assert_eq!((info.fps_num, info.fps_den), (50, 3));
        assert_eq!(info.frame_count, 100);
    }

    /// `timescale * sample_count` leaves `u32` long before the *video* is
    /// unusual: at a 16 kHz timescale it overflows after ~75 minutes of 60 fps,
    /// and at the equally common 90 kHz after ~13 (#314).
    ///
    /// A saturating multiply there does not fail loudly — it reports a plausible
    /// but wrong rate, which mis-scales the whole timeline. 60000/1000 over
    /// 100 000 frames is 60 fps exactly, and `60000 * 100000 = 6e9` is past
    /// `u32::MAX`; saturating gave 42.95 fps.
    #[test]
    fn a_long_track_keeps_its_exact_rate_instead_of_saturating() {
        let info = probe_moov(&moov(60_000, 1_000, 100_000, b"vide")).expect("probe");

        assert!(
            u64::from(60_000u32) * u64::from(100_000u32) > u64::from(u32::MAX),
            "the fixture must actually overflow u32, or it proves nothing"
        );
        assert_eq!((info.fps_num, info.fps_den), (60, 1));
        assert_eq!(info.frame_count, 100_000);
    }

    /// The same overflow sits in the `track_duration` fallback arm, which an
    /// empty `stts` selects — so it needs its own fixture rather than trusting
    /// that fixing one arm fixed both.
    #[test]
    fn the_duration_fallback_keeps_its_exact_rate_too() {
        let mut stts = vec![0u8; 4];
        stts.extend_from_slice(&0u32.to_be_bytes()); // no entries at all
        let mut stbl = stsz(100_000);
        stbl.extend(atom(b"stts", &stts));
        let minf = atom(b"minf", &atom(b"stbl", &stbl));
        let mut mdia = mdhd_v0(60_000, 100_000_000); // 100 000 frames x 1 000
        mdia.extend(hdlr(b"vide"));
        mdia.extend(minf);
        let mut trak = tkhd_v0(3840, 2160);
        trak.extend(atom(b"mdia", &mdia));

        let info = probe_moov(&atom(b"moov", &atom(b"trak", &trak))).expect("probe");

        assert_eq!((info.fps_num, info.fps_den), (60, 1));
        assert_eq!(info.frame_count, 100_000);
    }

    /// What the wrong rate actually broke: the browser derives the timeline
    /// length from the rational, and validates the container's duration against
    /// it. Any error in the rate scales straight into seconds, so this pins the
    /// derived duration rather than only the rational it came from.
    #[test]
    fn a_long_tracks_derived_duration_matches_its_real_one() {
        let info = probe_moov(&moov(60_000, 1_000, 100_000, b"vide")).expect("probe");

        let derived =
            f64::from(info.frame_count) * f64::from(info.fps_den) / f64::from(info.fps_num);
        let real = info.duration_us as f64 / 1e6;

        // 100 000 frames at 60 fps is 1 666.667 s; saturating reported 2 328.3 s.
        assert!(
            (derived - real).abs() < 0.001,
            "derived {derived:.3}s vs real {real:.3}s"
        );
    }

    /// Reducing in `u64` must not reintroduce the truncation one level up: a
    /// rational that cannot be narrowed is scaled, keeping the *rate* right,
    /// because an approximate rate is recoverable and a truncated one is not.
    #[test]
    fn an_unreducible_oversized_rational_is_scaled_not_clamped() {
        let num = u64::from(u32::MAX) * 4 + 2; // even, so it survives halving
        let (n, d) = reduce(num, num / 2);

        assert!(n > 0 && d > 0, "neither side may collapse to zero");
        assert_eq!(
            f64::from(n) / f64::from(d),
            2.0,
            "the ratio must survive narrowing"
        );
    }
}
