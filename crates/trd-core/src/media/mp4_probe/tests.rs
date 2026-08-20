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

fn video_mdia_with_handler(timescale: u32, delta: u32, frames: u32, handler: &[u8; 4]) -> Vec<u8> {
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
