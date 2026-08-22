use std::collections::HashMap;
use std::ffi::OsStr;
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::sync::Arc;

use crate::error::NativeVideoEditingError;

pub struct DecodedFrame {
    /// The frame this picture actually is, recovered from the timestamp ffmpeg
    /// reported for it — **not** the index that was asked for.
    ///
    /// The two differ on a variable-rate source: the seek time is computed from
    /// the timeline's nominal grid, which such a container does not sit on, so
    /// ffmpeg's at-or-after seek can hand back the neighbouring picture. Taking
    /// the request on trust is how that became a silent one-frame placement
    /// error rather than a visible one (#319).
    pub index: u32,
    /// The presentation timestamp ffmpeg reported, in seconds. Falls back to the
    /// requested instant when the report could not be read.
    pub media_time_seconds: f64,
    /// How long this frame is shown, as the container declares it; `0.0` when
    /// unknown, which leaves the timeline's nominal interval to stand in.
    pub duration_seconds: f64,
    pub rgba: Vec<u8>,
}

/// The flags that make ffmpeg describe what it emits, in the source clock.
///
/// **`-copyts` is load-bearing.** Without it `-ss` rebases the output clock to
/// zero, so `showinfo` reports an offset from the seek point instead of the
/// source timestamp — a number that looks perfectly reasonable and is wrong by
/// however far the seek went. Both call sites share this list so neither can
/// drift from the other.
const REPORTING_FLAGS: [&str; 4] = ["-hide_banner", "-v", "info", "-copyts"];

/// Where to look when a decode at `index` produced **no picture at all**.
///
/// A container can end with a packet that carries no picture: a recorder that
/// stops mid-interval writes a trailing sample marked `AV_PKT_FLAG_DISCARD`, and
/// `nb_frames` counts it. The timeline built from that count then has one index
/// more than the media has pictures, and asking for that index decodes nothing —
/// not a short read, *nothing* (#324).
///
/// The honest answer is the last real picture, one nominal interval back. It is
/// also what the browser's reader already does by construction: mediabunny
/// answers at-or-**before** the requested instant, so it simply returns the
/// preceding picture and the phantom index is invisible there. Stepping back
/// here is what makes the two surfaces agree.
///
/// A step of exactly one interval, not a search: this is a container ending one
/// packet past its last picture, not an arbitrary gap. If the step still finds
/// nothing, that is a genuine failure and the caller reports it.
///
/// `None` at index 0, where there is nothing earlier to fall back to.
fn fallback_timestamp_seconds(index: u32, fps_num: u32, fps_den: u32) -> Option<f64> {
    let earlier = index.checked_sub(1)?;
    Some(f64::from(earlier) * f64::from(fps_den) / f64::from(fps_num.max(1)))
}

/// What a decoded frame should report, given what ffmpeg said about it.
///
/// Split out from the two decode paths because this is the decision the bug was
/// in: a requested index used to be taken on trust, and here it is only a
/// fallback for when ffmpeg said nothing (#319).
fn resolve_timing(
    reported: Option<(f64, f64)>,
    requested_index: u32,
    fps_num: u32,
    fps_den: u32,
    frame_count: u32,
) -> (u32, f64, f64) {
    match reported {
        Some((pts, duration)) => (index_at(pts, fps_num, fps_den, frame_count), pts, duration),
        // No report: keep doing what this always did, and say the duration is
        // unknown rather than inventing the nominal one.
        None => (
            requested_index,
            f64::from(requested_index) * f64::from(fps_den) / f64::from(fps_num.max(1)),
            0.0,
        ),
    }
}

/// Pulls `(presentation timestamp, duration)` out of one `showinfo` line.
///
/// This is ffmpeg stating what it actually decoded, which is the only way to
/// know: a raw video pipe carries pixels and no timing at all.
///
/// ```text
/// [Parsed_showinfo_1 @ …] n:0 pts:69120 pts_time:5.625 duration:512 duration_time:0.0416667 fmt:rgba …
/// ```
///
/// The duration is optional — a stream that does not declare one still yields a
/// usable timestamp, and `0.0` tells the caller to fall back rather than trust a
/// fabricated interval.
fn parse_showinfo(line: &str) -> Option<(f64, f64)> {
    let pts = showinfo_field(line, "pts_time:")?;
    Some((pts, showinfo_field(line, "duration_time:").unwrap_or(0.0)))
}

fn showinfo_field(line: &str, key: &str) -> Option<f64> {
    line.split_once(key)?
        .1
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

#[derive(Debug, Clone)]
pub enum NativeVideoSource {
    Local(PathBuf),
    Url(String),
}

impl NativeVideoSource {
    fn as_os_str(&self) -> &OsStr {
        match self {
            Self::Local(path) => path.as_os_str(),
            Self::Url(url) => OsStr::new(url),
        }
    }

    /// What the Source panel shows: a bare file name, or the URL itself.
    pub fn display_name(&self) -> String {
        match self {
            Self::Local(path) => path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("")
                .to_owned(),
            Self::Url(url) => url.clone(),
        }
    }
}

pub struct NativeVideo {
    source: NativeVideoSource,
    fps_num: u32,
    fps_den: u32,
    frame_count: u32,
    generation: Arc<AtomicU64>,
    receiver: Option<Receiver<Result<DecodedFrame, String>>>,
    pub width: u32,
    pub height: u32,
}

/// The preview size a `--preview-width` implies for a source.
///
/// **The one derivation.** Both the decode size ffmpeg is asked to scale to and
/// the render-target size the shell allocates come from here, so they cannot
/// disagree — the divergence #170 reports was two call sites computing this
/// separately and a third not computing it at all.
///
/// Clamped to the source width, so `--preview-width` only ever scales *down*;
/// the height follows the source aspect.
pub(crate) fn preview_size(info: &trd_core::VideoInfo, preview_width: u32) -> (u32, u32) {
    let width = preview_width.min(info.width.max(1)).max(1);
    let height = ((u64::from(width) * u64::from(info.height.max(1)))
        .div_ceil(u64::from(info.width.max(1)))) as u32;
    (width, height.max(1))
}

impl NativeVideo {
    pub fn open(
        source: NativeVideoSource,
        info: &trd_core::VideoInfo,
        preview_width: u32,
    ) -> Result<Self, NativeVideoEditingError> {
        if let NativeVideoSource::Local(path) = &source {
            validate_file(path, info)?;
        }
        validate_probe(&source, info)?;
        Ok(Self::with_timeline(source, info, preview_width))
    }

    /// Opens a video **without a document to match it against**, deriving the
    /// timeline from the container instead (#264).
    ///
    /// Validation exists to catch a document paired with the wrong cut; with no
    /// document there is nothing to disagree with, so ffprobe's answer *is* the
    /// timeline. Returns the derived [`VideoInfo`](trd_core::VideoInfo) so the
    /// editor can adopt it.
    pub fn probe(
        source: NativeVideoSource,
        preview_width: u32,
    ) -> Result<(Self, trd_core::VideoInfo), NativeVideoEditingError> {
        let info = probe_video_info(&source)?;
        let video = Self::with_timeline(source, &info, preview_width);
        Ok((video, info))
    }

    fn with_timeline(
        source: NativeVideoSource,
        info: &trd_core::VideoInfo,
        preview_width: u32,
    ) -> Self {
        let (width, height) = preview_size(info, preview_width);
        Self {
            source,
            fps_num: info.fps_num,
            fps_den: info.fps_den,
            frame_count: info.frame_count,
            generation: Arc::new(AtomicU64::new(0)),
            receiver: None,
            width,
            height: height.max(1),
        }
    }

    pub fn decode_one(&self, index: u32) -> Result<DecodedFrame, NativeVideoEditingError> {
        let index = index.min(self.frame_count.saturating_sub(1));
        let mut output = self.run_decode(&self.timestamp(index))?;
        if output.stdout.is_empty() {
            // Nothing at all came back — not a short read, *nothing*. That is
            // the phantom final index (#324), so ask again one frame earlier.
            if let Some(seconds) = fallback_timestamp_seconds(index, self.fps_num, self.fps_den) {
                output = self.run_decode(&format!("{seconds:.9}"))?;
            }
        }
        let expected = self.frame_bytes();
        if output.stdout.len() != expected {
            return Err(NativeVideoEditingError::FrameLength {
                index,
                actual: output.stdout.len(),
                expected,
            });
        }
        let reported = String::from_utf8_lossy(&output.stderr)
            .lines()
            .find_map(parse_showinfo);
        // The index is derived from the timestamp the frame carries, so a
        // fallback decode reports the picture it actually returned rather than
        // the phantom index that was asked for (#319, #324).
        let (index, media_time_seconds, duration_seconds) = resolve_timing(
            reported,
            index,
            self.fps_num,
            self.fps_den,
            self.frame_count,
        );
        Ok(DecodedFrame {
            index,
            media_time_seconds,
            duration_seconds,
            rgba: output.stdout,
        })
    }

    /// One `ffmpeg` invocation that seeks to `timestamp` and writes a single
    /// scaled RGBA frame to stdout, with `showinfo` on stderr.
    fn run_decode(&self, timestamp: &str) -> Result<Output, NativeVideoEditingError> {
        let output = Command::new("ffmpeg")
            .args(REPORTING_FLAGS)
            .args(["-ss", timestamp, "-i"])
            .arg(self.source.as_os_str())
            .args([
                "-frames:v",
                "1",
                "-vf",
                &self.reporting_filter(),
                "-pix_fmt",
                "rgba",
                "-f",
                "rawvideo",
                "-",
            ])
            .output()
            .map_err(|source| NativeVideoEditingError::Spawn {
                program: "ffmpeg",
                source,
            })?;
        if !output.status.success() {
            return Err(NativeVideoEditingError::Command {
                program: "ffmpeg",
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }
        Ok(output)
    }

    pub fn play_from(&mut self, index: u32) -> Result<(), NativeVideoEditingError> {
        self.stop();
        let index = index.min(self.frame_count.saturating_sub(1));
        let generation = self.generation.load(Ordering::SeqCst);
        let generation_counter = self.generation.clone();
        let source = self.source.clone();
        let timestamp = self.timestamp(index);
        let scale = self.reporting_filter();
        let frame_bytes = self.frame_bytes();
        let frame_count = self.frame_count;
        let fps_num = self.fps_num;
        let fps_den = self.fps_den;
        let (sender, receiver) = mpsc::sync_channel(2);
        std::thread::Builder::new()
            .name("trd-native-video-decoder".to_owned())
            .spawn(move || {
                stream_frames(
                    source,
                    timestamp,
                    scale,
                    index,
                    frame_count,
                    frame_bytes,
                    fps_num,
                    fps_den,
                    generation,
                    generation_counter,
                    sender,
                );
            })
            .map_err(|source| NativeVideoEditingError::Spawn {
                program: "decoder thread",
                source,
            })?;
        self.receiver = Some(receiver);
        Ok(())
    }

    pub fn try_frame(&mut self) -> Option<Result<DecodedFrame, String>> {
        let receiver = self.receiver.as_ref()?;
        match receiver.try_recv() {
            Ok(frame) => Some(frame),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                self.receiver = None;
                None
            }
        }
    }

    pub fn is_streaming(&self) -> bool {
        self.receiver.is_some()
    }

    pub fn stop(&mut self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.receiver = None;
    }

    fn timestamp(&self, index: u32) -> String {
        format!("{:.9}", self.timestamp_seconds(index))
    }

    /// Where the nominal grid says frame `index` sits. Only ever a *request* —
    /// what came back is read from the frame itself (#319).
    fn timestamp_seconds(&self, index: u32) -> f64 {
        f64::from(index) * f64::from(self.fps_den) / f64::from(self.fps_num)
    }

    fn scale_filter(&self) -> String {
        format!("scale={}:{}", self.width, self.height)
    }

    /// The scale filter with `showinfo` behind it, so every frame ffmpeg emits
    /// is accompanied by the timestamp it was emitted for.
    fn reporting_filter(&self) -> String {
        format!("{},showinfo", self.scale_filter())
    }

    fn frame_bytes(&self) -> usize {
        self.width as usize * self.height as usize * 4
    }
}

impl Drop for NativeVideo {
    fn drop(&mut self) {
        self.stop();
    }
}

#[allow(clippy::too_many_arguments)]
fn stream_frames(
    source: NativeVideoSource,
    timestamp: String,
    scale: String,
    start_index: u32,
    frame_count: u32,
    frame_bytes: usize,
    fps_num: u32,
    fps_den: u32,
    generation: u64,
    generation_counter: Arc<AtomicU64>,
    sender: SyncSender<Result<DecodedFrame, String>>,
) {
    let spawned = Command::new("ffmpeg")
        // See `decode_one`: `-copyts` keeps `showinfo` reporting source
        // timestamps rather than an offset from the seek point.
        .args(REPORTING_FLAGS)
        .args(["-ss", &timestamp, "-i"])
        .arg(source.as_os_str())
        .args(["-vf", &scale, "-pix_fmt", "rgba", "-f", "rawvideo", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match spawned {
        Ok(child) => child,
        Err(error) => {
            let _ = sender.send(Err(format!("failed to start ffmpeg: {error}")));
            return;
        }
    };
    let Some(mut stdout) = child.stdout.take() else {
        let _ = sender.send(Err("ffmpeg stdout was not piped".to_owned()));
        let _ = child.kill();
        let _ = child.wait();
        return;
    };

    // stderr must be drained *while* stdout is read, not after the process
    // exits. `showinfo` emits a line per frame, so leaving it in the pipe would
    // fill the buffer and wedge ffmpeg part-way through a long playback — a
    // deadlock the old `-v error` invocation never produced only because it had
    // almost nothing to say.
    let (timings, captured) = spawn_stderr_reader(child.stderr.take());

    for offset in 0.. {
        let counted = start_index.saturating_add(offset);
        if counted >= frame_count {
            break;
        }
        let mut rgba = vec![0; frame_bytes];
        if let Err(error) = stdout.read_exact(&mut rgba) {
            if generation_counter.load(Ordering::SeqCst) == generation
                && error.kind() != std::io::ErrorKind::UnexpectedEof
            {
                let _ = sender.send(Err(format!("ffmpeg frame read failed: {error}")));
            }
            break;
        }
        // One `showinfo` line per emitted frame, in order, so the timings are
        // consumed one-for-one. A frame that arrives without one still plays;
        // it just falls back to counting, which is what this used to do always.
        let reported = timings
            .as_ref()
            .and_then(|rx| rx.recv_timeout(TIMING_WAIT).ok());
        let (index, media_time_seconds, duration_seconds) =
            resolve_timing(reported, counted, fps_num, fps_den, frame_count);
        if generation_counter.load(Ordering::SeqCst) != generation
            || sender
                .send(Ok(DecodedFrame {
                    index,
                    media_time_seconds,
                    duration_seconds,
                    rgba,
                }))
                .is_err()
        {
            break;
        }
    }

    if generation_counter.load(Ordering::SeqCst) != generation {
        let _ = child.kill();
    }
    let status = child.wait();
    if generation_counter.load(Ordering::SeqCst) == generation {
        match status {
            Ok(status) if !status.success() => {
                let stderr = captured
                    .and_then(|handle| handle.join().ok())
                    .unwrap_or_default();
                let _ = sender.send(Err(format!("ffmpeg exited with {status}: {stderr}")));
            }
            Err(error) => {
                let _ = sender.send(Err(format!("failed to wait for ffmpeg: {error}")));
            }
            _ => {}
        }
    }
}

/// How long a frame waits for its `showinfo` line before falling back to
/// counting. Generous, because it only elapses when ffmpeg is not reporting at
/// all — a frame's line is written before the frame itself reaches stdout.
const TIMING_WAIT: std::time::Duration = std::time::Duration::from_millis(500);

/// Drains ffmpeg's stderr on its own thread, splitting it into per-frame timings
/// and everything else — the latter kept for the error message.
#[allow(clippy::type_complexity)]
fn spawn_stderr_reader(
    stderr: Option<std::process::ChildStderr>,
) -> (
    Option<Receiver<(f64, f64)>>,
    Option<std::thread::JoinHandle<String>>,
) {
    let Some(pipe) = stderr else {
        return (None, None);
    };
    let (sender, receiver) = mpsc::channel();
    let handle = std::thread::Builder::new()
        .name("trd-native-video-stderr".to_owned())
        .spawn(move || {
            let mut rest = String::new();
            for line in std::io::BufReader::new(pipe).lines().map_while(Result::ok) {
                match parse_showinfo(&line) {
                    // Ignore a closed receiver rather than stopping: the pipe
                    // still has to be drained or ffmpeg blocks on it.
                    Some(timing) => {
                        let _ = sender.send(timing);
                    }
                    None => {
                        rest.push_str(&line);
                        rest.push('\n');
                    }
                }
            }
            rest
        })
        .ok();
    (Some(receiver), handle)
}

/// The frame a reported timestamp belongs to, mapped exactly as the browser maps
/// its own, so both surfaces number frames the same way.
fn index_at(media_time_seconds: f64, fps_num: u32, fps_den: u32, frame_count: u32) -> u32 {
    trd_gui::video_editing::frame_index_at_media_time(
        media_time_seconds,
        fps_num,
        fps_den,
        frame_count,
    )
}

fn validate_file(path: &Path, info: &trd_core::VideoInfo) -> Result<(), NativeVideoEditingError> {
    let actual_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if actual_name != info.source_name {
        return Err(NativeVideoEditingError::SourceMismatch(format!(
            "expected filename {}, got {actual_name}",
            info.source_name
        )));
    }
    let metadata = std::fs::metadata(path).map_err(|source| NativeVideoEditingError::Read {
        path: path.display().to_string(),
        source,
    })?;
    if metadata.len() != info.byte_length {
        return Err(NativeVideoEditingError::SourceMismatch(format!(
            "expected {} bytes, got {}",
            info.byte_length,
            metadata.len()
        )));
    }
    Ok(())
}

/// Reads a video's own timeline with `ffprobe` — the native counterpart of the
/// browser's `mp4_probe`, and the source of truth when no document exists.
///
/// The rate is kept as the container's **rational** (`r_frame_rate`), so 29.97
/// stays `30000/1001`. A missing `nb_frames` (common for streamed inputs) is
/// derived from duration × rate rather than refused: a frame count that is a
/// frame or two off still plays, while refusing to open does not.
fn probe_video_info(
    source: &NativeVideoSource,
) -> Result<trd_core::VideoInfo, NativeVideoEditingError> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,duration,r_frame_rate",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(source.as_os_str())
        .output()
        .map_err(|source| NativeVideoEditingError::Spawn {
            program: "ffprobe",
            source,
        })?;
    if !output.status.success() {
        return Err(NativeVideoEditingError::Command {
            program: "ffprobe",
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let fields: HashMap<_, _> = text
        .lines()
        .filter_map(|line| line.split_once('='))
        .collect();
    let field = |name: &'static str| -> Result<&str, NativeVideoEditingError> {
        fields
            .get(name)
            .copied()
            .ok_or(NativeVideoEditingError::ProbeField(name))
    };
    let parse_u32 = |name: &'static str| -> Result<u32, NativeVideoEditingError> {
        let value = field(name)?;
        value
            .parse()
            .map_err(|_| NativeVideoEditingError::ProbeValue {
                field: name,
                value: value.to_owned(),
            })
    };

    let width = parse_u32("width")?;
    let height = parse_u32("height")?;
    let rate = field("r_frame_rate")?;
    let (fps_num, fps_den) = rate
        .split_once('/')
        .and_then(|(num, den)| Some((num.parse::<u32>().ok()?, den.parse::<u32>().ok()?)))
        .filter(|(num, den)| *num > 0 && *den > 0)
        .ok_or_else(|| NativeVideoEditingError::ProbeValue {
            field: "r_frame_rate",
            value: rate.to_owned(),
        })?;
    let duration_seconds = fields
        .get("duration")
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0);
    let frame_count = parse_u32("nb_frames").unwrap_or_else(|_| {
        (duration_seconds * f64::from(fps_num) / f64::from(fps_den)).round() as u32
    });

    Ok(trd_core::VideoInfo {
        source_name: source.display_name(),
        mime: String::new(),
        codec: field("codec_name").unwrap_or_default().to_owned(),
        // Identity fields a document would carry: unknown here, and not needed —
        // there is no document to match against.
        sha256: String::new(),
        byte_length: match source {
            NativeVideoSource::Local(path) => std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
            NativeVideoSource::Url(_) => 0,
        },
        width,
        height,
        fps_num,
        fps_den,
        frame_count: frame_count.max(1),
        duration_us: (duration_seconds * 1_000_000.0) as i64,
    })
}

fn validate_probe(
    source: &NativeVideoSource,
    info: &trd_core::VideoInfo,
) -> Result<(), NativeVideoEditingError> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,duration",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(source.as_os_str())
        .output()
        .map_err(|source| NativeVideoEditingError::Spawn {
            program: "ffprobe",
            source,
        })?;
    if !output.status.success() {
        return Err(NativeVideoEditingError::Command {
            program: "ffprobe",
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let fields: HashMap<_, _> = text
        .lines()
        .filter_map(|line| line.split_once('='))
        .collect();
    let parse_u32 = |field: &'static str| -> Result<u32, NativeVideoEditingError> {
        let value = fields
            .get(field)
            .ok_or(NativeVideoEditingError::ProbeField(field))?;
        value
            .parse()
            .map_err(|_| NativeVideoEditingError::ProbeValue {
                field,
                value: (*value).to_owned(),
            })
    };
    let width = parse_u32("width")?;
    let height = parse_u32("height")?;
    if (width, height) != (info.width, info.height) {
        return Err(NativeVideoEditingError::SourceMismatch(format!(
            "expected {}x{}, got {width}x{height}",
            info.width, info.height
        )));
    }
    let codec = fields
        .get("codec_name")
        .copied()
        .ok_or(NativeVideoEditingError::ProbeField("codec_name"))?;
    if codec != info.codec {
        return Err(NativeVideoEditingError::SourceMismatch(format!(
            "expected codec {}, got {codec}",
            info.codec
        )));
    }
    if let Some(value) = fields
        .get("nb_frames")
        .copied()
        .filter(|value| *value != "N/A")
    {
        let frames: u32 = value
            .parse()
            .map_err(|_| NativeVideoEditingError::ProbeValue {
                field: "nb_frames",
                value: value.to_owned(),
            })?;
        if frames != info.frame_count {
            return Err(NativeVideoEditingError::SourceMismatch(format!(
                "expected {} frames, got {frames}",
                info.frame_count
            )));
        }
    }
    let duration_value = fields
        .get("duration")
        .copied()
        .ok_or(NativeVideoEditingError::ProbeField("duration"))?;
    let duration: f64 =
        duration_value
            .parse()
            .map_err(|_| NativeVideoEditingError::ProbeValue {
                field: "duration",
                value: duration_value.to_owned(),
            })?;
    let expected_duration =
        f64::from(info.frame_count) * f64::from(info.fps_den) / f64::from(info.fps_num);
    let frame_duration = f64::from(info.fps_den) / f64::from(info.fps_num);
    if (duration - expected_duration).abs() > frame_duration {
        return Err(NativeVideoEditingError::SourceMismatch(format!(
            "expected {expected_duration:.3}s duration, got {duration:.3}s"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(width: u32, height: u32) -> trd_core::VideoInfo {
        trd_core::VideoInfo {
            source_name: String::new(),
            mime: String::new(),
            codec: String::new(),
            sha256: String::new(),
            byte_length: 0,
            width,
            height,
            fps_num: 30,
            fps_den: 1,
            frame_count: 1,
            duration_us: 0,
        }
    }

    #[test]
    fn preview_width_only_ever_scales_down() {
        // Clamped to the source, so asking for more than the video has is a
        // no-op rather than an upscale.
        assert_eq!(preview_size(&info(1920, 1080), 960), (960, 540));
        assert_eq!(preview_size(&info(1920, 1080), 1920), (1920, 1080));
        assert_eq!(preview_size(&info(640, 360), 1920), (640, 360));
    }

    /// A real `showinfo` line, verbatim, including the ragged column padding
    /// ffmpeg emits — the parser has to read fields, not offsets.
    const SHOWINFO: &str = "[Parsed_showinfo_1 @ 000002236ab332c0] n:   0 pts:  69120 \
        pts_time:5.625   duration:    512 duration_time:0.0416667 fmt:rgba cl:unspecified \
        sar:1/1 s:960x540 i:P iskey:0 type:B checksum:7D68F5B2";

    #[test]
    fn a_showinfo_line_yields_the_timestamp_and_duration() {
        assert_eq!(parse_showinfo(SHOWINFO), Some((5.625, 0.0416667)));
    }

    /// `pts_time:` has to win over the `pts:` that precedes it on the same line;
    /// matching the shorter key first would read the raw timescale value as
    /// seconds and put the frame hours away.
    #[test]
    fn the_raw_pts_field_is_not_mistaken_for_the_timestamp() {
        let (pts, _) = parse_showinfo(SHOWINFO).expect("parsed");
        assert!(
            (pts - 5.625).abs() < 1e-9,
            "read {pts}, which looks like the raw pts field"
        );
    }

    #[test]
    fn a_line_without_a_duration_still_yields_its_timestamp() {
        assert_eq!(
            parse_showinfo("[Parsed_showinfo_1 @ 0x0] n:1 pts:512 pts_time:1.5 fmt:rgba"),
            Some((1.5, 0.0)),
            "a missing duration must fall back, not discard the timestamp"
        );
    }

    #[test]
    fn other_ffmpeg_output_is_not_a_timing_line() {
        assert_eq!(
            parse_showinfo("frame=   72 fps= 42 q=-0.0 Lsize=8315KiB"),
            None
        );
        assert_eq!(parse_showinfo(""), None);
        assert_eq!(
            parse_showinfo("[Parsed_showinfo_1 @ 0x0] n:1 pts_time:N/A"),
            None,
            "an unparseable timestamp is not a timing line"
        );
    }

    /// The timing of the variable-rate recording #319 was found on: 60 fps
    /// nominal, 283 678 frames, `stts` alternating 272/256 units at 16 kHz.
    const VFR: (u32, u32, u32) = (60, 1, 283_678);

    /// The heart of the fix. Asking for frame 200000 puts the request at
    /// `200000/60 = 3333.333333 s`, a third of a millisecond after the real
    /// timestamp `3333.333`, so ffmpeg answers with the *next* picture at
    /// `3333.350`. That picture is frame 200001 and must be reported as such —
    /// filing it under the requested index is the silent placement error.
    #[test]
    fn a_reported_timestamp_decides_the_index_not_the_request() {
        let (num, den, count) = VFR;

        let (index, media_time, duration) =
            resolve_timing(Some((3333.350, 0.017)), 200_000, num, den, count);

        assert_eq!(
            index, 200_001,
            "the frame reported must be the one received"
        );
        assert_eq!(
            media_time, 3333.350,
            "the timestamp must be ffmpeg's verbatim"
        );
        assert_eq!(
            duration, 0.017,
            "the frame's own window, not the nominal one"
        );
    }

    /// The second measured mislabel, at the far end of the same file — one
    /// sample could be a coincidence of where that seek happened to land.
    #[test]
    fn the_same_holds_at_the_far_end_of_a_long_timeline() {
        let (num, den, count) = VFR;

        let (index, media_time, _) =
            resolve_timing(Some((4575.350, 0.017)), 274_520, num, den, count);

        assert_eq!(index, 274_521);
        assert_eq!(media_time, 4575.350);
    }

    /// A request that *does* land keeps its index, so this is not a blanket
    /// off-by-one: two of the four measured probes on that file were exact.
    #[test]
    fn a_request_that_lands_keeps_its_index() {
        let (num, den, count) = VFR;

        let (index, ..) = resolve_timing(Some((1597.300, 0.017)), 95_838, num, den, count);

        assert_eq!(index, 95_838);
    }

    /// A constant-rate source has no divergence to resolve — the nominal grid
    /// *is* the timestamps — which is why every existing fixture passed while
    /// this bug was live.
    #[test]
    fn a_constant_rate_source_reports_exactly_what_was_asked_for() {
        let (index, media_time, duration) =
            resolve_timing(Some((5.625, 1.0 / 24.0)), 135, 24, 1, 288);

        assert_eq!(index, 135);
        assert_eq!(media_time, 5.625);
        assert!((duration - 1.0 / 24.0).abs() < 1e-9);
    }

    /// ffmpeg saying nothing must not break playback: the frame still arrives,
    /// falling back to the request — which is what this did unconditionally
    /// before — and declares its duration unknown rather than inventing one.
    #[test]
    fn without_a_report_the_request_stands_and_the_duration_is_unknown() {
        let (index, media_time, duration) = resolve_timing(None, 135, 24, 1, 288);

        assert_eq!(index, 135);
        assert!((media_time - 5.625).abs() < 1e-9);
        assert_eq!(
            duration, 0.0,
            "an unknown duration must be reported as unknown, so the caller falls back"
        );
    }

    /// #324, with the numbers measured on the 217.77 GiB recording: it declares
    /// `nb_frames = 694840`, so the timeline offers indices `0..=694839` — but
    /// its last packet is marked `AV_PKT_FLAG_DISCARD` and carries no picture,
    /// and the last frame any decoder emits is **694838** at 27793.52 s.
    ///
    /// Asking for the phantom index therefore decodes nothing, and the answer is
    /// the picture one interval back — which lands exactly on that frame.
    #[test]
    fn the_phantom_final_index_falls_back_to_the_last_real_picture() {
        let phantom = 694_839;
        let fallback = fallback_timestamp_seconds(phantom, 25, 1).expect("not the first frame");

        assert!(
            (fallback - 27793.52).abs() < 1e-9,
            "must land on the last decodable picture (694838 @ 27793.52 s), got {fallback}"
        );
    }

    /// Not a search — one interval, once. A container that ends one packet past
    /// its last picture needs exactly one step; anything further would be
    /// guessing at how much of the tail is missing, and a second empty decode is
    /// a genuine failure the caller has to report.
    #[test]
    fn the_step_is_one_nominal_interval_whatever_the_rate() {
        let at_24 = fallback_timestamp_seconds(287, 24, 1).expect("not the first frame");
        assert!((at_24 - 286.0 / 24.0).abs() < 1e-9);

        let at_60 = fallback_timestamp_seconds(100, 60, 1).expect("not the first frame");
        assert!((at_60 - 99.0 / 60.0).abs() < 1e-9);
    }

    /// The first frame has nothing behind it, so an empty decode there is a
    /// plain failure rather than something to step back from.
    #[test]
    fn the_first_frame_has_nothing_to_fall_back_to() {
        assert_eq!(fallback_timestamp_seconds(0, 25, 1), None);
    }

    /// `-copyts` is the difference between a source timestamp and an offset from
    /// the seek point, and dropping it fails *silently* — every reported time
    /// would simply be shifted. Worth pinning by name.
    #[test]
    fn the_reporting_flags_keep_timestamps_in_the_source_clock() {
        assert!(
            REPORTING_FLAGS.contains(&"-copyts"),
            "without -copyts, showinfo reports an offset from the seek point"
        );
        assert!(
            REPORTING_FLAGS.contains(&"info"),
            "showinfo logs at info level; a quieter level emits no timings at all"
        );
    }

    /// The reporter has to sit *behind* the scaler in one chain, or there is
    /// nothing to pair frames with.
    #[test]
    fn the_filter_chain_carries_the_reporter() {
        let video = NativeVideo {
            source: NativeVideoSource::Local(PathBuf::new()),
            fps_num: 24,
            fps_den: 1,
            frame_count: 288,
            generation: Arc::new(AtomicU64::new(0)),
            receiver: None,
            width: 960,
            height: 540,
        };

        assert_eq!(video.reporting_filter(), "scale=960:540,showinfo");
    }

    #[test]
    fn height_follows_the_source_aspect() {
        assert_eq!(preview_size(&info(1440, 1080), 720), (720, 540));
        // Rounds up rather than to zero, so a preview is never degenerate.
        assert_eq!(preview_size(&info(1000, 3), 100), (100, 1));
    }

    #[test]
    fn degenerate_sources_still_yield_a_usable_target() {
        assert_eq!(preview_size(&info(0, 0), 960), (1, 1));
        assert_eq!(preview_size(&info(1920, 1080), 0), (1, 1));
    }

    #[test]
    fn the_decoded_size_and_the_derived_preview_size_are_the_same_number() {
        // The #170 divergence: the shell sized its render target from one
        // derivation while ffmpeg decoded to another. Both now come from
        // `preview_size`, so this can only fail if a caller stops using it.
        let info = info(1920, 1080);
        let video = NativeVideo::with_timeline(
            NativeVideoSource::Local(PathBuf::from("unused.mp4")),
            &info,
            960,
        );
        assert_eq!((video.width, video.height), preview_size(&info, 960));
    }
}
