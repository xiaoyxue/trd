use std::collections::HashMap;
use std::ffi::OsStr;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::sync::Arc;

use crate::error::NativeVideoEditingError;

pub struct DecodedFrame {
    pub index: u32,
    pub rgba: Vec<u8>,
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

impl NativeVideo {
    pub fn open(
        source: NativeVideoSource,
        info: &trd_core::VideoInfo,
        preview_width: u32,
    ) -> Result<Self, NativeVideoEditingError> {
        // TOY BRANCH: the document/source coupling is off so any video can be
        // played. Strict validation is still available behind `--strict-source`.
        if strict_source() {
            if let NativeVideoSource::Local(path) = &source {
                validate_file(path, info)?;
            }
            validate_probe(&source, info)?;
        }
        let width = preview_width.min(info.width).max(1);
        let height =
            ((u64::from(width) * u64::from(info.height)).div_ceil(u64::from(info.width))) as u32;
        Ok(Self {
            source,
            fps_num: info.fps_num,
            fps_den: info.fps_den,
            frame_count: info.frame_count,
            generation: Arc::new(AtomicU64::new(0)),
            receiver: None,
            width,
            height,
        })
    }

    pub fn decode_one(&self, index: u32) -> Result<DecodedFrame, NativeVideoEditingError> {
        let index = index.min(self.frame_count.saturating_sub(1));
        let output = Command::new("ffmpeg")
            .args(["-v", "error", "-ss", &self.timestamp(index), "-i"])
            .arg(self.source.as_os_str())
            .args([
                "-frames:v",
                "1",
                "-vf",
                &self.scale_filter(),
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
        let expected = self.frame_bytes();
        if output.stdout.len() != expected {
            return Err(NativeVideoEditingError::FrameLength {
                index,
                actual: output.stdout.len(),
                expected,
            });
        }
        Ok(DecodedFrame {
            index,
            rgba: output.stdout,
        })
    }

    pub fn play_from(&mut self, index: u32) -> Result<(), NativeVideoEditingError> {
        self.stop();
        let index = index.min(self.frame_count.saturating_sub(1));
        let generation = self.generation.load(Ordering::SeqCst);
        let generation_counter = self.generation.clone();
        let source = self.source.clone();
        let timestamp = self.timestamp(index);
        let scale = self.scale_filter();
        let frame_bytes = self.frame_bytes();
        let frame_count = self.frame_count;
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
        let seconds = f64::from(index) * f64::from(self.fps_den) / f64::from(self.fps_num);
        format!("{seconds:.9}")
    }

    fn scale_filter(&self) -> String {
        format!("scale={}:{}", self.width, self.height)
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
    generation: u64,
    generation_counter: Arc<AtomicU64>,
    sender: SyncSender<Result<DecodedFrame, String>>,
) {
    let spawned = Command::new("ffmpeg")
        .args(["-v", "error", "-ss", &timestamp, "-i"])
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

    for index in start_index..frame_count {
        let mut rgba = vec![0; frame_bytes];
        if let Err(error) = stdout.read_exact(&mut rgba) {
            if generation_counter.load(Ordering::SeqCst) == generation
                && error.kind() != std::io::ErrorKind::UnexpectedEof
            {
                let _ = sender.send(Err(format!("ffmpeg frame read failed: {error}")));
            }
            break;
        }
        if generation_counter.load(Ordering::SeqCst) != generation
            || sender.send(Ok(DecodedFrame { index, rgba })).is_err()
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
                let mut stderr = String::new();
                if let Some(mut pipe) = child.stderr.take() {
                    let _ = pipe.read_to_string(&mut stderr);
                }
                let _ = sender.send(Err(format!("ffmpeg exited with {status}: {stderr}")));
            }
            Err(error) => {
                let _ = sender.send(Err(format!("failed to wait for ffmpeg: {error}")));
            }
            _ => {}
        }
    }
}

/// TOY BRANCH: strict document/source coupling, off by default.
///
/// Set `TRD_STRICT_SOURCE=1` to restore the original behaviour (filename, byte
/// length, codec, resolution, frame count and duration must all match the
/// document).
pub fn strict_source() -> bool {
    std::env::var_os("TRD_STRICT_SOURCE").is_some_and(|v| v != "0")
}

/// TOY BRANCH: derives a [`trd_core::VideoInfo`] straight from `ffprobe`, so a
/// video can be opened without a matching timeline document.
pub fn probe_video_info(
    source: &NativeVideoSource,
) -> Result<trd_core::VideoInfo, NativeVideoEditingError> {
    let fields = probe_fields(source)?;

    let get = |field: &'static str| -> Result<&str, NativeVideoEditingError> {
        fields
            .get(field)
            .map(String::as_str)
            .filter(|value| *value != "N/A")
            .ok_or(NativeVideoEditingError::ProbeField(field))
    };
    let parse_u32 = |field: &'static str| -> Result<u32, NativeVideoEditingError> {
        let value = get(field)?;
        value
            .parse()
            .map_err(|_| NativeVideoEditingError::ProbeValue {
                field,
                value: value.to_owned(),
            })
    };

    let width = parse_u32("width")?;
    let height = parse_u32("height")?;
    let codec = get("codec_name").unwrap_or("unknown").to_owned();

    // `r_frame_rate` is a rational such as `30000/1001`.
    let rate = get("r_frame_rate").unwrap_or("25/1");
    let (fps_num, fps_den) = rate
        .split_once('/')
        .and_then(|(n, d)| Some((n.parse::<u32>().ok()?, d.parse::<u32>().ok()?)))
        .filter(|(n, d)| *n > 0 && *d > 0)
        .unwrap_or((25, 1));

    let duration_seconds = get("duration")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0);

    // `nb_frames` is absent in many containers; fall back to duration * fps.
    let frame_count = get("nb_frames")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .or_else(|| {
            duration_seconds
                .map(|seconds| (seconds * f64::from(fps_num) / f64::from(fps_den)).round() as u32)
        })
        .filter(|count| *count > 0)
        .unwrap_or(1);

    let duration_us = duration_seconds.map_or_else(
        || i64::from(frame_count) * i64::from(fps_den) * 1_000_000 / i64::from(fps_num.max(1)),
        |seconds| (seconds * 1_000_000.0) as i64,
    );

    let (source_name, byte_length) = match source {
        NativeVideoSource::Local(path) => (
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("video")
                .to_owned(),
            std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
        ),
        NativeVideoSource::Url(url) => (url.clone(), 0),
    };

    Ok(trd_core::VideoInfo {
        source_name,
        mime: "video/mp4".to_owned(),
        codec,
        sha256: String::new(),
        byte_length,
        width,
        height,
        fps_num,
        fps_den,
        frame_count,
        duration_us,
    })
}

/// TOY BRANCH: a video-only timeline covering every frame of `video`, so the
/// editor has a document to drive playback when none was supplied.
pub fn synthesize_document(video: trd_core::VideoInfo) -> trd_core::VideoEditingDocument {
    let frames = (0..video.frame_count)
        .map(|index| trd_core::VideoEditingFrame {
            video_frame_index: index,
            present_index: index,
            timestamp_us: i64::from(index) * i64::from(video.fps_den) * 1_000_000
                / i64::from(video.fps_num.max(1)),
            k: None,
            placement_quad: None,
            tracked: false,
        })
        .collect();
    trd_core::VideoEditingDocument {
        video,
        poster_bytes: Vec::new(),
        frames,
    }
}

/// The `ffprobe` key/value output for the first video stream.
fn probe_fields(
    source: &NativeVideoSource,
) -> Result<HashMap<String, String>, NativeVideoEditingError> {
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
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect())
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
