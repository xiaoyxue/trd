//! What a video *is*, as far as trd is concerned.
//!
//! [`VideoTiming`] is the part a container can be asked for; [`VideoInfo`] is the
//! full description an authoring document carries. Two named conversions bridge
//! them ([`VideoInfo::from_probe`], [`VideoInfo::timing`]) so the mapping is
//! stated once instead of being hand-copied at each call site.

/// The facts that place a clip on a timeline — everything
/// [`probe_moov`](super::probe_moov) can recover from a container.
///
/// The frame rate is an exact **rational** rather than a float because the
/// editor numbers frames with it: 29.97 is `30000/1001`, and rounding it turns a
/// 250-frame clip into a 300-frame one (#264).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VideoTiming {
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    /// Total samples in the video track — the real frame count.
    pub frame_count: u32,
    pub duration_us: i64,
    /// Trailing samples the container stores but **never presents**.
    ///
    /// `frame_count` counts what the track *stores*, and a container can store a
    /// sample that is not a picture: a recorder stopping mid-interval writes a
    /// final sample outside the presentation window, which no decoder ever
    /// outputs (#324).
    ///
    /// Reported rather than subtracted. The count is not wrong — the file really
    /// does store that sample — so the honest thing is to say that one of them
    /// is not a picture, instead of silently shortening the timeline and leaving
    /// the number unexplained.
    ///
    /// `None` when the shell did not determine it, which is **not** the same as
    /// `Some(0)`: one means "not checked", the other "checked, there are none".
    /// Collapsing the two is the ambiguity this field exists to avoid.
    pub unpresented_tail: Option<UnpresentedTail>,
}

/// Trailing samples a container stores but never presents, **and the evidence
/// they were counted from**.
///
/// The two delivery surfaces answer this question about the same container by
/// different means, so a bare count does not say what was actually consulted —
/// and naming the wrong mechanism sends the next investigation looking for a
/// flag its code path never reads (#331).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnpresentedTail {
    pub samples: u32,
    pub evidence: UnpresentedTailEvidence,
}

/// What an [`UnpresentedTail`] was established from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnpresentedTailEvidence {
    /// The container's own sample tables (`stts` + `ctts`) walked against the
    /// track duration — [`probe_moov`](super::probe_moov), which is what the
    /// browser reaches.
    SampleTable,
    /// `AV_PKT_FLAG_DISCARD` on the trailing packets, as ffprobe reports it —
    /// the native adapter.
    PacketFlags,
}

impl UnpresentedTailEvidence {
    pub const fn label(self) -> &'static str {
        match self {
            Self::SampleTable => "stts",
            Self::PacketFlags => "AV_PKT_FLAG_DISCARD",
        }
    }
}

impl std::fmt::Display for UnpresentedTailEvidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// A clip as an authoring document describes it: the same timing a container
/// reports, plus the provenance that identifies *which* file it belongs to.
///
/// The timing fields are held **flat** rather than as a nested [`VideoTiming`].
/// This is a record read field-wise by the editor UI in ~90 places; nesting
/// would rename every one of those without making any of them clearer. What
/// mattered — that the two types were converted by hand, field by field, at a
/// call site — is fixed by [`from_probe`](Self::from_probe) and
/// [`timing`](Self::timing) instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoInfo {
    pub source_name: String,
    pub mime: String,
    pub codec: String,
    pub sha256: String,
    pub byte_length: u64,
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub frame_count: u32,
    pub duration_us: i64,
    /// See [`VideoTiming::unpresented_tail`].
    pub unpresented_tail: Option<UnpresentedTail>,
}

impl VideoInfo {
    /// The description a probed container implies: real timing, and provenance
    /// left blank because the bytes cannot supply it.
    ///
    /// Says "probed, so provenance is unknown" once, here, rather than as empty
    /// strings and zeros at each shell that probes.
    pub fn from_probe(timing: VideoTiming, source_name: String) -> Self {
        Self {
            source_name,
            mime: String::new(),
            codec: String::new(),
            sha256: String::new(),
            byte_length: 0,
            width: timing.width,
            height: timing.height,
            fps_num: timing.fps_num,
            fps_den: timing.fps_den,
            frame_count: timing.frame_count,
            duration_us: timing.duration_us,
            unpresented_tail: timing.unpresented_tail,
        }
    }

    /// This clip's timing on its own, for callers that need the timeline and not
    /// the provenance.
    pub fn timing(&self) -> VideoTiming {
        VideoTiming {
            width: self.width,
            height: self.height,
            fps_num: self.fps_num,
            fps_den: self.fps_den,
            frame_count: self.frame_count,
            duration_us: self.duration_us,
            unpresented_tail: self.unpresented_tail,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_and_document_agree_on_timing() {
        let timing = VideoTiming {
            width: 1920,
            height: 1080,
            fps_num: 30_000,
            fps_den: 1001,
            frame_count: 250,
            duration_us: 8_341_667,
            unpresented_tail: Some(UnpresentedTail {
                samples: 1,
                evidence: UnpresentedTailEvidence::SampleTable,
            }),
        };
        // A probed container carries real timing and no provenance...
        let probed = VideoInfo::from_probe(timing, "shot.mp4".to_owned());
        assert_eq!(probed.source_name, "shot.mp4");
        assert!(probed.mime.is_empty() && probed.codec.is_empty() && probed.sha256.is_empty());
        assert_eq!(probed.byte_length, 0);
        // ...and reading the timing back out is lossless, which is what lets one
        // document and one probe describe the same clip (#264).
        assert_eq!(probed.timing(), timing);
    }
}
