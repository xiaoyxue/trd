//! [`SessionState`] — the `Open → Finished | Failed` lifecycle both the input and
//! the output side of the protocol obey.
//!
//! Each direction re-declared this enum with the same three states and the same
//! guard. The rule is the same on both: a session is usable only while `Open`,
//! any error is terminal (a half-written batch must never be handed back as
//! success-shaped bytes, and a decoder that lost its framing cannot resynchronise),
//! and `finish` is the one legal way out. Only the *error type* differs, which is
//! why the guard takes the two errors as values.

/// Whether a session may still be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum SessionState {
    /// Accepting work.
    #[default]
    Open,
    /// Closed normally by `finish`.
    Finished,
    /// Closed by an error; every later call fails the same way.
    Failed,
}

impl SessionState {
    /// `Ok(())` while open, else the caller's own "already finished" /
    /// "previously failed" error.
    pub(crate) fn ensure_open<E>(self, finished: E, failed: E) -> Result<(), E> {
        match self {
            SessionState::Open => Ok(()),
            SessionState::Finished => Err(finished),
            SessionState::Failed => Err(failed),
        }
    }

    /// Records `result`'s outcome — success closes the session, failure poisons
    /// it — and hands the result back.
    pub(crate) fn close<T, E>(&mut self, result: Result<T, E>) -> Result<T, E> {
        *self = if result.is_ok() {
            SessionState::Finished
        } else {
            SessionState::Failed
        };
        result
    }

    /// Poisons the session and returns `error`, for a failure that is not a
    /// `finish`.
    pub(crate) fn fail<T, E>(&mut self, error: E) -> Result<T, E> {
        *self = SessionState::Failed;
        Err(error)
    }
}
