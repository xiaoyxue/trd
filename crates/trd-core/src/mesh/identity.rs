//! Opaque GPU mesh identity and the distinct wire-table row type.

use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_MESH_ID: AtomicU64 = AtomicU64::new(1);

/// A non-owning mesh identity, minted by a resource manager during upload.
///
/// Identities are never recycled or serialized. Independent registrations in
/// the same process/module instance cannot alias, even across renderers.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeshId(u64);

impl MeshId {
    pub(crate) fn fresh() -> Result<Self, MeshResourceError> {
        Self::issue(&NEXT_MESH_ID)
    }

    fn issue(next: &AtomicU64) -> Result<Self, MeshResourceError> {
        next.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map(Self)
        .map_err(|_| MeshResourceError::IdentityExhausted)
    }
}

/// A row in a wire mesh table, not a GPU resource handle.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeshTableIndex(u32);

impl MeshTableIndex {
    /// Names a wire row; scene assembly checks it against the registered table.
    pub const fn new(row: u32) -> Self {
        Self(row)
    }

    /// The row value written to or read from the protocol.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The row's slice index.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for MeshTableIndex {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Registration or residency failed without substituting a different resource.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MeshResourceError {
    #[error("mesh identity space is exhausted")]
    IdentityExhausted,
    #[error("mesh {mesh:?} is not resident in this renderer")]
    NotResident { mesh: MeshId },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities_are_small_copy_values_and_never_reissued() {
        let first = MeshId::fresh().unwrap();
        let second = MeshId::fresh().unwrap();
        let copy = first;
        assert_eq!(copy, first);
        assert_ne!(first, second);
        assert_eq!(std::mem::size_of::<MeshId>(), std::mem::size_of::<u64>());
    }

    #[test]
    fn concurrent_registration_uses_one_issuer() {
        let ids = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        (0..64)
                            .map(|_| MeshId::fresh().unwrap())
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
        let unique: std::collections::HashSet<_> = ids.iter().copied().collect();
        assert_eq!(unique.len(), ids.len());
    }

    #[test]
    fn exhaustion_does_not_wrap_or_reissue_an_identity() {
        let next = AtomicU64::new(u64::MAX - 1);
        assert_eq!(MeshId::issue(&next), Ok(MeshId(u64::MAX - 1)));
        assert_eq!(
            MeshId::issue(&next),
            Err(MeshResourceError::IdentityExhausted)
        );
        assert_eq!(
            MeshId::issue(&next),
            Err(MeshResourceError::IdentityExhausted)
        );
        assert_eq!(next.load(Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn wire_index_round_trip() {
        let row = MeshTableIndex::new(u32::MAX);
        assert_eq!(row.get(), u32::MAX);
        assert_eq!(row.index(), u32::MAX as usize);
    }
}
