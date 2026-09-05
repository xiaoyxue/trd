//! Device-free mesh registration, separate from renderer residency.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::Mesh;

static NEXT_MESH_ID: AtomicU64 = AtomicU64::new(1);

/// A logical mesh identity, minted by registration or runtime upload.
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

/// An immutable, ordered CPU mesh table and its registered identities.
///
/// Cloning preserves identity and shares the CPU data. Uploading the same table
/// to another renderer is explicit; constructing another table mints new IDs.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshTable {
    entries: Arc<[(MeshId, Mesh)]>,
}

impl MeshTable {
    /// Registers each mesh once, before a GPU device is needed.
    pub fn new(meshes: Vec<Mesh>) -> Result<Self, MeshResourceError> {
        let entries = meshes
            .into_iter()
            .map(|mesh| Ok((MeshId::fresh()?, mesh)))
            .collect::<Result<Vec<_>, MeshResourceError>>()?;
        Ok(Self {
            entries: entries.into(),
        })
    }

    /// The number of wire rows, independent of renderer residency.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Resolves a wire row without consulting a renderer or GPU device.
    pub fn id(&self, row: MeshTableIndex) -> Option<MeshId> {
        self.entries.get(row.index()).map(|(id, _)| *id)
    }

    /// Registered meshes in wire-table order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (MeshId, &Mesh)> {
        self.entries.iter().map(|(id, mesh)| (*id, mesh))
    }

    /// Registered identities in wire-table order.
    pub fn ids(&self) -> impl ExactSizeIterator<Item = MeshId> + '_ {
        self.entries.iter().map(|(id, _)| *id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_is_device_free_and_cloning_preserves_identity() {
        let table = MeshTable::new(vec![Mesh::hello_triangle(), Mesh::hello_triangle()]).unwrap();
        let copy = table.clone();
        assert_eq!(table, copy);
        assert_eq!(table.len(), 2);
        assert!(!table.is_empty());
        assert_ne!(
            table.id(MeshTableIndex::new(0)),
            table.id(MeshTableIndex::new(1))
        );
        assert_eq!(table.id(MeshTableIndex::new(2)), None);
        assert_eq!(table.iter().count(), table.ids().count());
    }

    #[test]
    fn independent_tables_never_share_an_identity() {
        let first = MeshTable::new(vec![Mesh::hello_triangle()]).unwrap();
        let second = MeshTable::new(vec![Mesh::hello_triangle()]).unwrap();
        assert_ne!(
            first.id(MeshTableIndex::new(0)),
            second.id(MeshTableIndex::new(0))
        );
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
    fn empty_table_and_wire_index_round_trip() {
        let table = MeshTable::new(Vec::new()).unwrap();
        assert!(table.is_empty());
        let row = MeshTableIndex::new(u32::MAX);
        assert_eq!(row.get(), u32::MAX);
        assert_eq!(row.index(), u32::MAX as usize);
        assert_eq!(table.id(row), None);
    }
}
