//! `trd-core::math` — a small, type-safe geometry layer over [`glam`].
//!
//! This module gives `trd` **affine-space-correct** types so we stop passing raw
//! `Mat4` / `[f32; 16]` around and can no longer confuse the three things that
//! transform differently under a 4×4 matrix:
//!
//! | type | homogeneous form | transformed by |
//! |------|------------------|----------------|
//! | [`Point3`]  | `(p, 1)` | `M · (p, 1)` (may need a perspective divide) |
//! | [`Vector3`] | `(v, 0)` | `M · (v, 0)` — the upper-left 3×3 only |
//! | [`Normal3`] | covector | `(M⁻¹)ᵀ · n` — the inverse-transpose |
//!
//! Every type is a `#[repr(transparent)]` newtype over a glam primitive with a
//! **private** inner field, so the affine rules can't be bypassed. The wrappers
//! are zero-cost and keep glam's SIMD.
//!
//! # Conventions (single source of truth)
//!
//! The rest of the crate references these; they are stated **once**, here:
//!
//! - **Matrices are column-major**, applied to **column vectors** by
//!   **left-multiplication**: `clip = P · V · M · v`.
//! - **Right-handed** throughout. Camera looks down `-z`.
//! - **Clip space `z ∈ [0, 1]`** (wgpu), y-up NDC — so we always use glam's `_rh`
//!   constructors ([`Transform::perspective_rh`], [`Transform::look_at_rh`]),
//!   never the `_rh_gl` (OpenGL, `z ∈ [-1, 1]`) variants.
//! - **Composition:** [`Transform::then`] applies the receiver first:
//!   `a.then(b) == b * a`.
//! - **Angles are `f32` radians** (no `Rad` / `Deg` newtype in v1).
//!
//! # Scalar precision
//!
//! v1 is `f32`-only (the GPU path is `f32`). A future `f64` need is expected to be
//! served by a Cargo **feature-flag scalar alias** (swap the glam backing types
//! per build) rather than a `Scalar` generic — the public type names stay stable.

mod aabb;
mod gpu;
mod linalg;
mod transform;

pub use aabb::{Aabb2, Aabb3};
pub use gpu::ToWgsl;
pub use linalg::{
    Matrix3, Matrix4, Normal3, Point2, Point3, Point4, Rotation, Vector2, Vector3, Vector4,
};
pub use transform::Transform;

/// The scalar type the whole module is built on (see "Scalar precision").
pub type Scalar = f32;

/// Default absolute tolerance for approximate geometric comparisons.
pub const EPSILON: f32 = 1.0e-6;
