//! Axis-aligned bounding boxes: [`Aabb2`] and [`Aabb3`].
//!
//! An AABB is stored as its `min`/`max` **corners** (not center/half-extents):
//! the corner form makes [`Aabb3::union`] / [`Aabb3::intersection`] a
//! component-wise `min`/`max` and needs no redundant field. The empty box is the
//! **inverted-infinite** sentinel (`min = +∞`, `max = -∞`), so [`Aabb3::union`]
//! has an identity and [`Aabb3::from_points`] folds cleanly from [`Aabb3::EMPTY`].
//!
//! The affine type split pays off here: the diagonal [`Aabb3::size`] is a
//! [`Vector3`](super::Vector3) and the [`Aabb3::center`] is
//! `min + (max - min) * 0.5` — the buggy `(min + max) * 0.5` simply doesn't
//! type-check, because `Point + Point` is not defined.
//!
//! AABBs are CPU-side geometry (culling, layout, fit-to-content); they are
//! intentionally **not** `Pod` (the empty sentinel is `±∞`, and they are not a
//! GPU-upload type).

use super::linalg::{Point2, Point3, Vector2, Vector3};

macro_rules! impl_aabb {
    ($Name:ident, $Point:ident, $Vector:ident, $Inner:ty, $N:literal) => {
        #[doc = concat!("An axis-aligned bounding box in ", stringify!($N), "-D (`min`/`max` corners).")]
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $Name {
            min: $Point,
            max: $Point,
        }

        impl Default for $Name {
            /// The empty box (see [`Self::EMPTY`]).
            #[inline]
            fn default() -> Self {
                Self::EMPTY
            }
        }

        impl $Name {
            /// The empty box: `min = +∞`, `max = -∞`. It is the identity of
            /// [`Self::union`] and the seed of [`Self::from_points`], and reports
            /// [`Self::is_empty`] `== true`.
            pub const EMPTY: Self = Self {
                min: $Point::from_glam(<$Inner>::splat(f32::INFINITY)),
                max: $Point::from_glam(<$Inner>::splat(f32::NEG_INFINITY)),
            };

            /// Builds from ordered corners; **assumes** `min <= max` component-wise.
            /// Use [`Self::from_corners`] when the order is unknown.
            #[inline]
            pub const fn new(min: $Point, max: $Point) -> Self {
                Self { min, max }
            }

            /// Builds the box spanning two arbitrary corners (component-wise
            /// `min`/`max`), so the argument order doesn't matter.
            #[inline]
            pub fn from_corners(a: $Point, b: $Point) -> Self {
                Self {
                    min: $Point::from_glam(a.into_inner().min(b.into_inner())),
                    max: $Point::from_glam(a.into_inner().max(b.into_inner())),
                }
            }

            /// The tight box enclosing all `points` (empty if the iterator is).
            #[inline]
            pub fn from_points(points: impl IntoIterator<Item = $Point>) -> Self {
                points
                    .into_iter()
                    .fold(Self::EMPTY, |acc, p| acc.union_point(p))
            }

            /// Builds from a center and (non-negative) half-extents.
            #[inline]
            pub fn from_center_half_extents(center: $Point, half_extents: $Vector) -> Self {
                Self {
                    min: center - half_extents,
                    max: center + half_extents,
                }
            }

            /// The minimum corner.
            #[inline]
            pub const fn min(self) -> $Point {
                self.min
            }
            /// The maximum corner.
            #[inline]
            pub const fn max(self) -> $Point {
                self.max
            }
            /// The center point, `min + (max - min) * 0.5`.
            #[inline]
            pub fn center(self) -> $Point {
                self.min + (self.max - self.min) * 0.5
            }
            /// The diagonal `max - min` (a direction). Meaningless when empty.
            #[inline]
            pub fn size(self) -> $Vector {
                self.max - self.min
            }
            /// Half the diagonal.
            #[inline]
            pub fn half_extents(self) -> $Vector {
                (self.max - self.min) * 0.5
            }

            /// Whether the box is empty (any `min` component exceeds `max`).
            #[inline]
            pub fn is_empty(self) -> bool {
                self.min.into_inner().cmpgt(self.max.into_inner()).any()
            }

            /// Whether `p` lies inside (inclusive of the boundary).
            #[inline]
            pub fn contains_point(self, p: $Point) -> bool {
                let p = p.into_inner();
                self.min.into_inner().cmple(p).all() && self.max.into_inner().cmpge(p).all()
            }

            /// Whether `self` fully encloses `other` (an empty `other` is
            /// always enclosed).
            #[inline]
            pub fn contains(self, other: Self) -> bool {
                other.is_empty()
                    || (self.min.into_inner().cmple(other.min.into_inner()).all()
                        && self.max.into_inner().cmpge(other.max.into_inner()).all())
            }

            /// Whether the two boxes overlap (touching counts; empties never
            /// intersect).
            #[inline]
            pub fn intersects(self, other: Self) -> bool {
                self.min.into_inner().cmple(other.max.into_inner()).all()
                    && self.max.into_inner().cmpge(other.min.into_inner()).all()
            }

            /// The smallest box enclosing both (`union` with [`Self::EMPTY`] is
            /// the identity).
            #[inline]
            pub fn union(self, other: Self) -> Self {
                Self {
                    min: $Point::from_glam(self.min.into_inner().min(other.min.into_inner())),
                    max: $Point::from_glam(self.max.into_inner().max(other.max.into_inner())),
                }
            }

            /// The smallest box enclosing `self` and the point `p`.
            #[inline]
            pub fn union_point(self, p: $Point) -> Self {
                Self {
                    min: $Point::from_glam(self.min.into_inner().min(p.into_inner())),
                    max: $Point::from_glam(self.max.into_inner().max(p.into_inner())),
                }
            }

            /// The overlap of the two boxes; **empty** (per [`Self::is_empty`])
            /// when they are disjoint.
            #[inline]
            pub fn intersection(self, other: Self) -> Self {
                Self {
                    min: $Point::from_glam(self.min.into_inner().max(other.min.into_inner())),
                    max: $Point::from_glam(self.max.into_inner().min(other.max.into_inner())),
                }
            }

            /// Grows the box by `margin` on every side (shrinks if negative).
            #[inline]
            pub fn expanded(self, margin: f32) -> Self {
                let m = $Vector::from_glam(<$Inner>::splat(margin));
                Self {
                    min: self.min - m,
                    max: self.max + m,
                }
            }
        }
    };
}

impl_aabb!(Aabb2, Point2, Vector2, glam::Vec2, 2);
impl_aabb!(Aabb3, Point3, Vector3, glam::Vec3, 3);

impl Aabb2 {
    /// The four corners, counter-clockwise from `min`.
    #[inline]
    pub fn corners(self) -> [Point2; 4] {
        let (lo, hi) = (self.min(), self.max());
        [
            Point2::new(lo.x(), lo.y()),
            Point2::new(hi.x(), lo.y()),
            Point2::new(hi.x(), hi.y()),
            Point2::new(lo.x(), hi.y()),
        ]
    }

    /// The enclosed area (`0` when empty).
    #[inline]
    pub fn area(self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let e = self.size();
        e.x() * e.y()
    }
}

impl Aabb3 {
    /// The eight corners of the box.
    #[inline]
    pub fn corners(self) -> [Point3; 8] {
        let (lo, hi) = (self.min(), self.max());
        [
            Point3::new(lo.x(), lo.y(), lo.z()),
            Point3::new(hi.x(), lo.y(), lo.z()),
            Point3::new(lo.x(), hi.y(), lo.z()),
            Point3::new(hi.x(), hi.y(), lo.z()),
            Point3::new(lo.x(), lo.y(), hi.z()),
            Point3::new(hi.x(), lo.y(), hi.z()),
            Point3::new(lo.x(), hi.y(), hi.z()),
            Point3::new(hi.x(), hi.y(), hi.z()),
        ]
    }

    /// The enclosed volume (`0` when empty).
    #[inline]
    pub fn volume(self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let e = self.size();
        e.x() * e.y() * e.z()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::EPSILON;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    fn finite() -> impl Strategy<Value = f32> {
        -100.0f32..100.0
    }
    fn point3() -> impl Strategy<Value = Point3> {
        (finite(), finite(), finite()).prop_map(|(x, y, z)| Point3::new(x, y, z))
    }
    /// A non-empty box with `min <= max`.
    fn aabb3() -> impl Strategy<Value = Aabb3> {
        (point3(), point3()).prop_map(|(a, b)| Aabb3::from_corners(a, b))
    }

    #[test]
    fn empty_is_empty_and_default() {
        assert!(Aabb3::EMPTY.is_empty());
        assert_eq!(Aabb3::default(), Aabb3::EMPTY);
        assert!(Aabb2::EMPTY.is_empty());
        assert_eq!(Aabb3::EMPTY.volume(), 0.0);
        assert_eq!(Aabb2::EMPTY.area(), 0.0);
        // Nothing is inside the empty box.
        assert!(!Aabb3::EMPTY.contains_point(Point3::ORIGIN));
    }

    #[test]
    fn from_corners_orders_bounds() {
        let b = Aabb3::from_corners(Point3::new(1.0, 5.0, -2.0), Point3::new(-3.0, 0.0, 4.0));
        assert_eq!(b.min(), Point3::new(-3.0, 0.0, -2.0));
        assert_eq!(b.max(), Point3::new(1.0, 5.0, 4.0));
        assert!(!b.is_empty());
    }

    #[test]
    fn center_uses_affine_algebra() {
        let b = Aabb3::from_corners(Point3::new(-1.0, -2.0, -3.0), Point3::new(3.0, 4.0, 5.0));
        assert_abs_diff_eq!(
            b.center().into_inner(),
            Point3::new(1.0, 1.0, 1.0).into_inner(),
            epsilon = EPSILON
        );
        assert_abs_diff_eq!(
            b.size().into_inner(),
            Vector3::new(4.0, 6.0, 8.0).into_inner(),
            epsilon = EPSILON
        );
        assert_eq!(b.volume(), 4.0 * 6.0 * 8.0);
    }

    #[test]
    fn from_center_half_extents_round_trips() {
        let c = Point3::new(2.0, -1.0, 0.5);
        let h = Vector3::new(1.0, 2.0, 3.0);
        let b = Aabb3::from_center_half_extents(c, h);
        assert_abs_diff_eq!(b.center().into_inner(), c.into_inner(), epsilon = EPSILON);
        assert_abs_diff_eq!(
            b.half_extents().into_inner(),
            h.into_inner(),
            epsilon = EPSILON
        );
    }

    #[test]
    fn intersection_of_disjoint_is_empty() {
        let a = Aabb3::from_corners(Point3::ORIGIN, Point3::new(1.0, 1.0, 1.0));
        let b = Aabb3::from_corners(Point3::new(2.0, 2.0, 2.0), Point3::new(3.0, 3.0, 3.0));
        assert!(!a.intersects(b));
        assert!(a.intersection(b).is_empty());
    }

    #[test]
    fn corners_are_all_contained() {
        let b = Aabb3::from_corners(Point3::new(-1.0, -1.0, -1.0), Point3::new(2.0, 3.0, 4.0));
        for c in b.corners() {
            assert!(b.contains_point(c), "corner {c:?} not contained");
        }
        assert_eq!(b.corners().len(), 8);
        let b2 = Aabb2::from_corners(Point2::new(0.0, 0.0), Point2::new(1.0, 2.0));
        assert_eq!(b2.corners().len(), 4);
        assert_eq!(b2.area(), 2.0);
    }

    proptest! {
        #[test]
        fn union_contains_both(a in aabb3(), b in aabb3()) {
            let u = a.union(b);
            prop_assert!(u.contains(a));
            prop_assert!(u.contains(b));
        }

        #[test]
        fn union_with_empty_is_identity(a in aabb3()) {
            prop_assert_eq!(a.union(Aabb3::EMPTY), a);
            prop_assert_eq!(Aabb3::EMPTY.union(a), a);
        }

        #[test]
        fn from_points_encloses_every_point(ps in prop::collection::vec(point3(), 1..12)) {
            let b = Aabb3::from_points(ps.iter().copied());
            for p in ps {
                prop_assert!(b.contains_point(p));
            }
        }

        #[test]
        fn intersection_is_contained_in_both(a in aabb3(), b in aabb3()) {
            let i = a.intersection(b);
            if !i.is_empty() {
                prop_assert!(a.contains(i));
                prop_assert!(b.contains(i));
            }
        }

        #[test]
        fn expanded_contains_original(a in aabb3(), m in 0.0f32..10.0) {
            prop_assert!(a.expanded(m).contains(a));
        }

        #[test]
        fn intersects_matches_nonempty_intersection(a in aabb3(), b in aabb3()) {
            prop_assert_eq!(a.intersects(b), !a.intersection(b).is_empty());
        }
    }
}
