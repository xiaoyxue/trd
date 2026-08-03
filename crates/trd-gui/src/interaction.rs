//! [`InteractionController`] — maps normalized user gestures to [`SceneState`]
//! changes (#97), the "events → matrix" core of the interaction loop.
//!
//! The controller is **UI-toolkit-agnostic**: it consumes a normalized
//! [`InteractionEvent`] (drag/pan/zoom fractions, not egui types) and mutates the
//! scene, so it is unit-testable without egui and reusable by the wasm target.
//! egui specifics (which mouse button, the image rect) live in [`crate::app`],
//! which translates raw input into these events.
//!
//! The [`InteractionTarget`] decides what a primary drag means: orbit the
//! **camera** or rotate the **object**. Panning always translates the object and
//! zooming always dollies the camera, regardless of target.

use crate::scene::SceneState;

/// A full-width primary drag sweeps this many radians (π ⇒ half a turn across
/// the image), used for both camera orbit and object rotation.
const ROTATE_SPEED: f32 = std::f32::consts::PI;
/// One notch of scroll dollies the camera distance by this fraction.
const ZOOM_SPEED: f32 = 0.1;
/// A full-width pan translates the object by this fraction of the camera
/// distance, so panning feels consistent as you dolly in and out.
const PAN_SPEED: f32 = 1.0;
/// One notch of scroll scales the object by this fraction (in Scale mode).
const SCALE_WHEEL_SPEED: f32 = 0.1;
/// A full-height Scale-mode drag scales the object by this fraction per unit of
/// vertical travel (drag **up** grows, drag **down** shrinks).
const SCALE_DRAG_SPEED: f32 = 1.5;

/// What a **primary** (left-button) drag manipulates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InteractionTarget {
    /// Primary drag orbits the camera around the object (the default).
    #[default]
    Camera,
    /// Primary drag manipulates the object per the active [`TransformMode`].
    Object,
}

/// When [`InteractionTarget::Object`] is active, which object transform a
/// **primary** drag (and the scroll wheel) edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransformMode {
    /// Primary drag rotates the object (yaw/pitch); the default.
    #[default]
    Rotate,
    /// Primary drag translates the object in the screen plane.
    Move,
    /// Primary drag (and the scroll wheel) uniformly scales the object.
    Scale,
}

/// A normalized interaction gesture. Drag/pan deltas are **fractions of the
/// image size** (`dx`/`dy` in roughly `[-1, 1]` for an edge-to-edge sweep) with
/// `+y` pointing down (egui screen convention); the controller flips `y` where a
/// world-up mapping is needed. `Zoom.delta` is a signed scroll amount
/// (`+` = zoom in / dolly closer).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InteractionEvent {
    /// Primary drag: orbit the camera, or rotate / move / scale the object
    /// (per [`InteractionTarget`] and [`TransformMode`]).
    Primary { dx: f32, dy: f32 },
    /// Secondary/middle drag: translate the object in the screen plane.
    Pan { dx: f32, dy: f32 },
    /// Scroll wheel: dolly the camera (`delta > 0` moves closer).
    Zoom { delta: f32 },
    /// Scroll wheel while scaling: uniformly scale the object (`delta > 0` grows).
    Scale { delta: f32 },
    /// Restore the scene to its initial state.
    Reset,
}

/// Owns the [`SceneState`] and applies [`InteractionEvent`]s to it. Keeps the
/// initial state so [`InteractionEvent::Reset`] can restore the view.
#[derive(Debug, Clone)]
pub struct InteractionController {
    /// The scene being edited (read by the render backend each frame).
    pub state: SceneState,
    /// What a primary drag currently manipulates (camera vs. object).
    pub target: InteractionTarget,
    /// Which object transform a primary drag edits when targeting the object.
    pub mode: TransformMode,
    /// The state to restore on [`InteractionEvent::Reset`].
    initial: SceneState,
}

impl InteractionController {
    /// Builds a controller around an initial scene (also the reset baseline).
    pub fn new(state: SceneState) -> Self {
        Self {
            state,
            target: InteractionTarget::default(),
            mode: TransformMode::default(),
            initial: state,
        }
    }

    /// Applies one gesture, returning `true` if the scene changed (so the caller
    /// can re-render only when needed).
    pub fn apply(&mut self, event: InteractionEvent) -> bool {
        match event {
            InteractionEvent::Primary { dx, dy } => {
                if dx == 0.0 && dy == 0.0 {
                    return false;
                }
                match self.target {
                    InteractionTarget::Camera => {
                        // Drag right → orbit right (yaw follows the pointer);
                        // drag down (dy > 0) → tilt the eye down (pitch down).
                        self.state
                            .camera
                            .orbit(dx * ROTATE_SPEED, -dy * ROTATE_SPEED);
                    }
                    InteractionTarget::Object => self.apply_object_drag(dx, dy),
                }
                true
            }
            InteractionEvent::Pan { dx, dy } => {
                if dx == 0.0 && dy == 0.0 {
                    return false;
                }
                self.translate_object_screen(dx, dy);
                true
            }
            InteractionEvent::Zoom { delta } => {
                if delta == 0.0 {
                    return false;
                }
                // delta > 0 ⇒ dolly closer (factor < 1).
                self.state.camera.dolly(1.0 - delta * ZOOM_SPEED);
                true
            }
            InteractionEvent::Scale { delta } => {
                if delta == 0.0 {
                    return false;
                }
                // delta > 0 ⇒ grow (factor > 1).
                self.state
                    .object
                    .scale_uniform(1.0 + delta * SCALE_WHEEL_SPEED);
                true
            }
            InteractionEvent::Reset => {
                if self.state == self.initial {
                    return false;
                }
                self.state = self.initial;
                true
            }
        }
    }

    /// Applies a primary drag to the object per the active [`TransformMode`]:
    /// rotate (yaw/pitch), move (screen plane), or uniformly scale.
    fn apply_object_drag(&mut self, dx: f32, dy: f32) {
        match self.mode {
            TransformMode::Rotate => {
                self.state
                    .object
                    .rotate(dx * ROTATE_SPEED, -dy * ROTATE_SPEED);
            }
            TransformMode::Move => self.translate_object_screen(dx, dy),
            TransformMode::Scale => {
                // Drag up (dy < 0) grows, drag down shrinks; exponential so the
                // gesture is symmetric and never crosses zero.
                self.state
                    .object
                    .scale_uniform((-dy * SCALE_DRAG_SPEED).exp());
            }
        }
    }

    /// Translates the object in the world XY plane, scaled by camera distance so
    /// it tracks the pointer regardless of zoom. Screen-up (`dy < 0`) maps to
    /// world `+Y`. Shared by [`InteractionEvent::Pan`] and the Move drag mode.
    fn translate_object_screen(&mut self, dx: f32, dy: f32) {
        let scale = PAN_SPEED * self.state.camera.distance;
        self.state.object.translate([dx * scale, -dy * scale, 0.0]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_drag_orbits_camera_by_default() {
        let mut c = InteractionController::new(SceneState::default());
        let yaw0 = c.state.camera.yaw;
        let obj0 = c.state.object;
        assert!(c.apply(InteractionEvent::Primary { dx: 0.5, dy: 0.0 }));
        assert!(c.state.camera.yaw > yaw0);
        // Object untouched when targeting the camera.
        assert_eq!(c.state.object, obj0);
    }

    #[test]
    fn primary_drag_rotates_object_when_targeted() {
        let mut c = InteractionController::new(SceneState::default());
        c.target = InteractionTarget::Object;
        let cam0 = c.state.camera;
        assert!(c.apply(InteractionEvent::Primary { dx: 0.5, dy: 0.2 }));
        assert!(c.state.object.yaw > 0.0);
        assert!(c.state.object.pitch < 0.0);
        // Camera untouched when targeting the object.
        assert_eq!(c.state.camera, cam0);
    }

    #[test]
    fn object_move_mode_drag_translates() {
        let mut c = InteractionController::new(SceneState::default());
        c.target = InteractionTarget::Object;
        c.mode = TransformMode::Move;
        let rot0 = (c.state.object.yaw, c.state.object.pitch);
        assert!(c.apply(InteractionEvent::Primary { dx: 0.25, dy: -0.5 }));
        let t = c.state.object.translation;
        assert!(t[0] > 0.0 && t[1] > 0.0 && t[2] == 0.0);
        // Rotation untouched in Move mode.
        assert_eq!((c.state.object.yaw, c.state.object.pitch), rot0);
    }

    #[test]
    fn object_scale_mode_drag_up_grows() {
        let mut c = InteractionController::new(SceneState::default());
        c.target = InteractionTarget::Object;
        c.mode = TransformMode::Scale;
        // Drag up (dy < 0) grows uniformly above unit scale.
        assert!(c.apply(InteractionEvent::Primary { dx: 0.0, dy: -0.3 }));
        for s in c.state.object.scale {
            assert!(s > 1.0);
        }
    }

    #[test]
    fn scale_event_scales_object_not_camera() {
        let mut c = InteractionController::new(SceneState::default());
        let dist0 = c.state.camera.distance;
        assert!(c.apply(InteractionEvent::Scale { delta: 1.0 }));
        for s in c.state.object.scale {
            assert!(s > 1.0);
        }
        // Scaling the object never dollies the camera.
        assert_eq!(c.state.camera.distance, dist0);
    }

    #[test]
    fn pan_translates_object_scaled_by_distance() {
        let mut c = InteractionController::new(SceneState::default());
        let dist = c.state.camera.distance;
        assert!(c.apply(InteractionEvent::Pan { dx: 0.25, dy: -0.5 }));
        let t = c.state.object.translation;
        assert!((t[0] - 0.25 * PAN_SPEED * dist).abs() < 1e-4);
        // Screen-up (dy < 0) → world +Y.
        assert!((t[1] - 0.5 * PAN_SPEED * dist).abs() < 1e-4);
        assert_eq!(t[2], 0.0);
    }

    #[test]
    fn zoom_in_reduces_distance() {
        let mut c = InteractionController::new(SceneState::default());
        let d0 = c.state.camera.distance;
        assert!(c.apply(InteractionEvent::Zoom { delta: 1.0 }));
        assert!(c.state.camera.distance < d0);
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut c = InteractionController::new(SceneState::default());
        c.apply(InteractionEvent::Primary { dx: 0.7, dy: 0.3 });
        c.apply(InteractionEvent::Zoom { delta: 2.0 });
        c.apply(InteractionEvent::Scale { delta: 3.0 });
        assert_ne!(c.state, c.initial);
        assert!(c.apply(InteractionEvent::Reset));
        assert_eq!(c.state, c.initial);
        // Reset also restored the object scale to unit.
        assert_eq!(c.state.object.scale, [1.0, 1.0, 1.0]);
        // A second reset is a no-op (nothing changed).
        assert!(!c.apply(InteractionEvent::Reset));
    }

    #[test]
    fn zero_gestures_report_no_change() {
        let mut c = InteractionController::new(SceneState::default());
        assert!(!c.apply(InteractionEvent::Primary { dx: 0.0, dy: 0.0 }));
        assert!(!c.apply(InteractionEvent::Pan { dx: 0.0, dy: 0.0 }));
        assert!(!c.apply(InteractionEvent::Zoom { delta: 0.0 }));
        assert!(!c.apply(InteractionEvent::Scale { delta: 0.0 }));
    }
}
