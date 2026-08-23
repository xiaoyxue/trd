//! [`InteractionController`] — maps normalized gestures to [`SceneState`] changes (#97).
//! Toolkit-agnostic: consumes [`InteractionEvent`] fractions, not egui types.

use crate::scene::SceneState;

/// Full-width drag: π radians (half-turn), for orbit and object rotation.
const ROTATE_SPEED: f32 = std::f32::consts::PI;
/// One notch of scroll dollies the camera distance by this fraction.
const ZOOM_SPEED: f32 = 0.1;
/// Full-width pan: fraction of camera distance (consistent feel across zoom).
const PAN_SPEED: f32 = 1.0;
/// One notch of scroll scales the object by this fraction (in Scale mode).
const SCALE_WHEEL_SPEED: f32 = 0.1;
/// Full-height Scale drag factor (drag up grows, down shrinks).
const SCALE_DRAG_SPEED: f32 = 1.5;

/// What a primary drag manipulates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InteractionTarget {
    #[default]
    Camera,
    Object,
}

/// Which object transform a primary drag (and scroll wheel) edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransformMode {
    #[default]
    Rotate,
    Move,
    Scale,
}

/// Mutually exclusive direction used by an object move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MoveDirection {
    #[default]
    Free,
    Reference1,
    Reference2,
    Reference3,
    LocalX,
    LocalY,
    LocalZ,
}

/// Constrains an object rotation to a single axis, locking the other two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AxisConstraint {
    #[default]
    Free,
    /// Lock to the object's X axis (rotate = pitch; translate = world X).
    X,
    /// Lock to the object's Y axis (rotate = yaw; translate = world Y).
    Y,
    /// Lock to the object's Z axis (rotate = roll; translate = world Z).
    Z,
}

impl AxisConstraint {
    fn index(self) -> Option<usize> {
        match self {
            AxisConstraint::Free => None,
            AxisConstraint::X => Some(0),
            AxisConstraint::Y => Some(1),
            AxisConstraint::Z => Some(2),
        }
    }
}

/// Normalized gesture: drag/pan deltas are fractions of image size, `+y` down.
/// `Zoom.delta > 0` zooms in; the controller flips `y` for world-up mappings.
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

/// Owns the [`SceneState`] and applies [`InteractionEvent`]s to it.
#[derive(Debug, Clone)]
pub struct InteractionController {
    pub state: SceneState,
    pub target: InteractionTarget,
    pub mode: TransformMode,
    pub move_direction: MoveDirection,
    /// Maps the three reference directions into the object's parent frame.
    pub move_reference_axes: [[f32; 3]; 3],
    pub axis: AxisConstraint,
    /// The state to restore on [`InteractionEvent::Reset`].
    initial: SceneState,
}

impl InteractionController {
    pub fn new(state: SceneState) -> Self {
        let initial = state.clone();
        Self {
            state,
            target: InteractionTarget::default(),
            mode: TransformMode::default(),
            move_direction: MoveDirection::default(),
            move_reference_axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            axis: AxisConstraint::default(),
            initial,
        }
    }

    pub fn rebase_reset(&mut self) {
        self.initial = self.state.clone();
    }

    /// Applies one gesture, returning `true` if the scene changed.
    pub fn apply(&mut self, event: InteractionEvent) -> bool {
        match event {
            InteractionEvent::Primary { dx, dy } => {
                if dx == 0.0 && dy == 0.0 {
                    return false;
                }
                match self.target {
                    InteractionTarget::Camera => {
                        self.state
                            .camera
                            .orbit(dx * ROTATE_SPEED, -dy * ROTATE_SPEED);
                    }
                    InteractionTarget::Object => return self.apply_object_drag(dx, dy),
                }
                true
            }
            InteractionEvent::Pan { dx, dy } => {
                if dx == 0.0 && dy == 0.0 {
                    return false;
                }
                // Pan moves the *selected* object; no selection ⇒ nothing to move.
                let dist = self.state.camera.distance;
                let Some(obj) = self.state.selected_object_mut() else {
                    return false;
                };
                let scale = PAN_SPEED * dist;
                obj.translate([dx * scale, -dy * scale, 0.0]);
                true
            }
            InteractionEvent::Zoom { delta } => {
                if delta == 0.0 {
                    return false;
                }
                self.state.camera.dolly(1.0 - delta * ZOOM_SPEED);
                true
            }
            InteractionEvent::Scale { delta } => {
                if delta == 0.0 {
                    return false;
                }
                let Some(obj) = self.state.selected_object_mut() else {
                    return false;
                };
                obj.scale_uniform(1.0 + delta * SCALE_WHEEL_SPEED);
                true
            }
            InteractionEvent::Reset => {
                if self.state == self.initial {
                    return false;
                }
                self.state = self.initial.clone();
                true
            }
        }
    }

    /// Applies a primary drag to the selected object; returns `false` if no object is selected.
    fn apply_object_drag(&mut self, dx: f32, dy: f32) -> bool {
        let mode = self.mode;
        let move_direction = self.move_direction;
        let move_reference_axes = self.move_reference_axes;
        let axis = self.axis;
        let dist = self.state.camera.distance;
        let Some(obj) = self.state.selected_object_mut() else {
            return false;
        };
        match mode {
            TransformMode::Rotate => match axis.index() {
                // Locked: rotate about the single axis (X=pitch, Y=yaw, Z=roll).
                Some(i) => {
                    let a = (dx - dy) * ROTATE_SPEED;
                    match i {
                        0 => obj.pitch += a,
                        1 => obj.yaw += a,
                        _ => obj.roll += a,
                    }
                }
                None => obj.rotate(dx * ROTATE_SPEED, -dy * ROTATE_SPEED),
            },
            TransformMode::Move => {
                if move_direction == MoveDirection::Free {
                    let scale = PAN_SPEED * dist;
                    obj.translate([dx * scale, -dy * scale, 0.0]);
                } else {
                    let amount = (dx - dy) * PAN_SPEED * dist;
                    match move_direction {
                        MoveDirection::Reference1 => {
                            obj.translate(scale3(move_reference_axes[0], amount));
                        }
                        MoveDirection::Reference2 => {
                            obj.translate(scale3(move_reference_axes[1], amount));
                        }
                        MoveDirection::Reference3 => {
                            obj.translate(scale3(move_reference_axes[2], amount));
                        }
                        MoveDirection::LocalX => obj.translate_local([amount, 0.0, 0.0]),
                        MoveDirection::LocalY => obj.translate_local([0.0, amount, 0.0]),
                        MoveDirection::LocalZ => obj.translate_local([0.0, 0.0, amount]),
                        MoveDirection::Free => unreachable!(),
                    }
                }
            }
            TransformMode::Scale => {
                // Drag up (dy < 0) grows; exponential, never crosses zero.
                obj.scale_uniform((-dy * SCALE_DRAG_SPEED).exp());
            }
        }
        true
    }
}

fn scale3(vector: [f32; 3], scalar: f32) -> [f32; 3] {
    [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selected_controller() -> InteractionController {
        let mut c = InteractionController::new(SceneState::default());
        c.target = InteractionTarget::Object;
        c.state.selected = Some(0);
        c
    }

    #[test]
    fn primary_drag_orbits_camera_by_default() {
        let mut c = InteractionController::new(SceneState::default());
        let yaw0 = c.state.camera.yaw;
        let obj0 = c.state.objects[0];
        assert!(c.apply(InteractionEvent::Primary { dx: 0.5, dy: 0.0 }));
        assert!(c.state.camera.yaw > yaw0);
        assert_eq!(c.state.objects[0], obj0);
    }

    #[test]
    fn object_transform_is_noop_without_a_selection() {
        let mut c = InteractionController::new(SceneState::default());
        c.target = InteractionTarget::Object;
        assert_eq!(c.state.selected, None);
        let obj0 = c.state.objects[0];
        assert!(!c.apply(InteractionEvent::Primary { dx: 0.5, dy: 0.2 }));
        c.mode = TransformMode::Move;
        assert!(!c.apply(InteractionEvent::Primary { dx: 0.3, dy: 0.3 }));
        c.mode = TransformMode::Scale;
        assert!(!c.apply(InteractionEvent::Primary { dx: 0.0, dy: -0.3 }));
        assert!(!c.apply(InteractionEvent::Pan { dx: 0.2, dy: 0.2 }));
        assert!(!c.apply(InteractionEvent::Scale { delta: 1.0 }));
        assert_eq!(
            c.state.objects[0], obj0,
            "object unchanged with no selection"
        );
    }

    #[test]
    fn primary_drag_rotates_object_when_targeted() {
        let mut c = selected_controller();
        let cam0 = c.state.camera;
        assert!(c.apply(InteractionEvent::Primary { dx: 0.5, dy: 0.2 }));
        assert!(c.state.objects[0].yaw > 0.0);
        assert!(c.state.objects[0].pitch < 0.0);
        assert_eq!(c.state.camera, cam0);
    }

    #[test]
    fn object_move_mode_drag_translates() {
        let mut c = selected_controller();
        c.mode = TransformMode::Move;
        let rot0 = (c.state.objects[0].yaw, c.state.objects[0].pitch);
        assert!(c.apply(InteractionEvent::Primary { dx: 0.25, dy: -0.5 }));
        let t = c.state.objects[0].translation;
        assert!(t[0] > 0.0 && t[1] > 0.0 && t[2] == 0.0);
        assert_eq!((c.state.objects[0].yaw, c.state.objects[0].pitch), rot0);
    }

    #[test]
    fn object_move_reference_direction_uses_configured_basis() {
        let mut c = selected_controller();
        c.mode = TransformMode::Move;
        c.move_direction = MoveDirection::Reference2;
        c.move_reference_axes[1] = [0.0, 0.0, -1.0];
        assert!(c.apply(InteractionEvent::Primary { dx: 0.25, dy: -0.5 }));
        let t = c.state.objects[0].translation;
        assert_eq!(t[0], 0.0);
        assert_eq!(t[1], 0.0);
        assert!(t[2] < 0.0);
    }

    #[test]
    fn object_scale_mode_drag_up_grows() {
        let mut c = selected_controller();
        c.mode = TransformMode::Scale;
        assert!(c.apply(InteractionEvent::Primary { dx: 0.0, dy: -0.3 }));
        for s in c.state.objects[0].scale {
            assert!(s > 1.0);
        }
    }

    #[test]
    fn scale_event_scales_object_not_camera() {
        let mut c = selected_controller();
        let dist0 = c.state.camera.distance;
        assert!(c.apply(InteractionEvent::Scale { delta: 1.0 }));
        for s in c.state.objects[0].scale {
            assert!(s > 1.0);
        }
        assert_eq!(c.state.camera.distance, dist0);
    }

    #[test]
    fn axis_locked_rotate_x_changes_only_pitch() {
        let mut c = selected_controller();
        c.mode = TransformMode::Rotate;
        c.axis = AxisConstraint::X;
        assert!(c.apply(InteractionEvent::Primary { dx: 0.3, dy: 0.1 }));
        assert_ne!(c.state.objects[0].pitch, 0.0);
        assert_eq!(c.state.objects[0].yaw, 0.0);
        assert_eq!(c.state.objects[0].roll, 0.0);
    }

    #[test]
    fn axis_locked_rotate_z_changes_only_roll() {
        let mut c = selected_controller();
        c.mode = TransformMode::Rotate;
        c.axis = AxisConstraint::Z;
        assert!(c.apply(InteractionEvent::Primary { dx: 0.4, dy: 0.0 }));
        assert_ne!(c.state.objects[0].roll, 0.0);
        assert_eq!(c.state.objects[0].yaw, 0.0);
        assert_eq!(c.state.objects[0].pitch, 0.0);
    }

    #[test]
    fn local_z_move_follows_rotated_object_basis() {
        let mut c = selected_controller();
        c.mode = TransformMode::Move;
        c.state.objects[0].yaw = std::f32::consts::FRAC_PI_2;
        c.move_direction = MoveDirection::LocalZ;
        assert!(c.apply(InteractionEvent::Primary { dx: 0.2, dy: -0.4 }));
        let t = c.state.objects[0].translation;
        assert!(t[0].abs() > 0.0);
        assert_eq!(t[1], 0.0);
        assert!(t[2].abs() < 1e-5);
    }

    #[test]
    fn pan_translates_object_scaled_by_distance() {
        let mut c = selected_controller();
        let dist = c.state.camera.distance;
        assert!(c.apply(InteractionEvent::Pan { dx: 0.25, dy: -0.5 }));
        let t = c.state.objects[0].translation;
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
        let mut c = selected_controller();
        c.apply(InteractionEvent::Primary { dx: 0.7, dy: 0.3 });
        c.apply(InteractionEvent::Zoom { delta: 2.0 });
        c.apply(InteractionEvent::Scale { delta: 3.0 });
        assert_ne!(c.state, c.initial);
        assert!(c.apply(InteractionEvent::Reset));
        assert_eq!(c.state, c.initial);
        assert_eq!(c.state.objects[0].scale, [1.0, 1.0, 1.0]);
        assert!(!c.apply(InteractionEvent::Reset));
    }

    #[test]
    fn zero_gestures_report_no_change() {
        let mut c = selected_controller();
        assert!(!c.apply(InteractionEvent::Primary { dx: 0.0, dy: 0.0 }));
        assert!(!c.apply(InteractionEvent::Pan { dx: 0.0, dy: 0.0 }));
        assert!(!c.apply(InteractionEvent::Zoom { delta: 0.0 }));
        assert!(!c.apply(InteractionEvent::Scale { delta: 0.0 }));
    }
}
