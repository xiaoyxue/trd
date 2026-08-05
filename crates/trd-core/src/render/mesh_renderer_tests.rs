use super::*;

fn model(tag: f32) -> [f32; 16] {
    let mut model = Matrix4::IDENTITY.to_cols_array();
    model[12] = tag;
    model
}

fn mesh(mesh_id: u32, tag: f32, mode: RenderMode) -> DrawableObject {
    DrawableObject::Mesh {
        mesh_id,
        model: model(tag),
        mode,
    }
}

#[test]
fn batches_in_layer_order_and_preserves_equal_kind_order() {
    let scene = [
        DrawableObject::CoordinateAxes { model: model(80.0) },
        mesh(1, 61.0, RenderMode::Wireframe),
        DrawableObject::FramePlane {
            fit: FrameFit::Stretch,
        },
        mesh(1, 12.0, RenderMode::Filled),
        mesh(0, 30.0, RenderMode::Pbr),
        DrawableObject::BlobShadow { model: model(1.0) },
        DrawableObject::AabbBox {
            mesh_id: 1,
            model: model(71.0),
        },
        DrawableObject::PlaneGrid {
            plane: GridPlane::Yz,
            model: model(52.0),
        },
        mesh(0, 10.0, RenderMode::Filled),
        mesh(0, 20.0, RenderMode::Textured),
        DrawableObject::PlaneGrid {
            plane: GridPlane::Xy,
            model: model(50.0),
        },
        mesh(0, 11.0, RenderMode::Filled),
        DrawableObject::AabbBox {
            mesh_id: 0,
            model: model(70.0),
        },
        mesh(1, 31.0, RenderMode::Pbr),
        mesh(99, 99.0, RenderMode::Filled),
        mesh(0, 98.0, RenderMode::Shadow),
        DrawableObject::FramePlane {
            fit: FrameFit::Cover,
        },
    ];
    let base_models = [Matrix4::IDENTITY, Matrix4::IDENTITY];

    let batches = build_batches(&scene, |mesh_id| base_models.get(mesh_id).copied());
    let commands = batches
        .commands
        .iter()
        .map(|command| (command.kind, command.start, command.count))
        .collect::<Vec<_>>();

    assert_eq!(
        commands,
        [
            (DrawKind::Shadow, 0, 1),
            (DrawKind::Filled(0), 1, 2),
            (DrawKind::Filled(1), 3, 1),
            (DrawKind::Textured(0), 4, 1),
            (DrawKind::Pbr(0), 5, 1),
            (DrawKind::Pbr(1), 6, 1),
            (DrawKind::Grid(0), 7, 1),
            (DrawKind::Grid(2), 8, 1),
            (DrawKind::Wireframe(1), 9, 1),
            (DrawKind::Aabb(0), 10, 1),
            (DrawKind::Aabb(1), 11, 1),
            (DrawKind::Axes, 12, 1),
        ]
    );
    assert_eq!(
        batches
            .instances
            .iter()
            .map(|instance| instance.model[12])
            .collect::<Vec<_>>(),
        [1.0, 10.0, 11.0, 12.0, 20.0, 30.0, 31.0, 50.0, 52.0, 61.0, 70.0, 71.0, 80.0]
    );
    assert_eq!(batches.frame_fit, Some(FrameFit::Cover));
}
