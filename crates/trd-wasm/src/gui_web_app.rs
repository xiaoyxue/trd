//! [`WebApp`] — the browser eframe application (#97, Slice 4).
//!
//! The wasm twin of [`TrdGuiApp`](crate::app::TrdGuiApp): it draws the same shared
//! [`ui`](crate::ui) (side controls + central image) but renders **asynchronously**.
//! Because a GPU readback can't block the browser event loop, a changed scene is
//! rendered by spawning the [`GuiRenderer`]'s async `render` on the microtask
//! queue (`wasm_bindgen_futures::spawn_local`); when it completes, the RGBA is
//! stashed and a repaint is requested, and the next `ui` pass uploads it to the
//! egui texture. A single-flight guard coalesces rapid interactions so at most one
//! render is in flight.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use egui::{TextureHandle, TextureOptions};

use trd_gui::interaction::InteractionController;
use trd_gui::renderer::GuiRenderer;
use trd_gui::renderer::ImageRgba;
use trd_gui::ui;

/// The bridge between the running viewer and the JS shell (#353).
///
/// Two one-slot mailboxes rather than a channel: "open the picker" travels out
/// as a JS call, the picked bytes come back in and wait until a frame where the
/// renderer is not borrowed by an in-flight render.
#[derive(Default)]
pub struct GuiShared {
    /// The picked model waiting to be loaded, retried each frame until applied.
    pending_model: RefCell<Option<trd_gui::model::PendingModel>>,
    /// Called to open the shell's `<input type=file>`.
    on_pick_model: Option<js_sys::Function>,
}

impl GuiShared {
    pub fn new(on_pick_model: Option<js_sys::Function>) -> Self {
        Self {
            pending_model: RefCell::new(None),
            on_pick_model,
        }
    }

    pub fn queue_model(&self, model: trd_gui::model::PendingModel) {
        self.pending_model.replace(Some(model));
    }

    /// Asks the shell to open its file picker. A gesture the browser refuses is
    /// the shell's problem to report, not a scene error.
    fn request_pick(&self) {
        if let Some(pick) = self.on_pick_model.as_ref() {
            if let Err(error) = pick.call0(&wasm_bindgen::JsValue::NULL) {
                log::error!("the model file picker could not be opened: {error:?}");
            }
        }
    }
}

/// The interactive viewer application (browser).
pub struct WebApp {
    controller: InteractionController,
    /// The offscreen renderer, shared into the async render task. Wrapped in
    /// `Option` so the async task can **take** it out (single-flight guarantees
    /// exclusive access) and render on the owned value — avoiding a `RefCell`
    /// borrow held across the `.await`.
    renderer: Rc<RefCell<Option<GuiRenderer>>>,
    /// The display texture the latest frame is uploaded into.
    texture: Option<TextureHandle>,
    render_size: (u32, u32),
    /// Set when the scene changed and a re-render must be scheduled.
    needs_render: bool,
    /// `true` while an async render is running (single-flight guard).
    render_in_flight: Rc<Cell<bool>>,
    /// The most recent completed frame, handed back from the async task.
    latest: Rc<RefCell<Option<ImageRgba>>>,
    /// A pending click-to-pick at render-target pixel coords (#141), set on click
    /// and retried each frame until the renderer is free to run the id pass.
    pending_pick: Rc<Cell<Option<(u32, u32)>>>,
    /// `true` while an async pick is running (shares the renderer, so it is
    /// mutually exclusive with a render via the two guards).
    pick_in_flight: Rc<Cell<bool>>,
    /// The most recent completed pick result (`Some(hit)` where `hit` is the
    /// selected object index or `None` for background), applied on the next pass.
    pick_result: Rc<RefCell<Option<Option<u32>>>>,
    /// The JS bridge: the file picker out, the picked model in (#353).
    shared: Rc<GuiShared>,
    /// The most recent model-load failure, shown in the panel until one succeeds.
    model_error: Option<String>,
    /// Meshes whose objects were deleted, waiting for a frame where the renderer
    /// is not out on an async render or pick (#353).
    pending_free: Vec<u32>,
}

impl WebApp {
    /// Builds the app around a controller and an (already-created) offscreen
    /// renderer. The first frame is scheduled on the first `ui` pass.
    pub fn new(
        controller: InteractionController,
        renderer: GuiRenderer,
        shared: Rc<GuiShared>,
    ) -> Self {
        let render_size = renderer.size();
        Self {
            controller,
            renderer: Rc::new(RefCell::new(Some(renderer))),
            texture: None,
            render_size,
            needs_render: true,
            render_in_flight: Rc::new(Cell::new(false)),
            latest: Rc::new(RefCell::new(None)),
            pending_pick: Rc::new(Cell::new(None)),
            pick_in_flight: Rc::new(Cell::new(false)),
            pick_result: Rc::new(RefCell::new(None)),
            shared,
            model_error: None,
            pending_free: Vec::new(),
        }
    }

    /// Frees the meshes of deleted objects, once the renderer is home.
    ///
    /// An async render holds the renderer by value, so a delete during one has
    /// to wait a frame — the scene already stopped drawing the mesh, this is
    /// only the memory.
    fn consume_pending_frees(&mut self) {
        if self.pending_free.is_empty() {
            return;
        }
        let mut slot = self.renderer.borrow_mut();
        let Some(renderer) = slot.as_mut() else {
            return;
        };
        for mesh_id in self.pending_free.drain(..) {
            if renderer.remove_mesh(mesh_id as usize) {
                log::info!("freed mesh {mesh_id}");
            }
        }
    }

    /// Loads a queued model, if the renderer is free to take it.
    ///
    /// The renderer is *moved out* of its cell while a render or pick is in
    /// flight, so the model simply waits for a frame where it is home — no
    /// error, and no scene mutated halfway.
    fn consume_pending_model(&mut self) {
        if self.shared.pending_model.borrow().is_none() {
            return;
        }
        let mut slot = self.renderer.borrow_mut();
        let Some(renderer) = slot.as_mut() else {
            return;
        };
        let Some(request) = self.shared.pending_model.borrow_mut().take() else {
            return;
        };
        match trd_gui::model::load_model(renderer, &mut self.controller.state, &request) {
            Ok(index) => {
                log::info!("loaded '{}' as object {index}", request.name);
                self.model_error = None;
                self.needs_render = true;
            }
            Err(error) => {
                log::error!("{error}");
                self.model_error = Some(error.to_string());
            }
        }
    }

    /// Uploads a completed frame's RGBA into the display texture.
    fn upload(&mut self, ctx: &egui::Context, image: &ImageRgba) {
        let color = egui::ColorImage::from_rgba_unmultiplied(
            [image.width as usize, image.height as usize],
            &image.rgba,
        );
        match &mut self.texture {
            Some(handle) => handle.set(color, TextureOptions::LINEAR),
            None => {
                self.texture = Some(ctx.load_texture("trd-scene", color, TextureOptions::LINEAR));
            }
        }
    }

    /// Spawns an async render of the current scene, storing the result for the
    /// next `ui` pass. Coalesced by the single-flight guard.
    fn schedule_render(&mut self, ctx: &egui::Context) {
        if self.render_in_flight.get() {
            return;
        }
        self.needs_render = false;
        self.render_in_flight.set(true);

        let renderer = self.renderer.clone();
        let latest = self.latest.clone();
        let in_flight = self.render_in_flight.clone();
        let state = self.controller.state.clone();
        let ctx = ctx.clone();
        wasm_bindgen_futures::spawn_local(async move {
            // Take the renderer out of the cell so no borrow is held across the
            // await; single-flight guarantees it is present.
            let Some(mut owned) = renderer.borrow_mut().take() else {
                in_flight.set(false);
                return;
            };
            let result = owned.render(&state).await;
            renderer.borrow_mut().replace(owned);
            match result {
                Ok(image) => *latest.borrow_mut() = Some(image),
                Err(err) => log::error!("wasm scene render failed: {err}"),
            }
            in_flight.set(false);
            ctx.request_repaint();
        });
    }

    /// Spawns an async **pick** for the pending click, if any, when the renderer
    /// is free (no render or pick in flight — both take the shared renderer). The
    /// result is stashed for the next `ui` pass to apply as the selection (#141).
    /// A click made while busy stays pending and is retried next frame.
    fn schedule_pick(&mut self, ctx: &egui::Context) {
        if self.render_in_flight.get() || self.pick_in_flight.get() {
            return;
        }
        let Some((x, y)) = self.pending_pick.get() else {
            return;
        };
        self.pending_pick.set(None);
        self.pick_in_flight.set(true);

        let renderer = self.renderer.clone();
        let result_cell = self.pick_result.clone();
        let in_flight = self.pick_in_flight.clone();
        let pending = self.pending_pick.clone();
        let state = self.controller.state.clone();
        let ctx = ctx.clone();
        wasm_bindgen_futures::spawn_local(async move {
            // If the renderer is momentarily unavailable, requeue the click.
            let Some(mut owned) = renderer.borrow_mut().take() else {
                pending.set(Some((x, y)));
                in_flight.set(false);
                return;
            };
            let hit = owned.pick(&state, x, y).await;
            renderer.borrow_mut().replace(owned);
            *result_cell.borrow_mut() = Some(hit);
            in_flight.set(false);
            ctx.request_repaint();
        });
    }
}

impl eframe::App for WebApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // Upload the most recent completed frame, if any. Take into a local so
        // the `RefMut` is dropped before the `&mut self` upload call.
        let completed = self.latest.borrow_mut().take();
        if let Some(image) = completed {
            self.upload(&ctx, &image);
        }

        // Apply a completed pick as the new selection, re-rendering if it changed
        // (so the selected object's AABB appears / clears).
        let picked = self.pick_result.borrow_mut().take();
        if let Some(hit) = picked {
            if hit != self.controller.state.selected {
                self.controller.state.selected = hit;
                self.needs_render = true;
            }
        }

        // A queued upload is applied before scheduling, so the frame that lands
        // already contains it.
        self.consume_pending_model();

        // Schedule a render when the scene changed (or on the first pass).
        if self.needs_render || self.texture.is_none() {
            self.schedule_render(&ctx);
        }

        let outcome = ui::show(
            ui,
            &mut ui::View {
                controller: &mut self.controller,
                texture: self.texture.as_ref(),
                render_size: self.render_size,
                last_render_ms: None,
                model_error: self.model_error.as_deref(),
            },
        );
        self.needs_render |= outcome.needs_render;

        if outcome.load_model {
            self.shared.request_pick();
        }

        if let Some(mesh_id) = outcome.freed_mesh {
            self.pending_free.push(mesh_id);
        }
        self.consume_pending_frees();

        // A click queued a pick; run it when the renderer is free (retried while busy).
        if let Some(xy) = outcome.pick {
            self.pending_pick.set(Some(xy));
        }
        if self.pending_pick.get().is_some() {
            self.schedule_pick(&ctx);
        }
    }
}
