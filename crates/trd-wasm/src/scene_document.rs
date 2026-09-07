use js_sys::{Array, Map, Uint8Array};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use trd_core::{GlbMesh, Matrix4, ModelEdit, SceneDocument};

/// An editable source document. Matrix access uses column-major renderer values;
/// export restores the source adapter's original column layout.
#[wasm_bindgen(js_name = ArrowSceneDocument)]
pub struct ArrowSceneDocument {
    pub(crate) inner: Rc<RefCell<SceneDocument>>,
}

#[wasm_bindgen(js_class = ArrowSceneDocument)]
impl ArrowSceneDocument {
    #[wasm_bindgen(js_name = fromArrow)]
    pub fn from_arrow(bytes: &[u8]) -> Result<ArrowSceneDocument, JsValue> {
        let inner = SceneDocument::read(bytes).map_err(crate::js_error)?;
        inner.frames().map_err(crate::js_error)?;
        Ok(Self {
            inner: Rc::new(RefCell::new(inner)),
        })
    }

    #[wasm_bindgen(js_name = fromArrowWithGlb)]
    pub fn from_arrow_with_glb(arrow: &[u8], glb: &[u8]) -> Result<ArrowSceneDocument, JsValue> {
        let document = Self::from_arrow(arrow)?;
        document
            .inner
            .borrow_mut()
            .bind_glb(glb)
            .map_err(crate::js_error)?;
        Ok(document)
    }

    #[wasm_bindgen(js_name = fromArrowWithMesh)]
    pub fn from_arrow_with_mesh(
        arrow: &[u8],
        mesh_id: &str,
        glb: &[u8],
    ) -> Result<ArrowSceneDocument, JsValue> {
        let document = Self::from_arrow(arrow)?;
        document
            .inner
            .borrow_mut()
            .bind_meshes(vec![GlbMesh::new(mesh_id, glb).map_err(crate::js_error)?])
            .map_err(crate::js_error)?;
        Ok(document)
    }

    /// `bindings` is a Map<string UUID, Uint8Array>, not an object-index map.
    #[wasm_bindgen(js_name = fromArrowWithGlbs)]
    pub fn from_arrow_with_glbs(
        arrow: &[u8],
        bindings: &Map,
    ) -> Result<ArrowSceneDocument, JsValue> {
        let document = Self::from_arrow(arrow)?;
        let mut meshes = Vec::new();
        for entry in bindings.entries() {
            let entry = Array::from(&entry?);
            let id = entry
                .get(0)
                .as_string()
                .ok_or_else(|| crate::js_error("mesh map keys must be UUID strings"))?;
            let bytes = entry
                .get(1)
                .dyn_into::<Uint8Array>()
                .map_err(|_| crate::js_error("mesh map values must be Uint8Array GLB bytes"))?;
            meshes.push(GlbMesh::new(&id, &bytes.to_vec()).map_err(crate::js_error)?);
        }
        // Stable implicit draw order, independent of a caller's map insertion order.
        meshes.sort_by_key(GlbMesh::id);
        document
            .inner
            .borrow_mut()
            .bind_meshes(meshes)
            .map_err(crate::js_error)?;
        Ok(document)
    }

    #[wasm_bindgen(js_name = frameCount)]
    pub fn frame_count(&self) -> Result<u32, JsValue> {
        u32::try_from(self.inner.borrow().row_count())
            .map_err(|_| crate::js_error("too many params rows for this browser API"))
    }

    #[wasm_bindgen(js_name = meshIds)]
    pub fn mesh_ids(&self) -> Vec<String> {
        self.inner
            .borrow()
            .meshes()
            .iter()
            .map(|mesh| mesh.id().to_string())
            .collect()
    }

    #[wasm_bindgen(js_name = objectCount)]
    pub fn object_count(&self, row: u32) -> Result<u32, JsValue> {
        let frame = self
            .inner
            .borrow()
            .frame(row as usize)
            .map_err(crate::js_error)?;
        u32::try_from(frame.objects.len()).map_err(|_| crate::js_error("too many objects"))
    }

    #[wasm_bindgen(js_name = getModel)]
    pub fn get_model(&self, row: u32, object: u32) -> Result<Vec<f32>, JsValue> {
        let frame = self
            .inner
            .borrow()
            .frame(row as usize)
            .map_err(crate::js_error)?;
        let instance = frame
            .objects
            .get(object as usize)
            .ok_or_else(|| crate::js_error("params row or object is out of range"))?;
        Ok(instance.model.to_cols_array().to_vec())
    }

    #[wasm_bindgen(js_name = setModel)]
    pub fn set_model(&mut self, row: u32, object: u32, values: &[f32]) -> Result<(), JsValue> {
        let values: &[f32; 16] = values
            .try_into()
            .map_err(|_| crate::js_error("a model matrix needs exactly 16 float32 values"))?;
        self.inner
            .borrow_mut()
            .apply_model_edits(&[ModelEdit {
                row: row as usize,
                object: object as usize,
                model: Matrix4::from_cols_array(values),
            }])
            .map_err(crate::js_error)
    }

    #[wasm_bindgen(js_name = exportArrow)]
    pub fn export_arrow(&self) -> Result<Vec<u8>, JsValue> {
        self.inner.borrow().write().map_err(crate::js_error)
    }
}

pub(crate) fn document_renderer(
    document: &SceneDocument,
    gpu: std::sync::Arc<trd_core::GpuContext>,
    format: wgpu::TextureFormat,
    pbr: Option<&crate::PbrState>,
    env: Option<&trd_core::EnvMapData>,
) -> Result<trd_core::Renderer, JsValue> {
    let assets = trd_placement::document_assets(document).map_err(crate::js_error)?;
    let mut renderer =
        trd_core::Renderer::with_assets(gpu, format, &assets).map_err(crate::js_error)?;
    if let Some(pbr) = pbr {
        pbr.apply(&mut renderer);
    }
    if let Some(operator) = document.tonemap_override().map_err(crate::js_error)? {
        renderer.set_tonemap_operator(trd_core::MeshTarget::All, operator);
    }
    if let Some(env) = env {
        renderer.set_env_map(env.clone());
    }
    Ok(renderer)
}
