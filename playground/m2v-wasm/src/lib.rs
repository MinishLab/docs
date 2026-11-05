use model2vec_rs::model::StaticModel;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Model(StaticModel);

#[wasm_bindgen]
impl Model {
    #[wasm_bindgen(constructor)]
    pub fn new(tokenizer: &[u8], weights: &[u8], config: &[u8]) -> Result<Model, JsError> {
        StaticModel::from_bytes(tokenizer, weights, config, None)
            .map(Model)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    pub fn encode(&self, texts: Vec<String>) -> Vec<f32> {
        self.0.encode(&texts).into_iter().flatten().collect()
    }
}
