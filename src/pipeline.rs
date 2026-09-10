use std::sync::Arc;

use crate::{
    chunker::Chunker,
    embedder::{EmbeddedChunk, Embedder},
    normalize::normalize_text,
    tokenizer::Tokenizer,
};

pub struct EmbeddingPipeline {
    tokenizer: Arc<dyn Tokenizer>,
    chunker: Chunker,
    embedder: Embedder,
}

impl EmbeddingPipeline {
    pub fn new(tokenizer: Arc<dyn Tokenizer>, chunker: Chunker, embedder: Embedder) -> Self {
        Self {
            tokenizer,
            chunker,
            embedder,
        }
    }

    pub async fn run_on_document(&self, doc_id: String, raw_text: &str) -> Vec<EmbeddedChunk> {
        let normalized_text = normalize_text(raw_text);
        let chunks = self.chunker.chunk_text(&normalized_text);
        let embeddings = self
            .embedder
            .embed_batch(
                &chunks
                    .iter()
                    .map(|text| EmbeddedChunk::new(doc_id.clone(), 0, text.clone(), vec![]))
                    .collect::<Vec<_>>(),
            )
            .await;

        let mut results = Vec::new();
        for (index, (chunk, embedding)) in chunks.iter().zip(embeddings.iter()).enumerate() {
            results.push(EmbeddedChunk::new(
                doc_id.clone(),
                index,
                chunk.clone(),
                embedding.clone(),
            ));
        }

        results
    }
}
