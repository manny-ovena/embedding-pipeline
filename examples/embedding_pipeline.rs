use std::sync::Arc;

use embedding_pipeline::chunker::{ChunkConfig, Chunker};
use embedding_pipeline::embedder::{Embedder, EmbeddingBackend, ReqwestClient, VllmBackend};
use embedding_pipeline::pipeline::EmbeddingPipeline;
use embedding_pipeline::tokenizer::QwenTokenizer;

const MODEL_TOKENIZER_PATH: &str = "/srv/ai-models/Qwen/Qwen3-Embedding-4B/tokenizer.json";
const BASE_URL: &str = "http://localhost:8000";
const MODEL_NAME: &str = "Qwen3-Embedding-4B";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Arc::new(ReqwestClient::new(reqwest::Client::new()));
    let tokenizer = Arc::new(QwenTokenizer::new(MODEL_TOKENIZER_PATH));
    let chunker = Chunker::new(
        ChunkConfig {
            max_chunk_size: 512,
            overlap_size: 64,
        },
        tokenizer.clone(),
    );
    let backend: Arc<dyn EmbeddingBackend> = Arc::new(VllmBackend::new(
        BASE_URL.to_string(),
        MODEL_NAME.to_string(),
        client,
    ));
    let embedder = Embedder::new(backend);
    let pipeline = EmbeddingPipeline::new(tokenizer, chunker, embedder);
    let results = pipeline
        .run_on_document(
            "doc-1".to_string(),
            "The quick brown fox jumps over the lazy dog.",
        )
        .await;

    println!("Generated {} chunks", results.len());
    Ok(())
}
