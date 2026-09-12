use std::path::Path;
use std::sync::Arc;

use embedding_pipeline::{
    ChunkConfig, Chunker, ChunkingStrategy, DistanceMetric, Embedder, EmbeddingBackend,
    EmbeddingPipeline, FastTokenizer, InMemoryVectorStore, MockBackend, ReqwestClient,
    SimpleTokenizer, Tokenizer, VllmBackend,
};

const MODEL_TOKENIZER_PATH: &str = "/srv/ai-models/Qwen/Qwen3-Embedding-4B/tokenizer.json";
const BASE_URL: &str = "http://localhost:8000";
const MODEL_NAME: &str = "Qwen3-Embedding-4B";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== High-Throughput Embedding & Semantic Search Pipeline ===");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(500))
        .build()?;

    let vllm_healthy = client
        .get(format!("{}/health", BASE_URL))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    // Determine whether to use live vLLM backend or deterministic offline Mock backend
    let (tokenizer, backend): (Arc<dyn Tokenizer>, Arc<dyn EmbeddingBackend>) = if vllm_healthy
        && Path::new(MODEL_TOKENIZER_PATH).exists()
    {
        println!("Connected to active vLLM backend on {}", BASE_URL);
        let http_client = Arc::new(ReqwestClient::default());
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(FastTokenizer::from_file(MODEL_TOKENIZER_PATH)?);
        let backend: Arc<dyn EmbeddingBackend> = Arc::new(VllmBackend::new(
            BASE_URL.to_string(),
            MODEL_NAME.to_string(),
            http_client,
        ));
        (tokenizer, backend)
    } else {
        println!(
            "vLLM server not reachable at {} - running with deterministic Mock backend (offline demo)",
            BASE_URL
        );
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(SimpleTokenizer::new());
        let backend: Arc<dyn EmbeddingBackend> = Arc::new(MockBackend::new(384));
        (tokenizer, backend)
    };

    let chunk_config = ChunkConfig::with_strategy(256, 32, ChunkingStrategy::SentenceAware)?;
    let chunker = Chunker::new(chunk_config, tokenizer.clone());
    let embedder = Embedder::new(backend);

    let pipeline = EmbeddingPipeline::new(tokenizer, chunker, embedder);

    let sample_documents = vec![
        (
            "rust-intro".to_string(),
            "Rust is a systems programming language focused on safety, speed, and concurrency. \
             It prevents segfaults and guarantees thread safety without garbage collection."
                .to_string(),
        ),
        (
            "async-tokio".to_string(),
            "Tokio is an asynchronous runtime for Rust. It provides multi-threaded work-stealing \
             schedulers, timers, and non-blocking I/O abstractions for high-throughput network services."
                .to_string(),
        ),
        (
            "vllm-backend".to_string(),
            "vLLM is a high-throughput and memory-efficient LLM serving engine. It uses PagedAttention \
             for fast GPU memory management and batched inference."
                .to_string(),
        ),
    ];

    println!("\nIndexing {} sample documents...", sample_documents.len());
    let mut vector_store = InMemoryVectorStore::new();
    let indexed_count = pipeline
        .index_documents(sample_documents, &mut vector_store)
        .await?;
    println!(
        "Successfully indexed {} chunks into vector store.",
        indexed_count
    );

    // Perform a sample semantic query
    let query = "high performance async runtime";
    println!("\nRunning semantic search query: \"{}\"", query);
    let query_vector = pipeline.embedder().embed(query).await?;
    let search_results = vector_store.search(&query_vector, 2, DistanceMetric::Cosine)?;

    println!("\nTop Search Results:");
    for (rank, result) in search_results.iter().enumerate() {
        println!(
            "  [{}] Doc ID: '{}' (Score: {:.4})\n      Text: \"{}\"",
            rank + 1,
            result.chunk.doc_id,
            result.score,
            result.chunk.text
        );
    }

    Ok(())
}
