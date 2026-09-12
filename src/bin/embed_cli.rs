use clap::{Parser, Subcommand, ValueEnum};
use embedding_pipeline::testing::{InMemoryVectorStore, MockBackend};
use embedding_pipeline::{
    ChunkConfig, Chunker, ChunkingStrategy, DistanceMetric, Embedder, EmbeddingBackend,
    EmbeddingPipeline, FastTokenizer, ReqwestClient, SimpleTokenizer, Tokenizer, VectorStore,
    VllmBackend,
};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(
    name = "embed-cli",
    about = "High-throughput text embedding & vector search pipeline CLI",
    version
)]
struct Cli {
    #[arg(short, long, value_enum, default_value_t = BackendType::Mock)]
    backend: BackendType,

    #[arg(long, default_value = "http://localhost:8000")]
    url: String,

    #[arg(long, default_value = "Qwen3-Embedding-4B")]
    model: String,

    #[arg(long)]
    tokenizer_path: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum BackendType {
    Mock,
    Vllm,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Ingests text files or raw text into the vector pipeline
    Ingest {
        #[arg(short, long, num_args = 1..)]
        files: Vec<PathBuf>,
    },
    /// Runs a self-contained interactive demo with sample documents
    Demo,
    /// Demonstrates semantic similarity between queries and a corpus
    Search {
        #[arg(short, long)]
        query: String,
        #[arg(short, long, default_value_t = 3)]
        top_k: usize,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let (tokenizer, backend): (Arc<dyn Tokenizer>, Arc<dyn EmbeddingBackend>) = match cli.backend {
        BackendType::Mock => {
            let tok: Arc<dyn Tokenizer> = Arc::new(SimpleTokenizer::new());
            let be: Arc<dyn EmbeddingBackend> = Arc::new(MockBackend::new(384));
            (tok, be)
        }
        BackendType::Vllm => {
            let client = Arc::new(ReqwestClient::default());
            let tok: Arc<dyn Tokenizer> = if let Some(path) = cli.tokenizer_path {
                Arc::new(FastTokenizer::from_file(path.to_str().unwrap_or_default())?)
            } else {
                Arc::new(SimpleTokenizer::new())
            };
            let be: Arc<dyn EmbeddingBackend> =
                Arc::new(VllmBackend::new(cli.url, cli.model, client));
            (tok, be)
        }
    };

    let config = ChunkConfig::with_strategy(256, 32, ChunkingStrategy::SentenceAware)?;
    let chunker = Chunker::new(config, tokenizer.clone());
    let embedder = Embedder::new(backend);
    let pipeline = EmbeddingPipeline::new(tokenizer, chunker, embedder);

    match cli.command {
        Commands::Demo => {
            println!("=== 🚀 Embedding Pipeline End-to-End Demo ===");
            let docs = vec![
                (
                    "rust-memory".to_string(),
                    "Rust achieves memory safety without a garbage collector via ownership and borrow checking rules enforced at compile time.".to_string(),
                ),
                (
                    "tokio-async".to_string(),
                    "Tokio provides non-blocking async primitives, multithreaded scheduler, and cooperative multitasking in Rust.".to_string(),
                ),
                (
                    "vector-db".to_string(),
                    "Vector databases index high-dimensional embeddings using algorithms like HNSW, IVF-PQ, and Cosine Similarity for semantic search.".to_string(),
                ),
            ];

            let mut store = InMemoryVectorStore::new();
            println!("\n📥 Indexing {} sample documents...", docs.len());
            let count = pipeline.index_documents(docs, &mut store).await?;
            println!(
                "✅ Successfully indexed {} chunks into vector store.\n",
                count
            );

            let queries = vec![
                "How does Rust guarantee memory safety?",
                "Async runtime with work stealing scheduler",
                "Nearest neighbor semantic retrieval algorithms",
            ];

            for query in queries {
                println!("🔍 Query: \"{}\"", query);
                let q_vec = pipeline.embedder().embed(query).await?;
                let results = store.search(&q_vec, 1, DistanceMetric::Cosine).await?;
                if let Some(top) = results.first() {
                    println!(
                        "   ⭐ Top Match: [{}] (Cosine Similarity: {:.4})\n      Text: \"{}\"\n",
                        top.chunk.doc_id, top.score, top.chunk.text
                    );
                }
            }
        }
        Commands::Ingest { files } => {
            let mut docs = Vec::new();
            for file in files {
                if file.is_file() {
                    let content = fs::read_to_string(&file)?;
                    let doc_id = file
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    docs.push((doc_id, content));
                }
            }

            println!("📥 Ingesting {} files...", docs.len());
            let mut store = InMemoryVectorStore::new();
            let count = pipeline.index_documents(docs, &mut store).await?;
            println!("✅ Ingested {} chunks into vector index.", count);
        }
        Commands::Search { query, top_k } => {
            println!("🔍 Embedding query: \"{}\" (top {})", query, top_k);
            let q_vec = pipeline.embedder().embed(&query).await?;
            println!("✅ Generated {}-dimensional query vector.", q_vec.len());
        }
    }

    Ok(())
}
