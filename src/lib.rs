pub mod chunker;
pub mod embedder;
pub mod error;
pub mod normalize;
pub mod pipeline;
pub mod tokenizer;
pub mod vector_store;

pub use chunker::{ChunkConfig, Chunker, ChunkingStrategy, TextChunk};
pub use embedder::{
    EmbeddedChunk, Embedder, EmbeddingBackend, MockBackend, ReqwestClient, VllmBackend,
};
pub use error::{ChunkingError, EmbeddingError, PipelineError, TokenizerError};
pub use normalize::{NormalizationOptions, normalize_text, normalize_text_with_options};
pub use pipeline::{EmbeddingPipeline, PipelineBuilder};
pub use tokenizer::{FastTokenizer, QwenTokenizer, SimpleTokenizer, Tokenizer};
pub use vector_store::{
    DistanceMetric, InMemoryVectorStore, SearchResult, cosine_similarity, dot_product,
    euclidean_distance,
};
