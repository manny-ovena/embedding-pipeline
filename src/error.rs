use thiserror::Error;

#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("Chunking error: {0}")]
    Chunking(#[from] ChunkingError),

    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),

    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Vector store error: {0}")]
    VectorStore(String),
}

#[derive(Error, Debug)]
pub enum ChunkingError {
    #[error("Invalid chunk configuration: {0}")]
    InvalidConfig(String),

    #[error("Chunking failed: {0}")]
    Failed(String),
}

#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("HTTP request error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Serialization / Deserialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Backend returned error (status {status}): {message}")]
    BackendError { status: u16, message: String },

    #[error("Empty embedding returned by backend")]
    EmptyEmbedding,

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Embedding request timed out")]
    Timeout,

    #[error("Batch size exceeded maximum allowed limit ({max}): requested {requested}")]
    BatchLimitExceeded { requested: usize, max: usize },
}

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Failed to load tokenizer from file {0}")]
    LoadError(String),

    #[error("Encoding error: {0}")]
    EncodeError(String),

    #[error("Decoding error: {0}")]
    DecodeError(String),
}
