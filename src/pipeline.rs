use futures::stream::{self, StreamExt};
use std::sync::Arc;
use tracing::{info, instrument};

use crate::{
    chunker::{ChunkConfig, Chunker},
    embedder::{EmbeddedChunk, Embedder, EmbeddingBackend},
    error::PipelineError,
    normalize::{NormalizationOptions, normalize_text_with_options},
    tokenizer::Tokenizer,
    vector_store::VectorStore,
};

pub struct EmbeddingPipeline {
    tokenizer: Arc<dyn Tokenizer>,
    chunker: Chunker,
    embedder: Embedder,
    norm_options: NormalizationOptions,
}

impl EmbeddingPipeline {
    pub fn new(tokenizer: Arc<dyn Tokenizer>, chunker: Chunker, embedder: Embedder) -> Self {
        Self {
            tokenizer,
            chunker,
            embedder,
            norm_options: NormalizationOptions::default(),
        }
    }

    pub fn builder() -> PipelineBuilder {
        PipelineBuilder::default()
    }

    pub fn tokenizer(&self) -> &Arc<dyn Tokenizer> {
        &self.tokenizer
    }

    pub fn embedder(&self) -> &Embedder {
        &self.embedder
    }

    #[instrument(skip(self, raw_text), fields(doc_id = %doc_id, text_len = raw_text.len()))]
    pub async fn run_on_document(
        &self,
        doc_id: String,
        raw_text: &str,
    ) -> Result<Vec<EmbeddedChunk>, PipelineError> {
        let normalized_text = normalize_text_with_options(raw_text, self.norm_options);
        let chunks = self.chunker.chunk_text(&normalized_text)?;

        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let embeddings = self.embedder.embed_strings(&chunks).await?;

        let mut results = Vec::with_capacity(chunks.len());
        for (index, (chunk, embedding)) in
            chunks.into_iter().zip(embeddings.into_iter()).enumerate()
        {
            results.push(EmbeddedChunk::new(doc_id.clone(), index, chunk, embedding));
        }

        info!(
            "Processed doc_id {} into {} embedded chunks",
            doc_id,
            results.len()
        );
        Ok(results)
    }

    /// Ingests multiple documents in parallel across documents with bounded concurrency.
    pub async fn run_on_documents(
        &self,
        docs: Vec<(String, String)>,
        concurrency: usize,
    ) -> Result<Vec<EmbeddedChunk>, PipelineError> {
        let stream = stream::iter(docs)
            .map(|(doc_id, text)| async move { self.run_on_document(doc_id, &text).await })
            .buffer_unordered(concurrency.max(1));

        let mut all_chunks = Vec::new();
        let mut stream = Box::pin(stream);
        while let Some(res) = stream.next().await {
            all_chunks.extend(res?);
        }

        Ok(all_chunks)
    }

    /// Ingests documents directly into a vector store.
    pub async fn index_documents<S: VectorStore + ?Sized>(
        &self,
        docs: Vec<(String, String)>,
        store: &mut S,
    ) -> Result<usize, PipelineError> {
        let chunks = self.run_on_documents(docs, 4).await?;
        let count = chunks.len();
        store.insert_batch(chunks).await?;
        Ok(count)
    }
}

#[derive(Default)]
pub struct PipelineBuilder {
    tokenizer: Option<Arc<dyn Tokenizer>>,
    chunk_config: Option<ChunkConfig>,
    backend: Option<Arc<dyn EmbeddingBackend>>,
    norm_options: NormalizationOptions,
}

impl PipelineBuilder {
    pub fn tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    pub fn chunk_config(mut self, config: ChunkConfig) -> Self {
        self.chunk_config = Some(config);
        self
    }

    pub fn backend(mut self, backend: Arc<dyn EmbeddingBackend>) -> Self {
        self.backend = Some(backend);
        self
    }

    pub fn normalization(mut self, options: NormalizationOptions) -> Self {
        self.norm_options = options;
        self
    }

    pub fn build(self) -> Result<EmbeddingPipeline, PipelineError> {
        let tokenizer = self
            .tokenizer
            .ok_or_else(|| PipelineError::VectorStore("Tokenizer is required".to_string()))?;
        let chunk_config = self.chunk_config.unwrap_or_default();
        let chunker = Chunker::new(chunk_config, tokenizer.clone());
        let backend = self.backend.ok_or_else(|| {
            PipelineError::VectorStore("Embedding backend is required".to_string())
        })?;
        let embedder = Embedder::new(backend);

        Ok(EmbeddingPipeline {
            tokenizer,
            chunker,
            embedder,
            norm_options: self.norm_options,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockBackend;
    use crate::tokenizer::SimpleTokenizer;
    use crate::vector_store::{DistanceMetric, SearchResult};
    use async_trait::async_trait;

    #[derive(Default)]
    struct RecordingVectorStore {
        received_chunks: Vec<EmbeddedChunk>,
    }

    #[async_trait]
    impl VectorStore for RecordingVectorStore {
        async fn insert(&mut self, chunk: EmbeddedChunk) -> Result<(), PipelineError> {
            self.received_chunks.push(chunk);
            Ok(())
        }

        async fn search(
            &self,
            _query: &[f32],
            _top_k: usize,
            _metric: DistanceMetric,
        ) -> Result<Vec<SearchResult>, PipelineError> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn test_index_documents_with_custom_store() {
        let tokenizer = Arc::new(SimpleTokenizer::new());
        let backend = Arc::new(MockBackend::new(128));
        let chunk_config = ChunkConfig::new(64, 16).unwrap();
        let chunker = Chunker::new(chunk_config, tokenizer.clone());
        let embedder = Embedder::new(backend);
        let pipeline = EmbeddingPipeline::new(tokenizer, chunker, embedder);

        let docs = vec![
            ("doc1".to_string(), "This is test document one.".to_string()),
            ("doc2".to_string(), "This is test document two.".to_string()),
        ];

        let mut custom = RecordingVectorStore::default();
        let count = pipeline.index_documents(docs, &mut custom).await.unwrap();
        assert_eq!(count, 2);
        assert_eq!(custom.received_chunks.len(), 2);
        assert_eq!(custom.received_chunks[0].doc_id, "doc1");
        assert_eq!(custom.received_chunks[1].doc_id, "doc2");
    }
}
