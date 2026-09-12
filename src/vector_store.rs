use crate::embedder::EmbeddedChunk;
use crate::error::PipelineError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    Cosine,
    DotProduct,
    Euclidean,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk: EmbeddedChunk,
    pub score: f32,
}

/// Trait defining a vector store interface for indexing and similarity search over embedded chunks.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Inserts a single embedded chunk into the vector store.
    async fn insert(&mut self, chunk: EmbeddedChunk) -> Result<(), PipelineError>;

    /// Inserts a batch of embedded chunks into the vector store.
    async fn insert_batch(&mut self, chunks: Vec<EmbeddedChunk>) -> Result<(), PipelineError> {
        for chunk in chunks {
            self.insert(chunk).await?;
        }
        Ok(())
    }

    /// Searches for the top_k most similar chunks to query_embedding.
    async fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>, PipelineError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{InMemoryVectorStore, cosine_similarity};

    #[test]
    fn test_cosine_similarity() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        let v3 = vec![0.0, 1.0, 0.0];

        assert!((cosine_similarity(&v1, &v2) - 1.0).abs() < 1e-5);
        assert!((cosine_similarity(&v1, &v3) - 0.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_vector_store_search() {
        let mut store = InMemoryVectorStore::new();
        store
            .insert(EmbeddedChunk::new(
                "doc1".to_string(),
                0,
                "Rust async pipeline".to_string(),
                vec![1.0, 0.0],
            ))
            .await
            .unwrap();
        store
            .insert(EmbeddedChunk::new(
                "doc2".to_string(),
                0,
                "Python data science".to_string(),
                vec![0.0, 1.0],
            ))
            .await
            .unwrap();

        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());

        let results = store
            .search(&[1.0, 0.0], 1, DistanceMetric::Cosine)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.doc_id, "doc1");

        store.clear();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[derive(Default)]
    struct CustomVectorStore {
        items: Vec<EmbeddedChunk>,
    }

    #[async_trait]
    impl VectorStore for CustomVectorStore {
        async fn insert(&mut self, chunk: EmbeddedChunk) -> Result<(), PipelineError> {
            self.items.push(chunk);
            Ok(())
        }

        async fn search(
            &self,
            _query_embedding: &[f32],
            top_k: usize,
            _metric: DistanceMetric,
        ) -> Result<Vec<SearchResult>, PipelineError> {
            Ok(self
                .items
                .iter()
                .take(top_k)
                .map(|c| SearchResult {
                    chunk: c.clone(),
                    score: 1.0,
                })
                .collect())
        }
    }

    #[tokio::test]
    async fn test_custom_vector_store() {
        let mut custom = CustomVectorStore::default();
        custom
            .insert_batch(vec![
                EmbeddedChunk::new("c1".into(), 0, "Custom 1".into(), vec![0.1, 0.2]),
                EmbeddedChunk::new("c2".into(), 1, "Custom 2".into(), vec![0.3, 0.4]),
            ])
            .await
            .unwrap();

        let results = custom
            .search(&[0.1, 0.2], 2, DistanceMetric::DotProduct)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.doc_id, "c1");
    }

    #[tokio::test]
    async fn test_trait_object_vector_store() {
        let mut store: Box<dyn VectorStore> = Box::new(InMemoryVectorStore::new());
        store
            .insert(EmbeddedChunk::new(
                "dyn1".to_string(),
                0,
                "Dynamic dispatch".to_string(),
                vec![0.5, 0.5],
            ))
            .await
            .unwrap();

        let results = store
            .search(&[0.5, 0.5], 1, DistanceMetric::Cosine)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.doc_id, "dyn1");
    }
}
