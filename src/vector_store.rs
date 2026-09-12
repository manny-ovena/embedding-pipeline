use crate::embedder::EmbeddedChunk;
use crate::error::PipelineError;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDocument {
    pub chunk: EmbeddedChunk,
}

/// In-memory vector store supporting fast similarity search over embedded chunks.
#[derive(Default, Clone, Debug)]
pub struct InMemoryVectorStore {
    documents: Vec<VectorDocument>,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
        }
    }

    pub fn insert(&mut self, chunk: EmbeddedChunk) {
        self.documents.push(VectorDocument { chunk });
    }

    pub fn insert_batch(&mut self, chunks: Vec<EmbeddedChunk>) {
        self.documents.reserve(chunks.len());
        for chunk in chunks {
            self.insert(chunk);
        }
    }

    pub fn len(&self) -> usize {
        self.documents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    pub fn clear(&mut self) {
        self.documents.clear();
    }

    /// Searches for top_k most similar chunks to query_embedding.
    pub fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>, PipelineError> {
        if self.documents.is_empty() {
            return Ok(Vec::new());
        }

        let mut scored: Vec<SearchResult> = self
            .documents
            .iter()
            .map(|doc| {
                let score = match metric {
                    DistanceMetric::Cosine => {
                        cosine_similarity(query_embedding, &doc.chunk.embedding)
                    }
                    DistanceMetric::DotProduct => {
                        dot_product(query_embedding, &doc.chunk.embedding)
                    }
                    DistanceMetric::Euclidean => {
                        -euclidean_distance(query_embedding, &doc.chunk.embedding)
                    }
                };
                SearchResult {
                    chunk: doc.chunk.clone(),
                    score,
                }
            })
            .collect();

        // Sort descending by score
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);

        Ok(scored)
    }
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-6 || norm_b < 1e-6 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        let v3 = vec![0.0, 1.0, 0.0];

        assert!((cosine_similarity(&v1, &v2) - 1.0).abs() < 1e-5);
        assert!((cosine_similarity(&v1, &v3) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_vector_store_search() {
        let mut store = InMemoryVectorStore::new();
        store.insert(EmbeddedChunk::new(
            "doc1".to_string(),
            0,
            "Rust async pipeline".to_string(),
            vec![1.0, 0.0],
        ));
        store.insert(EmbeddedChunk::new(
            "doc2".to_string(),
            0,
            "Python data science".to_string(),
            vec![0.0, 1.0],
        ));

        let results = store
            .search(&[1.0, 0.0], 1, DistanceMetric::Cosine)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.doc_id, "doc1");
    }
}
