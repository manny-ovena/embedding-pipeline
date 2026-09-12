use crate::embedder::{EmbeddedChunk, EmbeddingBackend};
use crate::error::{EmbeddingError, PipelineError};
use crate::vector_store::{DistanceMetric, SearchResult, VectorStore};
use async_trait::async_trait;

/// Deterministic mock backend for offline testing, demos, and benchmarks without a GPU.
#[derive(Debug, Clone)]
pub struct MockBackend {
    dimension: usize,
}

impl MockBackend {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    pub fn generate_vector(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.dimension];
        if text.is_empty() {
            return vec;
        }

        for (i, word) in text.split_whitespace().enumerate() {
            let hash = word
                .bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            let idx = (hash as usize) % self.dimension;
            let weight = 1.0 / ((i + 1) as f32).sqrt();
            vec[idx] += weight;
        }

        // Normalize to unit L2 norm
        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for v in &mut vec {
                *v /= norm;
            }
        }

        vec
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new(384)
    }
}

#[async_trait]
impl EmbeddingBackend for MockBackend {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        Ok(self.generate_vector(text))
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        Ok(texts.iter().map(|t| self.generate_vector(t)).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
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

/// In-memory vector store intended for testing, demos, and benchmarks.
#[derive(Default, Clone, Debug)]
pub struct InMemoryVectorStore {
    documents: Vec<EmbeddedChunk>,
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self {
            documents: Vec::new(),
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
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn insert(&mut self, chunk: EmbeddedChunk) -> Result<(), PipelineError> {
        self.documents.push(chunk);
        Ok(())
    }

    async fn insert_batch(&mut self, chunks: Vec<EmbeddedChunk>) -> Result<(), PipelineError> {
        self.documents.reserve(chunks.len());
        for chunk in chunks {
            self.documents.push(chunk);
        }
        Ok(())
    }

    async fn search(
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
            .map(|chunk| {
                let score = match metric {
                    DistanceMetric::Cosine => cosine_similarity(query_embedding, &chunk.embedding),
                    DistanceMetric::DotProduct => dot_product(query_embedding, &chunk.embedding),
                    DistanceMetric::Euclidean => {
                        -euclidean_distance(query_embedding, &chunk.embedding)
                    }
                };
                SearchResult {
                    chunk: chunk.clone(),
                    score,
                }
            })
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);

        Ok(scored)
    }
}
