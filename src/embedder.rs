use crate::error::EmbeddingError;
use async_trait::async_trait;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{instrument, warn};

#[async_trait]
pub trait EmbeddingBackend: Send + Sync {
    /// Embeds a single text string into a vector.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Embeds a batch of text strings into vectors.
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// Returns the embedding dimension produced by this backend.
    fn dimension(&self) -> usize;
}

#[cfg_attr(test, mockall::automock)]
#[async_trait]
pub trait HttpClient: Send + Sync {
    async fn post_json(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<HttpResponse, reqwest::Error>;
}

#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: StatusCode,
    pub body: String,
}

pub struct ReqwestClient {
    client: reqwest::Client,
}

impl ReqwestClient {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client }
    }
}

impl Default for ReqwestClient {
    fn default() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }
}

#[async_trait]
impl HttpClient for ReqwestClient {
    async fn post_json(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<HttpResponse, reqwest::Error> {
        let response = self.client.post(url).json(body).send().await?;
        let status = response.status();
        let body = response.text().await?;
        Ok(HttpResponse { status, body })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbeddedChunk {
    pub doc_id: String,
    pub chunk_index: usize,
    pub text: String,
    pub embedding: Vec<f32>,
}

impl EmbeddedChunk {
    pub fn new(doc_id: String, chunk_index: usize, text: String, embedding: Vec<f32>) -> Self {
        Self {
            doc_id,
            chunk_index,
            text,
            embedding,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ApiErrorPayload {
    #[serde(default)]
    message: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: EmbeddingInput,
    encoding_format: String,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    #[serde(default)]
    pub data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    pub index: usize,
    pub embedding: Vec<f32>,
}

pub struct VllmBackend {
    base_url: String,
    model_name: String,
    client: Arc<dyn HttpClient>,
    dimension: usize,
    max_batch_size: usize,
    semaphore: Arc<Semaphore>,
}

impl VllmBackend {
    pub fn new(base_url: String, model_name: String, client: Arc<dyn HttpClient>) -> Self {
        Self::with_config(base_url, model_name, client, 4096, 64, 8)
    }

    pub fn with_config(
        base_url: String,
        model_name: String,
        client: Arc<dyn HttpClient>,
        dimension: usize,
        max_batch_size: usize,
        max_concurrency: usize,
    ) -> Self {
        Self {
            base_url,
            model_name,
            client,
            dimension,
            max_batch_size,
            semaphore: Arc::new(Semaphore::new(max_concurrency)),
        }
    }

    async fn execute_request(
        &self,
        input: EmbeddingInput,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let _permit = self.semaphore.acquire().await;
        let url = format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            "v1/embeddings"
        );

        let body = EmbeddingRequest {
            model: self.model_name.clone(),
            input,
            encoding_format: "float".to_string(),
        };
        let body_json = serde_json::to_value(&body)?;

        // Request with retry logic for 429/503
        let mut attempts = 0;
        let max_attempts = 3;
        loop {
            attempts += 1;
            let response = self.client.post_json(&url, &body_json).await?;
            let status = response.status;
            let body_str = response.body;

            if status.is_success() {
                let parsed: EmbeddingResponse = serde_json::from_str(&body_str)?;
                let mut sorted_data = parsed.data;
                sorted_data.sort_by_key(|item| item.index);
                return Ok(sorted_data.into_iter().map(|item| item.embedding).collect());
            }

            if (status == StatusCode::TOO_MANY_REQUESTS
                || status == StatusCode::SERVICE_UNAVAILABLE)
                && attempts < max_attempts
            {
                let delay = std::time::Duration::from_millis(50 * (1 << attempts));
                warn!(
                    "Embedding backend returned {}, retrying attempt {}/{} in {:?}",
                    status, attempts, max_attempts, delay
                );
                tokio::time::sleep(delay).await;
                continue;
            }

            let msg = if let Ok(err_payload) = serde_json::from_str::<ApiErrorPayload>(&body_str) {
                err_payload
                    .message
                    .or(err_payload.error)
                    .unwrap_or(body_str)
            } else {
                body_str
            };

            return Err(EmbeddingError::BackendError {
                status: status.as_u16(),
                message: msg,
            });
        }
    }
}

#[async_trait]
impl EmbeddingBackend for VllmBackend {
    #[instrument(skip(self), fields(model = %self.model_name))]
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut results = self
            .execute_request(EmbeddingInput::Single(text.to_string()))
            .await?;
        results.pop().ok_or(EmbeddingError::EmptyEmbedding)
    }

    #[instrument(skip(self, texts), fields(count = texts.len()))]
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_results = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.max_batch_size) {
            let chunk_results = self
                .execute_request(EmbeddingInput::Batch(chunk.to_vec()))
                .await?;
            all_results.extend(chunk_results);
        }

        Ok(all_results)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

pub type OpenAiBackend = VllmBackend;

/// Deterministic mock backend for offline testing, demos, and benchmarks without a GPU.
pub struct MockBackend {
    dimension: usize,
}

impl MockBackend {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn generate_vector(&self, text: &str) -> Vec<f32> {
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

pub struct Embedder {
    backend: Arc<dyn EmbeddingBackend>,
}

impl Embedder {
    pub fn new(backend: Arc<dyn EmbeddingBackend>) -> Self {
        Self { backend }
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.backend.embed(text).await
    }

    pub async fn embed_batch(
        &self,
        chunks: &[EmbeddedChunk],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        self.backend.embed_batch(&texts).await
    }

    pub async fn embed_strings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.backend.embed_batch(texts).await
    }

    pub fn dimension(&self) -> usize {
        self.backend.dimension()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn successful_http_response() -> HttpResponse {
        HttpResponse {
            status: StatusCode::OK,
            body: serde_json::json!({
                "data": [{
                    "index": 0,
                    "embedding": [-0.25, 0.5]
                }]
            })
            .to_string(),
        }
    }

    #[test]
    fn mock_backend_returns_deterministic_vectors() {
        let backend = MockBackend::new(128);
        let v1 = backend.generate_vector("hello world");
        let v2 = backend.generate_vector("hello world");
        let v3 = backend.generate_vector("different text");

        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
        assert_eq!(v1.len(), 128);
    }

    #[tokio::test]
    async fn embeds_with_expected_request() {
        let mut client = MockHttpClient::new();
        client
            .expect_post_json()
            .withf(|url, body| {
                url == "http://localhost:8000/v1/embeddings"
                    && body
                        == &serde_json::json!({
                            "model": "Qwen3-Embedding-4B",
                            "input": "hello",
                            "encoding_format": "float"
                        })
            })
            .times(1)
            .returning(|_, _| Ok(successful_http_response()));

        let backend = VllmBackend::new(
            "http://localhost:8000/".to_string(),
            "Qwen3-Embedding-4B".to_string(),
            Arc::new(client),
        );

        let embedding = backend.embed("hello").await.unwrap();
        assert_eq!(embedding, vec![-0.25, 0.5]);
    }

    #[tokio::test]
    async fn returns_backend_error_on_http_failure() {
        let mut client = MockHttpClient::new();
        client.expect_post_json().times(1).returning(|_, _| {
            Ok(HttpResponse {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                body: serde_json::json!({
                    "message": "server error"
                })
                .to_string(),
            })
        });

        let backend = VllmBackend::new(
            "http://localhost:8000".to_string(),
            "Qwen3-Embedding-4B".to_string(),
            Arc::new(client),
        );

        let result = backend.embed("hello").await;
        assert!(matches!(
            result,
            Err(EmbeddingError::BackendError { status: 500, .. })
        ));
    }
}
