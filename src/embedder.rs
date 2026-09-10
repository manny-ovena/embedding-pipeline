use async_trait::async_trait;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[async_trait]
pub trait EmbeddingBackend: Send + Sync {
    async fn embed(&self, chunk: &EmbeddedChunk) -> Vec<f32>;
    async fn embed_batch(&self, chunks: &[EmbeddedChunk]) -> Vec<Vec<f32>>;
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

#[derive(Debug)]
pub struct HttpResponse {
    status: StatusCode,
    body: String,
}

pub struct ReqwestClient {
    client: reqwest::Client,
}

impl ReqwestClient {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client }
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

#[derive(Clone, Debug)]
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
struct EmbeddingError {
    code: u32,
    kind: String,
    message: String,
}

pub struct Embedder {
    backend: Arc<dyn EmbeddingBackend>,
}

impl Embedder {
    pub fn new(backend: Arc<dyn EmbeddingBackend>) -> Self {
        Self { backend }
    }

    pub async fn embed_batch(&self, chunks: &[EmbeddedChunk]) -> Vec<Vec<f32>> {
        self.backend.embed_batch(chunks).await
    }
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: String,
    encoding_format: String,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    data: Vec<EmbeddingItem>,
    usage: EmbeddingUsage,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    index: usize,
    object: String,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct EmbeddingUsage {
    prompt_tokens: u32,
    total_tokens: u32,
    completion_tokens: u32,
    prompt_tokens_details: Option<serde_json::Value>,
}

pub struct VllmBackend {
    base_url: String,
    model_name: String,
    client: Arc<dyn HttpClient>,
}

impl VllmBackend {
    pub fn new(base_url: String, model_name: String, client: Arc<dyn HttpClient>) -> Self {
        Self {
            base_url,
            model_name,
            client,
        }
    }

    async fn post_json(&self, path: &str, body: &serde_json::Value) -> serde_json::Value {
        let url = format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        );
        let response = self
            .client
            .post_json(&url, body)
            .await
            .expect("Embedding request failed");
        let status = response.status;
        let body = response.body;
        dbg!(&body);

        if !status.is_success() {
            let error: EmbeddingError =
                serde_json::from_str(&body).expect("Failed to parse response error");
            panic!("Generate request failed with status {status}: {error:?}");
        }

        serde_json::from_str(&body).expect("Failed to parse JSON response")
    }
}

#[async_trait]
impl EmbeddingBackend for VllmBackend {
    async fn embed(&self, chunk: &EmbeddedChunk) -> Vec<f32> {
        let body = EmbeddingRequest {
            model: self.model_name.clone(),
            input: chunk.text.clone(),
            encoding_format: "float".to_string(),
        };
        let body = serde_json::to_value(&body).expect("Failed to serialize embedding request");
        let response = self.post_json("/v1/embeddings", &body).await;
        let response: EmbeddingResponse =
            serde_json::from_value(response).expect("Failed to parse embedding response");

        response
            .data
            .first()
            .map(|item| item.embedding.clone())
            .unwrap_or_default()
    }

    async fn embed_batch(&self, chunks: &[EmbeddedChunk]) -> Vec<Vec<f32>> {
        let mut results = Vec::new();
        for chunk in chunks {
            results.push(self.embed(chunk).await);
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn successful_http_response() -> HttpResponse {
        HttpResponse {
            status: StatusCode::OK,
            body: serde_json::json!({
                "id": "embd-99a96378ed33115e",
                "object": "list",
                "created": 1789062205_u64,
                "model": "Qwen3-Embedding-4B",
                "data": [{
                    "index": 0,
                    "object": "embedding",
                    "embedding": [-0.25, 0.5]
                }],
                "usage": {
                    "prompt_tokens": 11,
                    "total_tokens": 11,
                    "completion_tokens": 0,
                    "prompt_tokens_details": null
                }
            })
            .to_string(),
        }
    }

    fn chunk(text: &str) -> EmbeddedChunk {
        EmbeddedChunk::new("doc-1".to_string(), 0, text.to_string(), Vec::new())
    }

    #[test]
    fn deserializes_embedding_response() {
        let response: EmbeddingResponse = serde_json::from_value(serde_json::json!({
            "id": "embd-99a96378ed33115e",
            "object": "list",
            "created": 1789062205,
            "model": "Qwen3-Embedding-4B",
            "data": [{
                "index": 0,
                "object": "embedding",
                "embedding": [-0.0003177309990860522, -0.02356986328959465]
            }],
            "usage": {
                "prompt_tokens": 11,
                "total_tokens": 11,
                "completion_tokens": 0,
                "prompt_tokens_details": null
            }
        }))
        .expect("response should match the embedding API shape");

        assert_eq!(response.id, "embd-99a96378ed33115e");
        assert_eq!(response.object, "list");
        assert_eq!(response.created, 1789062205);
        assert_eq!(response.model, "Qwen3-Embedding-4B");
        assert_eq!(response.data[0].index, 0);
        assert_eq!(response.data[0].object, "embedding");
        assert_eq!(response.data[0].embedding.len(), 2);
        assert_eq!(response.usage.prompt_tokens, 11);
        assert_eq!(response.usage.total_tokens, 11);
        assert_eq!(response.usage.completion_tokens, 0);
        assert!(response.usage.prompt_tokens_details.is_none());
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

        let embedding = backend.embed(&chunk("hello")).await;

        assert_eq!(embedding, vec![-0.25, 0.5]);
    }

    #[tokio::test]
    #[should_panic(expected = "Generate request failed with status 500 Internal Server Error")]
    async fn panics_on_http_error() {
        let mut client = MockHttpClient::new();
        client.expect_post_json().times(1).returning(|_, _| {
            Ok(HttpResponse {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                body: serde_json::json!({
                    "code": 500,
                    "kind": "server_error",
                    "message": "embedding failed"
                })
                .to_string(),
            })
        });
        let backend = VllmBackend::new(
            "http://localhost:8000".to_string(),
            "Qwen3-Embedding-4B".to_string(),
            Arc::new(client),
        );

        backend.embed(&chunk("hello")).await;
    }

    #[tokio::test]
    async fn embeds_each_chunk_in_batch() {
        let mut client = MockHttpClient::new();
        client
            .expect_post_json()
            .times(2)
            .returning(|_, _| Ok(successful_http_response()));
        let backend = VllmBackend::new(
            "http://localhost:8000".to_string(),
            "Qwen3-Embedding-4B".to_string(),
            Arc::new(client),
        );
        let chunks = vec![chunk("first"), chunk("second")];

        let embeddings = backend.embed_batch(&chunks).await;

        assert_eq!(embeddings, vec![vec![-0.25, 0.5], vec![-0.25, 0.5]]);
    }
}
