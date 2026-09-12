use crate::error::TokenizerError;
use tokenizers;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>;
    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError>;
}

/// Tokenizer backed by the HuggingFace `tokenizers` library (Qwen, Llama, BERT, etc.)
pub struct FastTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl FastTokenizer {
    pub fn from_file(path: &str) -> Result<Self, TokenizerError> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| TokenizerError::LoadError(format!("{}: {}", path, e)))?;
        Ok(Self { tokenizer })
    }

    pub fn from_tokenizer(tokenizer: tokenizers::Tokenizer) -> Self {
        Self { tokenizer }
    }
}

impl Tokenizer for FastTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| TokenizerError::EncodeError(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| TokenizerError::DecodeError(e.to_string()))
    }
}

pub type QwenTokenizer = FastTokenizer;
pub type LlamaTokenizer = FastTokenizer;

/// Standalone whitespace & punctuation tokenizer for zero-dependency local runs and testing.
#[derive(Default, Clone)]
pub struct SimpleTokenizer;

impl SimpleTokenizer {
    pub fn new() -> Self {
        Self
    }
}

impl Tokenizer for SimpleTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        // Deterministic hash-based token IDs from words
        let ids = text
            .split_whitespace()
            .enumerate()
            .map(|(i, word)| {
                let hash = word
                    .bytes()
                    .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
                (hash & 0x7FFF) | ((i as u32 % 100) << 15)
            })
            .collect();
        Ok(ids)
    }

    fn decode(&self, _ids: &[u32]) -> Result<String, TokenizerError> {
        Ok(String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokenizer() {
        let tok = SimpleTokenizer::new();
        let encoded = tok.encode("hello world").expect("should encode");
        assert_eq!(encoded.len(), 2);
    }
}
