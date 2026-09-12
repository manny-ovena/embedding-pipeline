use crate::error::ChunkingError;
use crate::tokenizer::Tokenizer;
use std::sync::Arc;
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkingStrategy {
    /// Pure token-based sliding window with overlap
    TokenSlidingWindow,
    /// Sentence-boundary aware chunking that avoids splitting sentences across chunks
    SentenceAware,
}

#[derive(Debug, Clone)]
pub struct ChunkConfig {
    pub max_chunk_size: usize,
    pub overlap_size: usize,
    pub strategy: ChunkingStrategy,
}

impl ChunkConfig {
    pub fn new(max_chunk_size: usize, overlap_size: usize) -> Result<Self, ChunkingError> {
        Self::with_strategy(
            max_chunk_size,
            overlap_size,
            ChunkingStrategy::TokenSlidingWindow,
        )
    }

    pub fn with_strategy(
        max_chunk_size: usize,
        overlap_size: usize,
        strategy: ChunkingStrategy,
    ) -> Result<Self, ChunkingError> {
        if max_chunk_size == 0 {
            return Err(ChunkingError::InvalidConfig(
                "max_chunk_size must be greater than 0".to_string(),
            ));
        }
        if overlap_size >= max_chunk_size {
            return Err(ChunkingError::InvalidConfig(
                "overlap_size must be strictly less than max_chunk_size".to_string(),
            ));
        }
        Ok(Self {
            max_chunk_size,
            overlap_size,
            strategy,
        })
    }
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 512,
            overlap_size: 64,
            strategy: ChunkingStrategy::TokenSlidingWindow,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextChunk {
    pub index: usize,
    pub text: String,
    pub token_count: usize,
    pub byte_range: (usize, usize),
}

pub struct Chunker {
    config: ChunkConfig,
    tokenizer: Arc<dyn Tokenizer>,
}

impl Chunker {
    pub fn new(config: ChunkConfig, tokenizer: Arc<dyn Tokenizer>) -> Self {
        Self { config, tokenizer }
    }

    /// Chunks input text according to configured strategy and returns structured chunks.
    pub fn chunk(&self, text: &str) -> Result<Vec<TextChunk>, ChunkingError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        match self.config.strategy {
            ChunkingStrategy::TokenSlidingWindow => self.chunk_token_sliding_window(text),
            ChunkingStrategy::SentenceAware => self.chunk_sentence_aware(text),
        }
    }

    /// Convenience method returning vector of chunk strings.
    pub fn chunk_text(&self, text: &str) -> Result<Vec<String>, ChunkingError> {
        let chunks = self.chunk(text)?;
        Ok(chunks.into_iter().map(|c| c.text).collect())
    }

    fn chunk_token_sliding_window(&self, text: &str) -> Result<Vec<TextChunk>, ChunkingError> {
        let tokens = self
            .tokenizer
            .encode(text)
            .map_err(|e| ChunkingError::Failed(e.to_string()))?;

        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        let mut chunks = Vec::new();
        let mut start = 0;
        let mut index = 0;

        while start < tokens.len() {
            let end = (start + self.config.max_chunk_size).min(tokens.len());
            let chunk_tokens = &tokens[start..end];
            let chunk_text = self
                .tokenizer
                .decode(chunk_tokens)
                .map_err(|e| ChunkingError::Failed(e.to_string()))?;

            chunks.push(TextChunk {
                index,
                text: chunk_text,
                token_count: chunk_tokens.len(),
                byte_range: (0, 0),
            });
            index += 1;

            if end == tokens.len() {
                break;
            }
            start = end.saturating_sub(self.config.overlap_size);
            if start <= chunks.last().map(|_| 0).unwrap_or(0) && chunks.len() > 1 && start == 0 {
                break;
            }
        }

        Ok(chunks)
    }

    fn chunk_sentence_aware(&self, text: &str) -> Result<Vec<TextChunk>, ChunkingError> {
        let sentences: Vec<&str> = text.unicode_sentences().collect();
        if sentences.is_empty() {
            return self.chunk_token_sliding_window(text);
        }

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_tokens = 0usize;
        let mut index = 0;

        for sentence in sentences {
            let sentence_token_count = self
                .tokenizer
                .encode(sentence)
                .map_err(|e| ChunkingError::Failed(e.to_string()))?
                .len();

            if current_tokens + sentence_token_count > self.config.max_chunk_size
                && !current_chunk.is_empty()
            {
                chunks.push(TextChunk {
                    index,
                    text: current_chunk.trim().to_string(),
                    token_count: current_tokens,
                    byte_range: (0, 0),
                });
                index += 1;
                current_chunk.clear();
                current_tokens = 0;
            }

            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(sentence.trim());
            current_tokens += sentence_token_count;
        }

        if !current_chunk.is_empty() {
            chunks.push(TextChunk {
                index,
                text: current_chunk.trim().to_string(),
                token_count: current_tokens,
                byte_range: (0, 0),
            });
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::TokenizerError;

    struct TestTokenizer;

    impl Tokenizer for TestTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
            Ok(text
                .split_whitespace()
                .enumerate()
                .map(|(index, _)| index as u32)
                .collect())
        }

        fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
            let words = ["a", "b", "c", "d", "e", "f", "g", "h", "i"];
            Ok(ids
                .iter()
                .map(|id| words.get(*id as usize).copied().unwrap_or("?"))
                .collect::<Vec<_>>()
                .join(" "))
        }
    }

    #[test]
    fn invalid_config_rejected() {
        assert!(ChunkConfig::new(0, 0).is_err());
        assert!(ChunkConfig::new(10, 10).is_err());
        assert!(ChunkConfig::new(10, 15).is_err());
    }

    #[test]
    fn chunks_text_with_overlap() {
        let tokenizer = Arc::new(TestTokenizer);
        let config = ChunkConfig::new(3, 1).unwrap();
        let chunker = Chunker::new(config, tokenizer);

        let chunks = chunker.chunk_text("a b c d e f").unwrap();

        assert_eq!(chunks, vec!["a b c", "c d e", "e f"]);
    }

    #[test]
    fn returns_single_chunk_when_text_fits() {
        let tokenizer = Arc::new(TestTokenizer);
        let config = ChunkConfig::new(5, 1).unwrap();
        let chunker = Chunker::new(config, tokenizer);

        let chunks = chunker.chunk_text("a b c").unwrap();

        assert_eq!(chunks, vec!["a b c"]);
    }

    #[test]
    fn sentence_aware_chunking() {
        let tokenizer = Arc::new(TestTokenizer);
        let config = ChunkConfig::with_strategy(6, 0, ChunkingStrategy::SentenceAware).unwrap();
        let chunker = Chunker::new(config, tokenizer);

        let text = "First sentence here. Second sentence follows. Third sentence ends.";
        let chunks = chunker.chunk(text).unwrap();

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].index, 0);
    }
}
