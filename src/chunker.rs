use crate::tokenizer::Tokenizer;
use std::sync;

pub struct ChunkConfig {
    pub max_chunk_size: u32,
    pub overlap_size: u32,
}

pub struct Chunker {
    config: ChunkConfig,
    tokenizer: sync::Arc<dyn Tokenizer>,
}

impl Chunker {
    pub fn new(config: ChunkConfig, tokenizer: sync::Arc<dyn Tokenizer>) -> Self {
        Self { config, tokenizer }
    }

    pub fn chunk_text(&self, text: &str) -> Vec<String> {
        let tokens = self.tokenizer.encode(text);
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < tokens.len() {
            let end = (start + self.config.max_chunk_size as usize).min(tokens.len());
            let chunk_tokens = &tokens[start..end];
            let chunk_text = self.tokenizer.decode(chunk_tokens);
            chunks.push(chunk_text);
            if end == tokens.len() {
                break;
            }
            start = end - self.config.overlap_size as usize;
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestTokenizer;

    impl Tokenizer for TestTokenizer {
        fn encode(&self, text: &str) -> Vec<u32> {
            text.split_whitespace()
                .enumerate()
                .map(|(index, _)| index as u32)
                .collect()
        }

        fn decode(&self, ids: &[u32]) -> String {
            let words = ["a", "b", "c", "d", "e", "f"];
            ids.iter()
                .map(|id| words[*id as usize])
                .collect::<Vec<_>>()
                .join(" ")
        }
    }

    #[test]
    fn chunks_text_with_overlap() {
        let tokenizer = sync::Arc::new(TestTokenizer);
        let chunker = Chunker::new(
            ChunkConfig {
                max_chunk_size: 3,
                overlap_size: 1,
            },
            tokenizer,
        );

        let chunks = chunker.chunk_text("a b c d e f");

        assert_eq!(chunks, vec!["a b c", "c d e", "e f"]);
    }

    #[test]
    fn returns_single_chunk_when_text_fits() {
        let tokenizer = sync::Arc::new(TestTokenizer);
        let chunker = Chunker::new(
            ChunkConfig {
                max_chunk_size: 5,
                overlap_size: 1,
            },
            tokenizer,
        );

        let chunks = chunker.chunk_text("a b c");

        assert_eq!(chunks, vec!["a b c"]);
    }
}
