use tokenizers;

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, ids: &[u32]) -> String;
}

pub struct QwenTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl QwenTokenizer {
    pub fn new(path: &str) -> Self {
        let tokenizer = tokenizers::Tokenizer::from_file(path).expect("Failed to load tokenizer");
        Self { tokenizer }
    }
}

impl Tokenizer for QwenTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .expect("Failed to tokenize text");
        encoding.get_ids().to_vec()
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer
            .decode(ids, true)
            .expect("Failed to detokenize tokens")
    }
}

pub type LlamaTokenizer = QwenTokenizer;
