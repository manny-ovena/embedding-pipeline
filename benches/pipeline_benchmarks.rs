use criterion::{Criterion, black_box, criterion_group, criterion_main};
use embedding_pipeline::{
    ChunkConfig, Chunker, ChunkingStrategy, DistanceMetric, InMemoryVectorStore, SimpleTokenizer,
    cosine_similarity, dot_product, normalize_text,
};
use std::sync::Arc;

fn bench_normalization(c: &mut Criterion) {
    let sample_text = "  The quick brown   fox \n\t jumps over the \u{0000} lazy dog. 🦀   高性能 text processing!  ".repeat(20);
    c.bench_function("normalize_text_500w", |b| {
        b.iter(|| normalize_text(black_box(&sample_text)))
    });
}

fn bench_chunking(c: &mut Criterion) {
    let sample_text = "This is sentence one. This is sentence two. Here is sentence three. \
                       Adding more content to generate large tokens and measure throughput. \
                       Testing sliding window performance under high load."
        .repeat(50);

    let tokenizer = Arc::new(SimpleTokenizer::new());
    let token_chunker = Chunker::new(ChunkConfig::new(64, 16).unwrap(), tokenizer.clone());
    let sentence_chunker = Chunker::new(
        ChunkConfig::with_strategy(64, 0, ChunkingStrategy::SentenceAware).unwrap(),
        tokenizer,
    );

    c.bench_function("chunk_token_sliding_window", |b| {
        b.iter(|| token_chunker.chunk(black_box(&sample_text)))
    });

    c.bench_function("chunk_sentence_aware", |b| {
        b.iter(|| sentence_chunker.chunk(black_box(&sample_text)))
    });
}

fn bench_vector_search(c: &mut Criterion) {
    let mut store = InMemoryVectorStore::new();

    for i in 0..1000 {
        let chunk = embedding_pipeline::EmbeddedChunk::new(
            format!("doc-{}", i),
            0,
            format!("Document content number {}", i),
            (0..384).map(|x| (x as f32 + i as f32).sin()).collect(),
        );
        store.insert(chunk);
    }

    let query: Vec<f32> = (0..384).map(|x| (x as f32).cos()).collect();

    c.bench_function("vector_store_search_1k_top10", |b| {
        b.iter(|| {
            store.search(
                black_box(&query),
                black_box(10),
                black_box(DistanceMetric::Cosine),
            )
        })
    });

    let v1 = vec![0.5f32; 384];
    let v2 = vec![0.5f32; 384];
    c.bench_function("cosine_similarity_384d", |b| {
        b.iter(|| cosine_similarity(black_box(&v1), black_box(&v2)))
    });

    c.bench_function("dot_product_384d", |b| {
        b.iter(|| dot_product(black_box(&v1), black_box(&v2)))
    });
}

criterion_group!(
    benches,
    bench_normalization,
    bench_chunking,
    bench_vector_search
);
criterion_main!(benches);
