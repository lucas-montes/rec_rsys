use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rec_rsys::benchmarks::config;
use rec_rsys::similarity::{
    cosine_similarity, euclidean_distance, pearson_correlation_uncentered,
    spearman_correlation,
};

use ndarray::prelude::*;

fn bench(c: &mut Criterion) {
    let mut bench = c.benchmark_group("similarity");
    config::set_default_benchmark_configs(&mut bench);
    for x in [100, 250, 1000, 10_000, 50_000, 100_000, 250_000] {
        let dist = Uniform::new(0., 1.);
        let a = Array1::random(x, dist);
        let a2 = Array1::random(x, dist);

        bench.bench_function(
            BenchmarkId::new("pearson_correlation_uncentered", x),
            |b| {
                b.iter(|| {
                    pearson_correlation_uncentered(
                        black_box(&a.view()),
                        black_box(&a2.view()),
                    )
                })
            },
        );

        bench.bench_function(BenchmarkId::new("spearman_correlation", x), |b| {
            b.iter(|| spearman_correlation(black_box(&a.view()), black_box(&a2.view())))
        });

        bench.bench_function(BenchmarkId::new("cosine_similarity", x), |b| {
            b.iter(|| cosine_similarity(black_box(&a.view()), black_box(&a2.view())))
        });

        bench.bench_function(BenchmarkId::new("euclidean_distance", x), |b| {
            b.iter(|| euclidean_distance(black_box(&a.view()), black_box(&a2.view())))
        });
    }
    bench.finish();
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = config::get_default_profiling_configs();
    targets = bench
}
#[cfg(target_os = "windows")]
criterion_group!(benches, bench,);

criterion_main!(benches);
