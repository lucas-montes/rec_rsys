use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use ndarray::Array;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rec_rsys::{
    algorithms::knn::KNearestNeighbors, benchmarks::config, models::DatasetBase,
    similarity::SimilarityAlgorithm,
};

fn bench(c: &mut Criterion) {
    let mut bench = c.benchmark_group("knn");
    config::set_default_benchmark_configs(&mut bench);
    for (vector_size, neighbors_pool, num_neighbors) in [
        (250, 2_500, 10),
        (100, 10_000, 50),
        (250, 50_000, 50),
        (100, 500_000, 50),
    ] {
        let records = Array::random((neighbors_pool, vector_size), Uniform::new(0., 1.));
        let dataset = DatasetBase::new(records);

        for algorithm in [
            SimilarityAlgorithm::AdjustedCosineSimilarity,
            SimilarityAlgorithm::EuclideanDistance,
            SimilarityAlgorithm::CosineSimilarity,
            SimilarityAlgorithm::PearsonCorrelation,
            SimilarityAlgorithm::SpearmanCorrelation,
            SimilarityAlgorithm::MSDSimilarity,
        ] {
            let model = KNearestNeighbors::new(dataset.clone())
                .set_num_neighbors(num_neighbors)
                .set_algorithm(algorithm.clone());

            bench.bench_function(
                BenchmarkId::new(
                    "fast_exit",
                    format!(
                        "vector_size{}-neighbors_pool{}-num_neighbors{}-algorithm{:?}",
                        vector_size, neighbors_pool, num_neighbors, algorithm
                    ),
                ),
                |b| b.iter(|| model.predict(0)),
            );

            let model = model.set_early_return_threshold(None);

            bench.bench_function(
                BenchmarkId::new(
                    "all_calculations",
                    format!(
                        "vector_size{}-neighbors_pool{}-num_neighbors{}-algorithm{:?}",
                        vector_size, neighbors_pool, num_neighbors, algorithm
                    ),
                ),
                |b| b.iter(|| model.predict(0)),
            );
        }
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
