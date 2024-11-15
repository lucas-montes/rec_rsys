use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rec_rsys::algorithms::knn::KNN;
use rec_rsys::benchmarks::{config, testing_tools::create_vector};
use rec_rsys::models::Item;

fn knn_bench(c: &mut Criterion) {
    let mut bench = c.benchmark_group("knn");
    config::set_default_benchmark_configs(&mut bench);
    for (vector_size, neighbors_pool, num_neighbors) in
        [(250, 250, 10), (1_000, 1_000, 50), (500, 5_000, 50)]
    {
        let m = create_vector(vector_size, -1.0, 1.0);
        let item = Item::new(0, m, None);
        let items = (0..neighbors_pool)
            .map(|i| Item::new(i + 1, create_vector(vector_size, -1.0, 1.0), None))
            .collect();
        let result = KNN::new(item, items).set_num_neighbors(num_neighbors);
        bench.bench_function(
            BenchmarkId::new(
                "result",
                format!(
                    "vector_size{}-neighbors_pool{}-num_neighbors{}",
                    vector_size, neighbors_pool, num_neighbors
                ),
            ),
            |b| b.iter(|| result.result()),
        );
    }
    bench.finish();
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = config::get_default_profiling_configs();
    targets = knn_bench
}
#[cfg(target_os = "windows")]
criterion_group!(benches, knn_bench,);

criterion_main!(benches);
