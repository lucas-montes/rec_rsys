use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rec_rsys::benchmarks::{config, testing_tools::create_matrix};
use rec_rsys::matrix::transpose;

fn transpose_bench(c: &mut Criterion) {
    let mut bench = c.benchmark_group("matrices transpositions");
    config::set_default_benchmark_configs(&mut bench);
    for (rows, cols) in [
        (250, 1000),
        (1000, 25_000),
        (25_000, 100_000),
        (100_000, 250_000),
    ] {
        let m = create_matrix(rows, cols, -1.0, 1.0);
        bench.bench_function(
            BenchmarkId::new("trans", format!("rows{}-cols{}", rows, cols)),
            |b| b.iter(|| transpose(&black_box(m.clone()))),
        );
    }
    bench.finish();
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = config::get_default_profiling_configs();
    targets = transpose_bench
}
#[cfg(target_os = "windows")]
criterion_group!(benches, transpose_bench,);

criterion_main!(benches);
