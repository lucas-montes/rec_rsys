use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rec_rsys::benchmarks::{config, testing_tools::create_vector};
use rec_rsys::utils::{dot, euclidean_norm, squared_diff_sum};


fn utils_bench(c: &mut Criterion) {
    let mut bench = c.benchmark_group("utils");
    config::set_default_benchmark_configs(&mut bench);
    for x in [10, 25, 50, 100, 250, 1000, 10000] {
        let m = create_vector(x,-1.0, 1.0);
        bench.bench_function(BenchmarkId::new("dot-product", x), |b| {
            b.iter(|| dot(&black_box(m.clone()),&black_box(m.clone())))
        });
    }
    bench.finish();
}

#[cfg(not(target_os = "windows"))]
criterion_group! {
    name = benches;
    config = config::get_default_profiling_configs();
    targets = utils_bench
}
#[cfg(target_os = "windows")]
criterion_group!(benches, utils_bench,);

criterion_main!(benches);
