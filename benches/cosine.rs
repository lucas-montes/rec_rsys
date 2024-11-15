use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rec_rsys::benchmarks::{config, testing_tools::create_vector};
use rec_rsys::utils::{dot, euclidean_norm, squared_diff_sum};

use ndarray::prelude::*;

fn utils_bench(c: &mut Criterion) {
    let mut bench = c.benchmark_group("utils");
    config::set_default_benchmark_configs(&mut bench);
    for x in [100, 250, 1000, 10_000, 50_000, 100_000, 250_000] {
        let m = create_vector(x, -1.0, 1.0);
        let m2 = create_vector(x, -1.0, 1.0);
        bench.bench_function(BenchmarkId::new("dot-product", x), |b| {
            b.iter(|| dot(&black_box(m.clone()), &black_box(m2.clone())))
        });

        let a = Array1::from_vec(m);
        let a2 = Array1::from_vec(m2);

        bench.bench_function(BenchmarkId::new("dot-product-ndarray", x), |b| {
            b.iter(|| a.dot(&a2))
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
