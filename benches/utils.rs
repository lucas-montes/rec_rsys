use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rec_rsys::benchmarks::config;
use rec_rsys::utils::{euclidean_norm, squared_diff_sum};

use ndarray::prelude::*;

fn bench(c: &mut Criterion) {
    let mut bench = c.benchmark_group("utils");
    config::set_default_benchmark_configs(&mut bench);
    for x in [100, 250, 1000, 10_000, 50_000, 100_000, 250_000] {
        let dist = Uniform::new(0., 1.);
        let a = Array1::random(x, dist);
        let a2 = Array1::random(x, dist);

        bench.bench_function(BenchmarkId::new("squared_diff_sum", x), |b| {
            b.iter(|| squared_diff_sum(black_box(&a.view()), black_box(&a2.view())))
        });

        bench.bench_function(BenchmarkId::new("euclidean_norm", x), |b| {
            b.iter(|| euclidean_norm(black_box(&a.view())))
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
