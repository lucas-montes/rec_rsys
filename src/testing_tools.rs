use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant};

use crate::statistics::{mean, median, quartiles, standard_deviation};

pub fn timeit<F>(method: F) -> Duration
where
    F: FnOnce(),
{
    let now: Instant = Instant::now();
    method();
    now.elapsed()
}

struct CustomRng {
    state: u64,
}

impl CustomRng {
    fn new(seed: u64) -> CustomRng {
        CustomRng { state: seed }
    }

    fn next(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005) + 1;
        (self.state >> 32) as u32
    }
}

// Function to create a vector with random values within a range
pub fn create_vector(num_elements: usize, min_value: f32, max_value: f32) -> Vec<f32> {
    let mut vec: Vec<f32> = Vec::with_capacity(num_elements);
    let mut rng: CustomRng = CustomRng::new(0);

    for _ in 0..num_elements {
        let value = (rng.next() % (max_value - min_value + 1.0) as u32) as f32 + min_value;
        vec.push(value);
    }

    vec
}

/// # Compare Execution Times
/// Compares the execution times of multiple functions and stores the results.
///
/// ## Parameters:
/// * `n`: The number of times to execute each function.
/// * `functions`: A vector of tuples containing the name and function to be evaluated.
///
/// ## Returns:
/// * A hashmap where the key is the function name and the value is a vector of execution times.
///
/// ## Examples:
/// ```
/// use std::cell::RefCell;
/// use std::rc::Rc;
/// let functions: Vec<(&str, Rc<RefCell<dyn Fn()>>)> = vec![
/// (
///     "Transpose Matrix",
///     Rc::new(RefCell::new(move || {
///         transpose_matrix(&m1);
///     })),
/// ),
/// (
///     "Transpose Matrix 2",
///     Rc::new(RefCell::new(move || {
///         transpose_matrix2(&m2);
///     })),
/// ),
/// ];
/// let results = compare_execution_times(100, functions);
/// ```
pub fn compare_execution_times(n: u64, functions: Vec<(&str, Rc<RefCell<dyn Fn()>>)>) {
    let mut results: HashMap<String, Vec<Duration>> = HashMap::new();

    for (name, function) in functions {
        let mut execution_times: Vec<Duration> = Vec::with_capacity(n as usize);

        for _ in 0..n {
            let execution_time = timeit(|| function.borrow()());
            execution_times.push(execution_time);
        }

        results.insert(name.to_string(), execution_times);
    }

    analyze_execution_results(results);
}

struct FunctionStatistics {
    name: String,
    mean: f32,
    median: f32,
    std_deviation: f32,
    percentile_25: f32,
    percentile_75: f32,
    speedup: Option<f32>,
}

fn analyze_execution_results(results: HashMap<String, Vec<Duration>>) {
    let mut function_stats: Vec<FunctionStatistics> = results
        .iter()
        .map(|(name, duration_times)| {
            let mut execution_times: Vec<f32> = durations_to_f32s(duration_times);
            let (percentile_25, percentile_75) = quartiles(&mut execution_times);

            FunctionStatistics {
                name: name.clone(),
                mean: mean(&execution_times),
                median: median(&execution_times),
                std_deviation: standard_deviation(&execution_times),
                percentile_25,
                percentile_75,
                speedup: None,
            }
        })
        .collect();

    // Sort functions by mean time in ascending order
    function_stats.sort_by(|stats1, stats2| {
        stats1
            .mean
            .partial_cmp(&stats2.mean)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let reference_mean = function_stats
        .first()
        .map(|stats| stats.mean)
        .unwrap_or(0.0);

    // Calculate speedup factor relative to the reference function
    for stats in &mut function_stats {
        if reference_mean != 0.0 {
            let speedup_factor = reference_mean / stats.mean;
            stats.speedup = Some(speedup_factor);
        }
    }

    for (rank, stats) in function_stats.iter().enumerate() {
        let rank = rank + 1;
        println!("Rank {}: Function: {}", rank, stats.name);
        println!("Mean: {:.6} ms", stats.mean * 1000.0);
        println!("Median: {:.6} ms", stats.median * 1000.0);
        // println!("Standard Deviation: {:.6} ms", stats.std_deviation * 1000.0);
        println!("25th Percentile: {:.6} ms", stats.percentile_25 * 1000.0);
        println!("75th Percentile: {:.6} ms", stats.percentile_75 * 1000.0);

        // if let Some(speedup_factor) = stats.speedup {
        //     println!("Speedup: {:.2}x slower", 1.0 - speedup_factor);
        // }

        println!("---------------------------------");
    }
}

fn durations_to_f32s(durations: &Vec<Duration>) -> Vec<f32> {
    durations
        .iter()
        .map(|duration| duration.as_secs_f32())
        .collect()
}

pub fn test_implementation() {
    // for v in [5, 6, 7, 8, 9, 10, 20] {
    //     let mut matrix = Vec::new();
    //     for _ in 0..v {
    //         matrix.push(create_vector(v, -1000.0, 1000.0));
    //     }

    //     let m1: Vec<Vec<f32>> = matrix.clone();
    //     let m2: Vec<Vec<f32>> = matrix.clone();
    //     let _m3: Vec<Vec<f32>> = matrix.clone();
    //     let _m4: Vec<Vec<f32>> = matrix.clone();
    //     let _m5: Vec<Vec<f32>> = matrix.clone();

    //     let functions: Vec<(&str, Rc<RefCell<dyn Fn()>>)> = vec![
    //         (
    //             "covariance",
    //             Rc::new(RefCell::new(move || {
    //                 covariance(&m1);
    //             })),
    //         ),
    //         (
    //             "covariance_matrix",
    //             Rc::new(RefCell::new(move || {
    //                 covariance_matrix(&m2);
    //             })),
    //         ),
    //     ];
    //     println!("------Execution for {:?}-------", v);
    //     compare_execution_times(100, functions);
    //     println!("---------------------------------");
    //     println!("---------------------------------");
    //     println!("---------------------------------");
    // }
}
