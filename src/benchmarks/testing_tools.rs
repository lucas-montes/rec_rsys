use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
    fn default() -> CustomRng {
        CustomRng {
            state: CustomRng::get_current_time(),
        }
    }

    fn new() -> CustomRng {
        CustomRng::default()
    }

    fn seed(mut self, seed: u64) -> Self {
        self.state = seed;
        self
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        let current_time = CustomRng::get_current_time();
        let time_string = current_time.to_string();
        let digits = time_string
            .chars()
            .filter_map(|c| c.to_digit(10))
            .collect::<Vec<u32>>();
        for chunk in digits.chunks(2) {
            let r = chunk.iter().sum::<u32>();
            if r < 12 {
                x ^= x << r
            } else {
                x ^= x >> r
            }
        }
        self.state = x;
        x
    }

    fn get_current_time() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Clock may have gone backwards")
            .as_secs()
            / 3600
    }

    fn random_f32(&mut self, min: f32, max: f32) -> f32 {
        min + (self.next() as f32 / u32::MAX as f32) * (max - min)
    }

    fn random_number<T>(&mut self, min: T, max: T) -> T
    where
        T: std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Add<Output = T>
            + std::ops::Rem<Output = T>
            + std::ops::Div<Output = T>
            + Copy
            + From<u64>,
    {
        min + (max - min * T::from(self.next()) % T::from(u64::MAX)) / T::from(u64::MAX)
    }
}

// Function to create a vector with random values within a range
pub fn create_vector(num_elements: usize, min_value: f32, max_value: f32) -> Vec<f32> {
    let mut vec: Vec<f32> = Vec::with_capacity(num_elements);
    let mut rng: CustomRng = CustomRng::new();
    for _ in 0..num_elements {
        vec.push(rng.random_f32(min_value, max_value));
    }
    vec
}

// Function to craete a matrix
pub fn create_matrix(
    num_rows: usize,
    num_cols: usize,
    min_value: f32,
    max_value: f32,
) -> Vec<Vec<f32>> {
    let mut vec: Vec<Vec<f32>> = Vec::with_capacity(num_rows);
    for _ in 0..num_rows {
        vec.push(create_vector(num_cols, min_value, max_value));
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
/// ```ignore
/// use rec_rsys::testing_tools::compare_execution_times;
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
struct NewCustomRng {
    state: u64,
}

impl NewCustomRng {
    fn new() -> NewCustomRng {
        NewCustomRng {
            state: CustomRng::new().state,
        }
    }

    fn next(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        (x & 0xFFFFFFFF) as u32
    }

    fn random_f32(&mut self, min: f32, max: f32) -> f32 {
        let rand_val = self.next() as f32 / u32::MAX as f32;
        min + (max - min) * rand_val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_current_time() {
        let mut current_time = CustomRng::new();
        println!(
            "number: {:?}",
            current_time.next() as f64 / u64::MAX as f64 * 10e5
        );
    }

    #[test]
    fn random_f() {
        let mut rng = NewCustomRng::new();
        let result = rng.random_f32(-1.0, 1.0);
        println!("Random f32: {}", result);
        println!("number: {:?}", CustomRng::new().random_f32(-1.0, 1.0));
    }
}
