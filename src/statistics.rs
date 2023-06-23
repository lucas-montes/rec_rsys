//! # A collection of statistical functions
//!
use super::utils::local_sort;

/// # Mean
/// Function to calculate the mean (average) of a set of data.
///
/// ## Parameters:
/// * `data`: The set of data.
///
/// ## Returns:
/// * The mean value of the data.
///
#[doc = include_str!("../docs/statistics/mean.md")]
pub fn mean(data: &Vec<f32>) -> f32 {
    data.iter().sum::<f32>() / data.len() as f32
}

/// # Quartiles
/// Function to calculate the quartiles of a set of data.
///
/// ## Parameters:
/// * `data`: The set of data.
///
/// ## Returns:
/// * A tuple (Q1, Q3) containing the first and third quartiles of the data.
///
#[doc = include_str!("../docs/statistics/quartiles.md")]
pub fn quartiles(data: &mut Vec<f32>) -> (f32, f32) {
    local_sort(data);
    let q1: f32 = percentile_of_sorted(data, 25_f32);
    let q3: f32 = percentile_of_sorted(data, 75_f32);
    (q1, q3)
}

/// Function to calculate the percentile of sorted samples.
///
/// ## Parameters:
/// * `sorted_samples`: The sorted set of samples.
/// * `pct`: The desired percentile (0.0 to 100.0).
///
/// ## Returns:
/// * The value at the specified percentile.
///
#[doc = include_str!("../docs/statistics/percentile_of_sorted.md")]
fn percentile_of_sorted(sorted_samples: &Vec<f32>, pct: f32) -> f32 {
    let sorted_len = sorted_samples.len();
    if sorted_len == 1 {
        return sorted_samples[0];
    }
    if pct == 100_f32 {
        return sorted_samples[sorted_len - 1];
    }
    let rank = (pct / 100_f32) * (sorted_len - 1) as f32;
    let lrank = rank.floor();
    let n = lrank as usize;
    let lo = sorted_samples[n];
    lo + (sorted_samples[n + 1] - lo) * (rank - lrank)
}

/// # Median
/// Function to calculate the median of a set of data.
///
/// ## Parameters:
/// * `data`: The set of data.
///
/// ## Returns:
/// * The median value of the data.
///
#[doc = include_str!("../docs/statistics/median.md")]
pub fn median(data: &Vec<f32>) -> f32 {
    percentile_of_sorted(data, 50_f32)
}

/// # Covariance
/// Function to calculate the covariance between two sets of data.
///
/// ## Parameters:
/// * `x`: The first set of data.
/// * `y`: The second set of data.
///
/// ## Returns:
/// * The covariance between x and y.
///
#[doc = include_str!("../docs/statistics/median_abs_dev.md")]
pub fn covariance(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    let mean_x = mean(x);
    let mean_y = mean(y);

    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f32>()
        / (x.len() - 1) as f32
}
// pub fn covariance(vec1: &Vec<f32>, vec2: &Vec<f32>, len: Option<f32>) -> f32 {
//     let mean_vec1: f32 = mean(vec1);
//     let mean_vec2: f32 = mean(vec2);
//     vec1.iter()
//         .zip(vec2.iter())
//         .map(|(x, y)| (x - mean_vec1) * (y - mean_vec2))
//         .sum::<f32>()
//         / match len {
//             Some(value) => value,
//             None => vec1.len() as f32,
//         }
// }
/// # Variance
/// Variance is a statistical measure of how spread out a set of data points is. It provides a measure of the variability or dispersion of the data. In the context of recommender systems, variance can be used to quantify the spread or diversity of ratings or preferences given by users.
///
/// ## Parameters:
/// * `data`: A slice of f32 values representing the data points.
///
/// ## Returns:
/// * The variance of the data as an f32 value.
///
/// ## Examples:
/// ```
/// use rec_rsys::statistics::variance;
/// assert_eq!(variance(&vec![1.0, 2.0, 3.0, 4.0, 5.0]), 2.0);
/// ```
///
#[doc = include_str!("../docs/statistics/variance.md")]
pub fn variance(data: &Vec<f32>) -> f32 {
    let mean = mean(data);
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32
}

/// # Standard Deviation
/// Function to calculate the standard deviation of a set of data.
///
/// ## Parameters:
/// * `data`: The set of data.
///
/// ## Returns:
/// * The standard deviation of the data.
///
#[doc = include_str!("../docs/statistics/standard_deviation.md")]
pub fn standard_deviation(data: &Vec<f32>) -> f32 {
    let mean = mean(data);
    let sum_squared_deviations = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>();
    (sum_squared_deviations / data.len() as f32).sqrt()
}

/// TODO
/// #[doc = include_str!("../docs/statistics/median_abs_dev.md")]
pub fn standard_deviation_pct(data: &Vec<f32>) -> f32 {
    (standard_deviation(data) / mean(data)) * 100_f32
}

/// TODO
#[doc = include_str!("../docs/statistics/median_abs_dev.md")]
pub fn median_abs_dev(data: &Vec<f32>) -> f32 {
    let med = median(data);
    let abs_devs: Vec<f32> = data.iter().map(|&v| (med - v).abs()).collect();
    // This constant is derived by smarter statistics brains than me, but it is
    // consistent with how R and other packages treat the MAD.
    median(&abs_devs) * 1.4826
}

/// TODO
/// #[doc = include_str!("../docs/statistics/median_abs_dev.md")]
pub fn median_abs_dev_pct(data: &Vec<f32>) -> f32 {
    (median_abs_dev(data) / median(data)) * 100_f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variance() {
        assert_eq!(variance(&vec![1.0, 2.0, 3.0, 4.0, 5.0]), 2.0);
    }

    #[test]
    fn test_covariance() {
        assert_eq!(
            covariance(&vec![3.0, 45.0, 7.0, 2.0], &vec![2.0, 54.0, 13.0, 15.0]),
            453.3333333333333,
        );
    }

    #[test]
    fn test_standard_deviation() {
        assert_eq!(
            standard_deviation(&vec![3.0, 45.0, 7.0, 2.0]),
            17.851820635442202,
        );
    }

    // #[test]
    // fn test_quartiles() {
    //     assert_eq!(quartiles(&mut [3.0, 45.0, 7.0, 2.0]), (2.75, 16.5),);
    // }

    // #[test]
    // fn test_median() {
    //     assert_eq!(median(&[3.0, 45.0, 7.0, 2.0]), 17.851820635442202,);
    // }

    // #[test]
    // fn standard_deviation_pct() {
    //     assert_eq!(
    //         standard_deviation_pct(&vec![3.0,45.0,7.0,2.0], 0);
    //     )
    // }
    // #[test]
    // fn median_abs_dev() {
    //     assert_eq!(
    //         median_abs_dev(&vec![3.0,45.0,7.0,2.0], 0);
    //     )
    // }

    // #[test]
    // fn median_abs_dev_pct() {
    //     assert_eq!(
    //         median_abs_dev_pct(&vec![3.0,45.0,7.0,2.0], 0);
    //     )
    // }
}
