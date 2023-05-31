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
/// ## Formula:
/// $$ \text{mean} = \frac{{\sum_{i=1}^{n} x_i}}{{n}} $$
///
/// ## Explanation:
/// The mean is calculated by summing up all the data points in the set
/// and dividing the sum by the total number of data points. It represents the central
/// tendency or average value of the data set.
pub fn mean(data: &Vec<f64>) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
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
/// Formula for first quartile (Q1): Q1 = \text{median}(\text{lower half of the sorted data})
/// Formula for third quartile (Q3): Q3 = \text{median}(\text{upper half of the sorted data})
///
/// ## Explanation:
/// Quartiles divide a data set into four equal parts, each containing approximately
/// 25% of the data. The first quartile (Q1) is the median of the lower half of the sorted data,
/// and the third quartile (Q3) is the median of the upper half of the sorted data.
pub fn quartiles(data: &mut Vec<f64>) -> (f64, f64) {
    local_sort(data);
    let q1: f64 = percentile_of_sorted(data, 25_f64);
    let q3: f64 = percentile_of_sorted(data, 75_f64);
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
/// ## Formula:
/// $$ \text{percentile} = L + (U - L) \cdot (P - \text{floor}(P)) $$
///
/// ## Explanation:
/// The percentile calculation is based on linear interpolation. Given a sorted
/// set of samples, the percentile formula calculates the value at the desired percentile (P)
/// by interpolating between the lower value (L) and the upper value (U) closest to the rank
/// corresponding to the desired percentile. The interpolation factor is determined by the
/// fractional part of the rank.
fn percentile_of_sorted(sorted_samples: &Vec<f64>, pct: f64) -> f64 {
    let sorted_len = sorted_samples.len();
    if sorted_len == 1 {
        return sorted_samples[0];
    }
    if pct == 100_f64 {
        return sorted_samples[sorted_len - 1];
    }
    let rank: f64 = (pct / 100_f64) * (sorted_len - 1) as f64;
    let lrank: f64 = rank.floor();
    let n: usize = lrank as usize;
    let lo: f64 = sorted_samples[n];
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
/// ## Explanation:
/// The median is the middle value of a set of data when it is sorted in ascending order.
/// If the number of data points is odd, the median is the middle value. If the number of data points
/// is even, the median is the average of the two middle values.
pub fn median(data: &Vec<f64>) -> f64 {
    percentile_of_sorted(data, 50_f64)
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
/// ## Formula:
/// $$ Cov(x, y) = \frac{{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}}{{n}} $$
///
/// ## Explanation:
/// The covariance measures the direction and magnitude of the linear relationship
/// between two sets of data, x and y. It calculates the sum of the products of the deviations
/// of each data point from their respective means, divided by the number of data points.
pub fn covariance(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
    let mean_x: f64 = mean(x);
    let mean_y: f64 = mean(y);

    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>()
        / (x.len() - 1) as f64
}
// pub fn covariance(vec1: &Vec<f64>, vec2: &Vec<f64>, len: Option<f64>) -> f64 {
//     let mean_vec1: f64 = mean(vec1);
//     let mean_vec2: f64 = mean(vec2);
//     vec1.iter()
//         .zip(vec2.iter())
//         .map(|(x, y)| (x - mean_vec1) * (y - mean_vec2))
//         .sum::<f64>()
//         / match len {
//             Some(value) => value,
//             None => vec1.len() as f64,
//         }
// }
/// # Variance
/// Variance is a statistical measure of how spread out a set of data points is. It provides a measure of the variability or dispersion of the data. In the context of recommender systems, variance can be used to quantify the spread or diversity of ratings or preferences given by users.
///
/// ## Parameters:
/// * `data`: A slice of f64 values representing the data points.
///
/// ## Returns:
/// * The variance of the data as an f64 value.
///
/// ## Examples:
/// ```
/// assert_eq!(variance(&vec![1.0, 2.0, 3.0, 4.0, 5.0]), 2.0);
/// ```
///
/// ## Explanation:
/// The variance of a set of data points is calculated by finding the average of the squared differences between each data point and the mean of the data. It measures how much the data points deviate from the mean.
///
/// ## Formula:
/// $$ \text{{Variance}} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2 $$
///
/// ### Where:
/// * \(n\) is the number of data points.
/// * \(x_i\) is the \(i\)th data point.
/// * \(\mu\) is the mean of the data points.
pub fn variance(data: &Vec<f64>) -> f64 {
    let mean = mean(data);
    data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
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
/// ## Formula:
/// \text{std\_deviation} = \sqrt{\frac{{\sum_{i=1}^{n} (x_i - \bar{x})^2}}{{n}}}
///
/// ## Explanation:
/// The standard deviation measures the amount of variation or dispersion
/// in a set of data. It is calculated by first finding the deviations of each data point
/// from the mean, squaring the deviations, summing them up, dividing by the number of data
/// points, and then taking the square root. The standard deviation indicates how spread out
/// the data points are around the mean.
pub fn standard_deviation(data: &Vec<f64>) -> f64 {
    let mean: f64 = mean(data);
    let sum_squared_deviations = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();
    (sum_squared_deviations / data.len() as f64).sqrt()
}

pub fn standard_deviation_pct(data: &Vec<f64>) -> f64 {
    (standard_deviation(data) / mean(data)) * 100_f64
}

pub fn median_abs_dev(data: &Vec<f64>) -> f64 {
    let med: f64 = median(data);
    let abs_devs: Vec<f64> = data.iter().map(|&v| (med - v).abs()).collect();
    // This constant is derived by smarter statistics brains than me, but it is
    // consistent with how R and other packages treat the MAD.
    median(&abs_devs) * 1.4826
}

pub fn median_abs_dev_pct(data: &Vec<f64>) -> f64 {
    (median_abs_dev(data) / median(data)) * 100_f64
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
