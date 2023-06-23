//! # A collection of tools to compute similarities
//!
use super::statistics::mean;
use super::utils::{argsort, dot, euclidean_norm, squared_diff_sum};
use std::collections::HashSet;

pub enum SimilarityAlgos {
    Euclidean,
    Cosine,
    AdjustedCosine,
    PearsonCorrelation,
    Spearman,
    MSD,
}
/// # Jaccard Similarity
/// Calculated the Jaccard similarity between to sets.
///
/// ## Parameters:
/// * `a`: A set of values
/// * `b`: A set of values
///
/// ## Returns:
/// *
///
/// ## Examples:
///
/// ```
/// ```
///
#[doc = include_str!("../docs/similarity/jaccard_similarity.md")]
pub fn jaccard_similarity(a: &HashSet<&i8>, b: &HashSet<&i8>) -> f32 {
    a.intersection(b).count() as f32 / a.union(b).count() as f32
}

/// # Cosine Similarity
/// Compute the cosine similarity between two vectors.
///
/// ## Parameters:
/// * `vec1`: The first vector.
/// * `vec2`: The second vector.
///
/// ## Returns:
/// The cosine similarity between the two vectors.
///
/// ## Examples:
///
/// ```
/// ```
///
#[doc = include_str!("../docs/similarity/cosine_similarity.md")]
pub fn cosine_similarity(vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
    dot(vec1, vec2) / (euclidean_norm(vec1) * euclidean_norm(vec2))
}

/// # Adjusted Cosine Similarity
/// Function to calculate the Adjusted Cosine Similarity between two items.
///
/// ## Parameters:
/// * `ratings_a`: The ratings of item A by the users.
/// * `ratings_b`: The ratings of item B by the users.
///
/// ## Returns:
/// * The Adjusted Cosine Similarity between items A and B.
///
/// ## Examples:
///
/// ```
/// ```
///
#[doc = include_str!("../docs/similarity/adjusted_cosine_similarity.md")]
pub fn adjusted_cosine_similarity(vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
    dot(vec1, vec2) / (euclidean_norm(vec1) * euclidean_norm(vec2))
}

/// # Compute the euclidean distance.
/// Measure the similarity between two users according to their distance
/// Closer to 0.0 better.
///
/// ## Parameters:
/// * `vec1`: The first vector.
/// * `vec2`: The second vector.
///
/// ## Returns:
/// The euclidean distance between the two vectors.
///
/// ## Examples:
///
/// ```
/// ```
///
#[doc = include_str!("../docs/similarity/euclidean_distance.md")]
pub fn euclidean_distance(vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
    squared_diff_sum(vec1, vec2).sqrt()
}

/// # Exponential Decay Similarity
/// Calculates the exponential decay similarity between two values based on a decay rate.
///
/// ## Parameters:
/// * `value1`: The first value for comparison.
/// * `value2`: The second value for comparison.
/// * `decay_rate`: The decay rate to determine the decay factor.
///
/// ## Examples:
/// ```
/// use rec_rsys::similarity::exponential_decay_similarity;
/// let value1 = 10.0;
/// let value2 = 20.0;
/// let decay_rate = 0.2;
/// let similarity = exponential_decay_similarity(value1, value2, decay_rate);
/// println!("Similarity: {}", similarity);
/// ```
///
/// ## Returns:
/// The calculated similarity value as a `f32` (floating-point number) between 0 and 1.
///
#[doc = include_str!("../docs/similarity/exponential_decay_similarity.md")]
pub fn exponential_decay_similarity(value1: f32, value2: f32, decay_rate: f32) -> f32 {
    (-(value1 - value2).abs() / decay_rate).exp()
}

/// # Pearson correlation
/// Function to calculate the Pearson correlation coefficient between two sets of data.
///
/// ## Parameters:
/// * `x`: The first set of data.
/// * `y`: The second set of data.
///
/// ## Returns:
/// * The Pearson correlation coefficient (r) between x and y.
///
/// ## Examples:
///
/// ```
/// ```
///
#[doc = include_str!("../docs/similarity/pearson_correlation.md")]
pub fn pearson_correlation(vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
    let mean_vec1 = mean(vec1);
    let mean_vec2 = mean(vec2);

    let mut covariance = 0.0;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;

    vec1.iter().zip(vec2.iter()).for_each(|(x, y)| {
        let deviation_x = x - mean_vec1;
        let deviation_y = y - mean_vec2;

        covariance += deviation_x * deviation_y;
        variance_x += deviation_x * deviation_x;
        variance_y += deviation_y * deviation_y;
    });

    covariance / (variance_x.sqrt() * variance_y.sqrt())
}

/// # Pearson Baseline similarity
/// Function to calculate the Pearson Baseline similarity between two vectors, incorporating baseline estimates.
///
/// ## Parameters:
/// * `vec1`: The first vector of ratings.
/// * `vec2`: The second vector of ratings.
/// * `shrinkage`: The shrinkage for the vectors.
///
/// ## Returns:
/// * The Pearson Baseline similarity between the vectors.
///
/// ## Examples:
///
/// ```
/// ```
///
#[doc = include_str!("../docs/similarity/pearson_baseline_similarity.md")]
pub fn pearson_baseline_similarity(
    vec1: &Vec<f32>,
    vec2: &Vec<f32>,
    shrinkage: f32,
) -> f32 {
    // let intersection_count = vec1
    //     .iter()
    //     .zip(vec2.iter())
    //     .filter(|&(a, b)| !a.is_nan() && !b.is_nan())
    //     .count();

    // let adjusted_intersection = intersection_count.checked_sub(1).unwrap_or(0) as f32;
    let adjusted_intersection = vec1.len().checked_sub(1).unwrap_or(0) as f32;

    (adjusted_intersection / (adjusted_intersection + shrinkage))
        * pearson_correlation(vec1, vec2)
}

/// # Mean Squared Difference
/// Function to calculate the Mean Squared Difference (MSD).
///
/// ## Parameters:
/// * `x`: The first vector.
/// * `y`: The second vector.
///
/// ## Returns:
/// * The MSD between the vectors.
///
/// ## Examples:
///
/// ```
/// ```
///
#[doc = include_str!("../docs/similarity/msd.md")]
pub fn msd(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    squared_diff_sum(x, y) / x.len() as f32
}

/// # Mean Squared Difference Similarity
///
/// ## Parameters:
/// * `x`: Vector of user's ratings.
/// * `y`: Vector of user's ratings.
///
/// ## Returns:
/// * The MSD similarity between two vectors.
///
/// ## Examples:
///
/// ```
/// use rec_rsys::similarity::msd_similarity;
/// let user1 = vec![23.0,15.2,11.2222];
/// let user2 = vec![23.0,7.8,87.02];
/// let similarity = msd_similarity(&user1,&user2);
/// println!("Similarity: {}", similarity);
/// ```
///
#[doc = include_str!("../docs/similarity/msd_similarity.md")]
pub fn msd_similarity(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    1.0 / (msd(x, y) + 1.0)
}

/// # Spearman correlation
/// Function to calculate the Spearman correlation coefficient between two vectors.
///
/// ## Parameters:
/// * `x`: The first vector.
/// * `y`: The second vector.
///
/// ## Returns:
/// * The Spearman correlation coefficient between the vectors.
///
/// ## Examples:
///
/// ```
/// ```
///
#[doc = include_str!("../docs/similarity/spearman_correlation.md")]
pub fn spearman_correlation(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    let n = x.len() as f32;
    1.0 - (6.0 * squared_diff_sum(&spearman_rank(x), &spearman_rank(y)))
        / (n * (n.powi(2) - 1.0))
}

fn spearman_rank(x: &Vec<f32>) -> Vec<f32> {
    argsort(&argsort(x))
}

/// # Minkowski distance
/// Function to calculate the Minkowski distance between two vectors.
///
/// ## Parameters:
/// * `x`: The first vector.
/// * `y`: The second vector.
/// * `p`: The order of the Minkowski distance.
///
/// ## Returns:
/// * The Minkowski distance between the vectors.
///
/// ## Examples:
///
/// ```
/// ```
///
#[doc = include_str!("../docs/similarity/minkowski_distance.md")]
pub fn minkowski_distance(x: &[f32], y: &[f32], p: f32) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi).abs().powf(p))
        .sum::<f32>()
        .powf(1.0 / p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard_similarity() {
        let set_a: HashSet<&i8> = [3, 45, 7, 2].iter().collect();
        let set_b: HashSet<&i8> = [2, 54, 13, 15].iter().collect();
        assert_eq!(jaccard_similarity(&set_a, &set_b), 0.14285714285714285);
    }

    #[test]
    fn test_cosine_similarity() {
        assert_eq!(
            cosine_similarity(&vec![3.0, 45.0, 7.0, 2.0], &vec![2.0, 54.0, 13.0, 15.0]),
            0.9722842517123499,
        );
    }

    #[test]
    fn test_euclidean_distance() {
        assert_eq!(
            euclidean_distance(&vec![3.0, 45.0, 7.0, 2.0], &vec![2.0, 54.0, 13.0, 15.0]),
            16.941074346097416,
        );
    }

    #[test]
    fn test_pearson_correlation() {
        assert_eq!(
            pearson_correlation(&vec![3.0, 45.0, 7.0, 2.0], &vec![2.0, 54.0, 13.0, 15.0]),
            0.9675213315629456,
        );
    }

    #[test]
    fn test_exponential_decay_similarity() {
        assert_eq!(
            exponential_decay_similarity(23.5, 44.33333333, 10.0),
            0.12451448,
        );
    }

    #[test]
    fn test_msd() {
        assert_eq!(
            msd(&vec![3.0, 45.0, 7.0, 2.0], &vec![2.0, 54.0, 13.0, 15.0]),
            71.75,
        );
    }

    #[test]
    fn test_msd_similarity() {
        assert_eq!(
            msd_similarity(&vec![3.0, 45.0, 7.0, 2.0], &vec![2.0, 54.0, 13.0, 15.0]),
            0.013745704467353952,
        );
    }

    #[test]
    fn test_pearson_baseline_similarity() {
        assert_eq!(
            pearson_baseline_similarity(
                &vec![3.0, 45.0, 7.0, 2.0],
                &vec![2.0, 54.0, 13.0, 15.0],
                3.2
            ),
            0.46815547,
        );
    }

    #[test]
    fn test_spearman_correlation() {
        assert_eq!(
            spearman_correlation(
                &vec![3.0, 45.0, 7.0, 2.0],
                &vec![2.0, 54.0, 13.0, 15.0]
            ),
            0.39999998,
        );
    }

    #[test]
    fn test_spearman_rank() {
        assert_eq!(
            spearman_rank(&vec![3.0, 45.0, 7.0, 2.0]),
            vec![1.0, 3.0, 2.0, 0.0],
        );
    }

    #[test]
    fn test_minkowski_distance() {
        assert_eq!(
            minkowski_distance(
                &vec![3.0, 45.0, 7.0, 2.0],
                &vec![2.0, 54.0, 13.0, 15.0],
                2.1,
            ),
            16.566132683373674,
        );
    }
}
