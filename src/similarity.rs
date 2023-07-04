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
/// * `u`: The first vector.
/// * `v`: The second vector.
///
/// ## Returns:
/// The cosine similarity between the two vectors.
///
/// ## Examples:
///
/// ```
/// ```
///
#[doc = include_str!("../docs/norms/cosine_similarity.md")]
pub fn cosine_similarity(u: &[f32], v: &[f32]) -> f32 {
    dot(u, v) / (euclidean_norm(u) * euclidean_norm(v))
}

/// # Adjusted Cosine Similarity
/// Function to calculate the Adjusted Cosine Similarity between two items.
///
/// ## Parameters:
/// * `u`: The ratings of item A by the users.
/// * `v`: The ratings of item B by the users.
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
pub fn adjusted_cosine_similarity(u: &[f32], v: &[f32]) -> f32 {
    dot(u, v) / (euclidean_norm(u) * euclidean_norm(v))
}

/// # Compute the euclidean distance.
/// Measure the similarity between two vectors according to their distance
///
/// ## Parameters:
/// * `u`: The first vector.
/// * `v`: The second vector.
///
/// ## Returns:
/// The euclidean distance between the two vectors.
///
/// ## Examples:
///
/// ```
/// ```
///
#[doc = include_str!("../docs/norms/euclidean_distance.md")]
pub fn euclidean_distance(u: &[f32], v: &[f32]) -> f32 {
    squared_diff_sum(u, v).sqrt()
}

/// # Exponential Decay Similarity
/// Calculates the exponential decay similarity between two values based on a decay rate.
///
/// ## Parameters:
/// * `value1`: The first value for comparison.
/// * `value2`: The second value for comparison.
/// * `decay_rate`: The decay rate to determine the decay factor.
///
/// ## Returns:
/// The calculated similarity value as a `f32` (floating-point number) between 0 and 1.
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
#[doc = include_str!("../docs/similarity/exponential_decay_similarity.md")]
pub fn exponential_decay_similarity(value1: f32, value2: f32, decay_rate: f32) -> f32 {
    (-(value1 - value2).abs() / decay_rate).exp()
}

/// # Pearson correlation
/// Function to calculate the Pearson correlation coefficient between two sets of data.
///
/// ## Parameters:
/// * `u`: The first set of data.
/// * `v`: The second set of data.
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
pub fn pearson_correlation(u: &[f32], v: &[f32]) -> f32 {
    let mean_u = mean(u);
    let mean_v = mean(v);

    let mut covariance = 0.0;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;

    u.iter().zip(v.iter()).for_each(|(x, y)| {
        let deviation_x = x - mean_u;
        let deviation_y = y - mean_v;

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
/// * `u`: The first vector of ratings.
/// * `v`: The second vector of ratings.
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
pub fn pearson_baseline_similarity(u: &[f32], v: &[f32], shrinkage: f32) -> f32 {
    let adjusted_intersection = u.len().saturating_sub(1) as f32;
    (adjusted_intersection / (adjusted_intersection + shrinkage))
        * pearson_correlation(u, v)
}

/// # Mean Squared Difference
/// Function to calculate the Mean Squared Difference (MSD).
///
/// ## Parameters:
/// * `u`: The first vector.
/// * `v`: The second vector.
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
pub fn msd(u: &[f32], v: &[f32]) -> f32 {
    squared_diff_sum(u, v) / u.len() as f32
}

/// # Mean Squared Difference Similarity
///
/// ## Parameters:
/// * `u`: Vector of user's ratings.
/// * `v`: Vector of user's ratings.
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
pub fn msd_similarity(u: &[f32], v: &[f32]) -> f32 {
    1.0 / (msd(u, v) + 1.0)
}

/// # Spearman correlation
/// Function to calculate the Spearman correlation coefficient between two vectors.
///
/// ## Parameters:
/// * `u`: The first vector.
/// * `v`: The second vector.
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
pub fn spearman_correlation(u: &[f32], v: &[f32]) -> f32 {
    let n = u.len() as f32;
    1.0 - (6.0 * squared_diff_sum(&spearman_rank(u), &spearman_rank(v)))
        / (n * (n.powi(2) - 1.0))
}

fn spearman_rank(x: &[f32]) -> Vec<f32> {
    argsort(&argsort(x))
}

/// # Minkowski distance
/// Function to calculate the Minkowski distance between two vectors.
///
/// ## Parameters:
/// * `u`: The first vector.
/// * `v`: The second vector.
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
pub fn minkowski_distance(u: &[f32], v: &[f32], p: f32) -> f32 {
    u.iter()
        .zip(v.iter())
        .map(|(&ui, &vi)| (ui - vi).abs().powf(p))
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
        assert_eq!(jaccard_similarity(&set_a, &set_b), 0.142_857_15);
    }

    #[test]
    fn test_cosine_similarity() {
        assert_eq!(
            cosine_similarity(&[3.0, 45.0, 7.0, 2.0], &[2.0, 54.0, 13.0, 15.0]),
            0.972_284_26,
        );
    }

    #[test]
    fn test_euclidean_distance() {
        assert_eq!(
            euclidean_distance(&[3.0, 45.0, 7.0, 2.0], &[2.0, 54.0, 13.0, 15.0]),
            16.941_074,
        );
    }

    #[test]
    fn test_pearson_correlation() {
        assert_eq!(
            pearson_correlation(&[3.0, 45.0, 7.0, 2.0], &[2.0, 54.0, 13.0, 15.0]),
            0.967_521_3,
        );
    }

    #[test]
    fn test_exponential_decay_similarity() {
        assert_eq!(
            exponential_decay_similarity(23.5, 44.333_332, 10.0),
            0.12451448,
        );
    }

    #[test]
    fn test_msd() {
        assert_eq!(msd(&[3.0, 45.0, 7.0, 2.0], &[2.0, 54.0, 13.0, 15.0]), 71.75,);
    }

    #[test]
    fn test_msd_similarity() {
        assert_eq!(
            msd_similarity(&[3.0, 45.0, 7.0, 2.0], &[2.0, 54.0, 13.0, 15.0]),
            0.013_745_705,
        );
    }

    #[test]
    fn test_pearson_baseline_similarity() {
        assert_eq!(
            pearson_baseline_similarity(
                &[3.0, 45.0, 7.0, 2.0],
                &[2.0, 54.0, 13.0, 15.0],
                3.2
            ),
            0.46815547,
        );
    }

    #[test]
    fn test_spearman_correlation() {
        assert_eq!(
            spearman_correlation(&[3.0, 45.0, 7.0, 2.0], &[2.0, 54.0, 13.0, 15.0]),
            0.39999998,
        );
    }

    #[test]
    fn test_spearman_rank() {
        assert_eq!(
            spearman_rank(&[3.0, 45.0, 7.0, 2.0]),
            vec![1.0, 3.0, 2.0, 0.0],
        );
    }

    #[test]
    fn test_minkowski_distance() {
        assert_eq!(
            minkowski_distance(&[3.0, 45.0, 7.0, 2.0], &[2.0, 54.0, 13.0, 15.0], 2.1,),
            16.566_133,
        );
    }
}
