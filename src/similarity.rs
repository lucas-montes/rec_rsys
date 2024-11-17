//! # A collection of tools to compute similarities
//!
use ndarray::{azip, Array1, ArrayView1, Zip};
use num_traits::float::FloatCore;
use num_traits::{Float, FromPrimitive, Pow};

use super::statistics::mean;
use super::utils::{argsort, euclidean_norm, squared_diff_sum};
use std::collections::HashSet;
use std::ops::{Mul, Sub};

#[derive(Default)]
pub enum SimilarityAlgorithm {
    EuclideanDistance,
    #[default]
    CosineSimilarity,
    AdjustedCosineSimilarity,
    PearsonCorrelation,
    SpearmanCorrelation,
    MSDSimilarity,
}

impl SimilarityAlgorithm {
    pub fn get_function<F: Float + FromPrimitive + std::iter::Sum + 'static>(
        &self,
    ) -> Box<dyn Fn(&ArrayView1<F>, &ArrayView1<F>) -> F> {
        match self {
            Self::EuclideanDistance => Box::new(euclidean_distance),
            Self::CosineSimilarity => Box::new(cosine_similarity),
            Self::AdjustedCosineSimilarity => Box::new(adjusted_cosine_similarity),
            Self::PearsonCorrelation => Box::new(pearson_correlation_uncentered),
            Self::SpearmanCorrelation => Box::new(spearman_correlation),
            Self::MSDSimilarity => Box::new(msd_similarity),
        }
    }
}
/// # Jaccard Similarity
/// Calculated the Jaccard similarity between to sets.
///
/// ## Parameters:
/// * `a`: F set of values
/// * `b`: F set of values
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
pub fn cosine_similarity<F: Float + FromPrimitive + std::iter::Sum + 'static>(
    u: &ArrayView1<F>,
    v: &ArrayView1<F>,
) -> F {
    u.dot(v) / (euclidean_norm(u) * euclidean_norm(v))
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
pub fn adjusted_cosine_similarity<F: Float + FromPrimitive + std::iter::Sum + 'static>(
    u: &ArrayView1<F>,
    v: &ArrayView1<F>,
) -> F {
    //TODO
    u.dot(v) / (euclidean_norm(u) * euclidean_norm(v))
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
pub fn euclidean_distance<F: Float + FromPrimitive + std::iter::Sum>(
    u: &ArrayView1<F>,
    v: &ArrayView1<F>,
) -> F {
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
pub fn exponential_decay_similarity<F: Float + FromPrimitive>(
    value1: F,
    value2: F,
    decay_rate: F,
) -> F {
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
pub fn pearson_correlation_uncentered<F: Float + FromPrimitive>(
    u: &ArrayView1<F>,
    v: &ArrayView1<F>,
) -> F {
    let mean_u = u.mean().unwrap();
    let mean_v = v.mean().unwrap();

    let mut covariance = F::zero();
    let mut variance_x = F::zero();
    let mut variance_y = F::zero();

    Zip::from(u).and(v).for_each(|&x, &y| {
        let deviation_x = x - mean_u;
        let deviation_y = y - mean_v;

        covariance = deviation_x.mul_add(deviation_y, covariance);
        variance_x = deviation_x.mul_add(deviation_x, variance_x);
        variance_y = deviation_y.mul_add(deviation_y, variance_y);
    });

    covariance / (variance_x * variance_y).sqrt()
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
pub fn pearson_baseline_similarity<F: Float + FromPrimitive>(
    u: &ArrayView1<F>,
    v: &ArrayView1<F>,
    shrinkage: F,
) -> F {
    let adjusted_intersection = F::from_usize(u.len().saturating_sub(1)).unwrap();
    (adjusted_intersection / (adjusted_intersection + shrinkage))
        * pearson_correlation_uncentered(u, v)
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
pub fn msd<F: Float + FromPrimitive + std::iter::Sum>(
    u: &ArrayView1<F>,
    v: &ArrayView1<F>,
) -> F {
    squared_diff_sum(u, v) / F::from_usize(u.len()).unwrap()
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
pub fn msd_similarity<F: Float + FromPrimitive + std::iter::Sum>(
    u: &ArrayView1<F>,
    v: &ArrayView1<F>,
) -> F {
    let i = F::from(1.0).unwrap();
    i / (msd(u, v) + i)
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
pub fn spearman_correlation<F: Float + FromPrimitive + std::iter::Sum>(
    u: &ArrayView1<F>,
    v: &ArrayView1<F>,
) -> F {
    let n = F::from(u.len()).unwrap();

    let one = F::one();
    let six = F::from(6.0).unwrap();

    one - (six * squared_diff_sum(&spearman_rank(u).view(), &spearman_rank(v).view()))
        / (n * (n.powi(2) - one))
}

fn spearman_rank<F: Float + FromPrimitive>(x: &ArrayView1<F>) -> Array1<F> {
    argsort(&argsort(x).view())
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
pub fn minkowski_distance<F: Float + Clone + FromPrimitive + std::iter::Sum>(
    u: &ArrayView1<F>,
    v: &ArrayView1<F>,
    p: F,
) -> F {
    u.iter()
        .zip(v.iter())
        .map(|(&ui, &vi)| (ui - vi).abs().powf(p))
        .sum::<F>()
        .powf(F::one() / p)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

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
            cosine_similarity(
                &array![3.0, 45.0, 7.0, 2.0].view(),
                &array![2.0, 54.0, 13.0, 15.0].view()
            ),
            0.9722842517123499,
        );
    }

    #[test]
    fn test_euclidean_distance() {
        assert_eq!(
            euclidean_distance(
                &array![3.0, 45.0, 7.0, 2.0].view(),
                &array![2.0, 54.0, 13.0, 15.0].view()
            ),
            16.941074346097416,
        );
    }

    #[test]
    fn test_pearson_correlation_uncentered() {
        assert_eq!(
            pearson_correlation_uncentered(
                &array![3.0, 45.0, 7.0, 2.0].view(),
                &array![2.0, 54.0, 13.0, 15.0].view()
            ),
            0.9_675_213_315_629_456,
        );
    }

    #[test]
    fn test_exponential_decay_similarity() {
        assert_eq!(
            exponential_decay_similarity(23.5, 44.333_332, 10.0),
            0.12451448804605365,
        );
    }

    #[test]
    fn test_msd() {
        assert_eq!(
            msd(
                &array![3.0, 45.0, 7.0, 2.0].view(),
                &array![2.0, 54.0, 13.0, 15.0].view()
            ),
            71.75,
        );
    }

    #[test]
    fn test_msd_similarity() {
        assert_eq!(
            msd_similarity(
                &array![3.0, 45.0, 7.0, 2.0].view(),
                &array![2.0, 54.0, 13.0, 15.0].view()
            ),
            0.013745704467353952,
        );
    }

    #[test]
    fn test_pearson_baseline_similarity() {
        assert_eq!(
            pearson_baseline_similarity(
                &array![3.0, 45.0, 7.0, 2.0].view(),
                &array![2.0, 54.0, 13.0, 15.0].view(),
                3.2
            ),
            0.46815548301432847,
        );
    }

    #[test]
    fn test_spearman_correlation() {
        assert_eq!(
            spearman_correlation(
                &array![3.0, 45.0, 7.0, 2.0].view(),
                &array![2.0, 54.0, 13.0, 15.0].view()
            ),
            0.4,
        );
    }

    #[test]
    fn test_spearman_rank() {
        assert_eq!(
            spearman_rank(&array![3.0, 45.0, 7.0, 2.0].view()),
            array![1.0, 3.0, 2.0, 0.0],
        );
    }

    #[test]
    fn test_minkowski_distance() {
        assert_eq!(
            minkowski_distance(
                &array![3.0, 45.0, 7.0, 2.0].view(),
                &array![2.0, 54.0, 13.0, 15.0].view(),
                2.1,
            ),
            16.566132683373674,
        );
    }
}
