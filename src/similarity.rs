//! # A collection of tools to compute similarities
//!
use super::statistics::mean;
use super::utils::{dot, euclidean_norm, rank_vector, squared_diff_sum};
use std::collections::HashSet;

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
/// ## Explanation:
/// The Jaccard similarity measures the similarity between two sets, A and B.
/// It calculates the size of the intersection of the sets divided by the size of the union of the sets.
///
/// ## Formula:
/// $$ J(A, B) = \frac{{|A \cap B|}}{{|A \cup B|}} $$
///
/// ### Where:
/// * `(A, B)`: Users
pub fn jaccard_similarity(a: &HashSet<&i8>, b: &HashSet<&i8>) -> f64 {
    a.intersection(b).count() as f64 / a.union(b).count() as f64
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
/// ## Explanation:
///
///
/// ## Formula:
/// $$ CosSim(x,y)=\frac{\sum_{i}x_iy_i}{\sqrt{\sum_{i}x_i^2}\sqrt{\sum_{i}y_i^2}} $$
///
/// ### Where:
/// * $x, y = \text{}$
pub fn cosine_similarity(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
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
/// ## Explanation:
/// The Adjusted Cosine Similarity is a measure of similarity between two items, A and B,
/// based on user ratings. It takes into account the user's rating patterns and adjusts for variations
/// in their rating scales or biases. The formula calculates the sum of the products of the differences
/// between each user's rating of item A and their average rating, and the differences between each user's
/// rating of item B and their average rating. The numerator represents the adjusted covariance, and the
/// denominators represent the adjusted standard deviations. The result is a value between -1 and 1, where
/// higher values indicate higher similarity.
///
/// ## Formula:
/// $$\text{AdjustedCosine}(A, B) = \frac{{\sum_{u \in U}(R_{u,A} - \overline{R}_u) \cdot (R_{u,B} - \overline{R}_u)}}{{\sqrt{\sum_{u \in U}(R_{u,A} - \overline{R}_u)^2} \cdot \sqrt{\sum_{u \in U}(R_{u,B} - \overline{R}_u)^2}}}$$
///
/// ### Where:
/// * `Ru,ARu,A`​ represents the rating of user uu for item A.
/// * `Ru,BRu,B`​ represents the rating of user uu for item B.
/// * `R‾uRu​ `represents the average rating of user uu across all items.
/// * `UU` represents the set of users who have rated both item A and item B.
pub fn adjusted_cosine_similarity(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
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
/// ## Explanation:
///
/// ## Formula:
/// $$ d(p,q)=\sqrt{\sum_{i = 1}^{n}(q_i - p_i)²} $$
pub fn euclidean_distance(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
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
/// let value1 = 10.0;
/// let value2 = 20.0;
/// let decay_rate = 0.2;
/// let similarity = exponential_decay_similarity(value1, value2, decay_rate);
/// println!("Similarity: {}", similarity);
/// ```
///
/// ## Explanation:
///
///
/// ## Formula:
///
/// ## Returns:
/// The calculated similarity value as a `f64` (floating-point number) between 0 and 1.
pub fn exponential_decay_similarity(value1: f64, value2: f64, decay_rate: f64) -> f64 {
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
/// ## Explanantion:
/// The Pearson correlation coefficient (r) is calculated as the covariance of the two sets divided by the product of their standard deviations.
///
/// ## Formula:
/// $$ r = \frac{{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}}{{\sqrt{{\sum_{i=1}^{n} (x_i - \bar{x})^2}} \cdot \sqrt{{\sum_{i=1}^{n} (y_i - \bar{y})^2}}}} $$
pub fn pearson_correlation(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
    let mean_vec1: f64 = mean(vec1);
    let mean_vec2: f64 = mean(vec2);

    let mut covariance: f64 = 0.0;
    let mut variance_x: f64 = 0.0;
    let mut variance_y: f64 = 0.0;

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
/// * `bx`: The baseline estimate for the first vector.
/// * `by`: The baseline estimate for the second vector.
///
/// ## Returns:
/// * The Pearson Baseline similarity between the vectors.
///
/// ## Examples:
///
/// ```
/// ```
///
/// ## Explanation:
/// The Pearson Baseline similarity calculates the correlation between two vectors,
/// taking into account the baseline estimates for each element. It subtracts the baseline estimates
/// from the actual ratings to obtain the residuals. The similarity is then calculated as the
/// cosine of the angle between the residual vectors. The baseline estimates represent the average
/// rating for each user/item, which helps adjust for the overall user/item biases.
///
/// ## Formula:
/// $$\text{similarity} = \frac{{\sum_{i=1}^{n}((r_{xi} - b_{xi}) \cdot (r_{yi} - b_{yi}))}}{{\sqrt{{\sum_{i=1}^{n}(r_{xi} - b_{xi})^2}} \cdot \sqrt{{\sum_{i=1}^{n}(r_{yi} - b_{yi})^2}}}}$$
pub fn pearson_baseline_similarity(vec1: &[f64], vec2: &[f64], bx: f64, by: f64) -> f64 {
    let mut numerator: f64 = 0.0;
    let mut denominator_x: f64 = 0.0;
    let mut denominator_y: f64 = 0.0;

    vec1.iter().zip(vec2.iter()).for_each(|(x, y)| {
        let x_bx: f64 = x - bx;
        let y_by: f64 = y - by;
        numerator += x_bx * y_by;
        denominator_x += x_bx.powi(2);
        denominator_y += y_by.powi(2);
    });

    numerator / (denominator_x.sqrt() * denominator_y.sqrt())
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
/// ## Explanation:
/// The MSD calculates the average squared difference between
/// the corresponding elements of two vectors. It measures the dissimilarity between the vectors,
/// with a lower value indicating a higher similarity.
///
/// ## Formula:
/// $$ \text{MSD(x,y)} = \frac{{\sum_{i=1}^{n}(x_i - y_i)^2}}{{n}} $$
pub fn msd(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
    squared_diff_sum(x, y) / x.len() as f64
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
/// // struct User {items: Vec<f64>}
/// let user1 = User::from(vec![23.0,15.2,11.2222]);
/// let user2 = User::from(vec![23.0,7.8,87.02]);
/// let similarity = msd_similarity(user1.items, user2.items);
/// println!("Similarity: {}", similarity);
/// ```
///
/// ## Explanation:
///
///
/// ## Formula:
/// $$ \text{MSD(x,y)} = \frac{{\sum_{i \in I_{xy}}(R_{X_i} - R_{Y_i})^2}}{{I_{xy}}} $$
///
/// ### Where:
/// * $X, Y = \text{User, User}$
/// * $I = \text{Item}$
/// * $R = \text{Rating}$
/// * $I_{xy} = \text{Items in common}$
pub fn msd_similarity(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
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
/// ## Explanation:
/// The Spearman correlation coefficient measures the strength and direction of the monotonic relationship between two variables. It is calculated based on the ranks of the values in the vectors, rather than their actual values. The correlation coefficient ranges from -1 to 1, where a value of 1 indicates a perfect increasing monotonic relationship, -1 indicates a perfect decreasing monotonic relationship, and 0 indicates no monotonic relationship.
///
/// ## Formula:
/// $$ \text{correlation} = 1 - \frac{{6 \sum_{i=1}^{n}(d_i)^2}}{{n(n^2 - 1)}} $$
pub fn spearman_correlation(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
    let n: f64 = x.len() as f64;
    1.0 - (6.0 * squared_diff_sum(&rank_vector(x), &rank_vector(y))) / (n * (n.powi(2) - 1.0))
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
/// ## Explanation:
/// The Minkowski distance is a generalization of other distance metrics such as the Euclidean distance and the Manhattan distance. It measures the distance or dissimilarity between two vectors based on their element-wise differences raised to the power of p. The value of p determines the order of the Minkowski distance, where p = 1 corresponds to the Manhattan distance and p = 2 corresponds to the Euclidean distance.
///
/// ## Formula:
/// $$\text{distance} = \left(\sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}}$$
pub fn minkowski_distance(x: &[f64], y: &[f64], p: f64) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi).abs().powf(p))
        .sum::<f64>()
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
            0.12451447148562779,
        );
    }

    #[test]
    fn test_msd() {
        assert_eq!(
            msd(&vec![3.0, 45.0, 7.0, 2.0], &vec![2.0, 54.0, 13.0, 15.0]),
            0.9675213315629456,
        );
    }

    #[test]
    fn test_msd_similarity() {
        assert_eq!(
            msd_similarity(&vec![3.0, 45.0, 7.0, 2.0], &vec![2.0, 54.0, 13.0, 15.0]),
            0.9675213315629456,
        );
    }

    #[test]
    fn test_pearson_baseline_similarity() {
        assert_eq!(
            pearson_baseline_similarity(&[3.0, 45.0, 7.0, 2.0], &[2.0, 54.0, 13.0, 15.0], 2.3, 3.1,),
            0.9675213315629456,
        );
    }

    #[test]
    fn test_spearman_correlation() {
        assert_eq!(
            spearman_correlation(&vec![3.0, 45.0, 7.0, 2.0], &vec![2.0, 54.0, 13.0, 15.0]),
            0.9675213315629456,
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
            0.9675213315629456,
        );
    }
}
