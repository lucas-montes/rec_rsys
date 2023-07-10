//! KNN
use crate::models::Item;
use crate::similarity::{
    adjusted_cosine_similarity, cosine_similarity, euclidean_distance, msd_similarity,
    pearson_baseline_similarity, pearson_correlation, spearman_correlation,
    SimilarityAlgos,
};
use crate::utils::{sort_and_trucate, sort_with_direction};

type ParamDistanceFunction = dyn Fn(&[f32], &[f32]) -> f32;

/// # KNN
/// K-nearest neighbors (KNN) is a machine learning algorithm used for classification and regression. It predicts the class or value of a new data point based on the majority class or average value of its k nearest neighbors in the feature space.
///
/// ## Parameters:
/// * new_item: The new item for which the algorithm will predict a result.
/// * references: The reference items used for comparison and prediction.
/// * k: The number of nearest neighbors to consider in the prediction.
///
/// ## Returns:
/// * A vector of items representing the predicted results.
///
/// ## Examples:
/// ```
/// use rec_rsys::{algorithms::knn::KNN, models::Item, similarity::SimilarityAlgos};
/// let new_item = Item { id: 1, values: vec![1.0, 2.0, 3.0], result: f32::NAN };
/// let references = vec![ Item { id: 2, values: vec![4.0, 5.0, 6.0], result: f32::NAN }, Item { id: 3, values: vec![7.0, 8.0, 9.0], result: f32::NAN }, Item { id: 4, values: vec![10.0, 11.0, 12.0], result: f32::NAN } ];
/// let k = 2;
/// let knn = KNN::new(new_item, references, k);
/// let result = knn.result(SimilarityAlgos::Cosine);
/// println!("{:?}", result);
/// ```
///
#[doc = include_str!("../../docs/algorithms/knn.md")]
pub struct KNN {
    new_item: Item,
    references: Vec<Item>,
    k: u8,
}

impl KNN {
    pub fn new(new_item: Item, references: Vec<Item>, k: u8) -> Self {
        KNN {
            new_item,
            references,
            k,
        }
    }
    /// Performs the KNN prediction based on the specified similarity algorithm.
    ///
    /// ## Parameters:
    /// * `algorithm`: The similarity algorithm to use for calculating distances.
    ///
    /// ## Returns:
    /// * A vector of items representing the predicted results.
    pub fn result(&self, algorithm: SimilarityAlgos) -> Vec<Item> {
        let (formula, reverse) = KNN::get_formula(algorithm);
        let mut best_matches: Vec<Item> = Vec::new();
        self.references.iter().for_each(|item| {
            let cloned_item = item.clone();
            best_matches
                .push(cloned_item.result(formula(&self.new_item.values, &item.values)))
        });

        sort_and_trucate(best_matches, reverse, self.k)
    }

    /// Retrieves the distance formula and reverse flag for the specified similarity algorithm.
    ///
    /// ## Parameters:
    /// * `algorithm`: The similarity algorithm.
    ///
    /// ## Returns:
    /// * A tuple containing the distance formula function and a flag indicating if the results should be reversed.
    fn get_formula(algorithm: SimilarityAlgos) -> (&'static ParamDistanceFunction, bool) {
        match algorithm {
            SimilarityAlgos::Cosine => (&cosine_similarity, true),
            SimilarityAlgos::AdjustedCosine => (&adjusted_cosine_similarity, true),
            SimilarityAlgos::Euclidean => (&euclidean_distance, false),
            SimilarityAlgos::PearsonCorrelation => (&pearson_correlation, true),
            SimilarityAlgos::Spearman => (&spearman_correlation, true),
            SimilarityAlgos::MSD => (&msd_similarity, true),
        }
    }
}
