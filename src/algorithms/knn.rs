//! KNN
use crate::models::Item;
use crate::similarity::{
    adjusted_cosine_similarity, cosine_similarity, euclidean_distance, msd_similarity,
    pearson_baseline_similarity, pearson_correlation_uncentered, spearman_correlation,
    SimilarityAlgos,
};
use crate::utils::{sort_and_trucate, sort_with_direction};
use ndarray::{Array1, Array2};
use num_traits::Float;

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
/// let knn = KNN::new(new_item, references);
/// let result = knn.result();
/// println!("{:?}", result);
/// ```
///
#[doc = include_str!("../../docs/algorithms/knn.md")]
pub struct KNearestNeighbors<F> {
    neighbors_pool: Array2<F>,
    params: KNearestNeighborsParams<F>,
}

impl<F: Float> KNearestNeighbors<F> {
    pub fn new(neighbors_pool: Array2<F>) -> Self {
        Self {
            neighbors_pool,
            params: KNearestNeighborsParams::default(),
        }
    }
    pub fn set_algorithm(mut self, algorithm: SimilarityAlgos) -> Self {
        self.params.algorithm = algorithm;
        self
    }
    pub fn set_num_neighbors(mut self, num_neighbors: usize) -> Self {
        self.params.num_neighbors = num_neighbors;
        self
    }
    pub fn set_early_return_threshold(
        mut self,
        early_return_threshold: Option<F>,
    ) -> Self {
        self.params.early_return_threshold = early_return_threshold;
        self
    }
}

pub struct KNearestNeighborsParams<F> {
    algorithm: SimilarityAlgos,
    num_neighbors: usize,
    early_return_threshold: Option<F>,
}

type ParamDistanceFunction<F> = dyn Fn(&Array1<F>, &Array1<F>) -> F;

impl<F: Float> KNearestNeighborsParams<F> {
    fn default() -> Self {
        let threshold = F::from(0.999).unwrap();
        Self {
            algorithm: SimilarityAlgos::default(),
            num_neighbors: 10,
            early_return_threshold: Some(threshold),
        }
    }

    // fn get_formula(&self) -> (ParamDistanceFunction<F>, bool) {
    //     match self.algorithm {
    //         SimilarityAlgos::Cosine => (cosine_similarity, true),
    //         SimilarityAlgos::AdjustedCosine => (adjusted_cosine_similarity, true),
    //         SimilarityAlgos::Euclidean => (euclidean_distance, false),
    //         SimilarityAlgos::PearsonCorrelation => (pearson_correlation_uncentered, true),
    //         SimilarityAlgos::Spearman => (spearman_correlation, true),
    //         SimilarityAlgos::MSD => (msd_similarity, true),
    //     }
    // }
}

pub struct KNNResult {
    query_item: Item,
    result: SimilarityAlgos,
}
