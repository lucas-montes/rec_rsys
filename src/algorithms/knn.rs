//! KNN
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::iter::Sum;

use crate::models::{DatasetBase, ItemResult, Numeric};
use crate::similarity::SimilarityAlgorithm;
use crate::utils::sort_with_direction;
use ndarray::{s, Array1, Array2, ArrayView1, Axis};
use num_traits::{Float, FromPrimitive};

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
/// use ndarray::array;
/// use rec_rsys::{
///     algorithms::knn::KNearestNeighbors, models::DatasetBase,
///     similarity::SimilarityAlgorithm,
/// };
/// let records = array![
///     [0.9193, 0.9097, 0.4990, 0.3292, 0.8811],
///     [0.9826, 0.9977, 0.6924, 0.7509, 0.7644],
///     [0.4817, 0.7548, 0.1974, 0.2229, 0.1256],
/// ];
/// let dataset = DatasetBase::new(records);
/// let model = KNearestNeighbors::new(dataset);
/// let result = model.predict(0).results();
/// println!("{:?}", result);
/// ```
///
#[doc = include_str!("../../docs/algorithms/knn.md")]
pub struct KNearestNeighbors<F> {
    dataset: DatasetBase<F>,
    params: KNearestNeighborsParams<F>,
}

impl<F: Numeric> KNearestNeighbors<F> {
    pub fn new(dataset: DatasetBase<F>) -> Self {
        Self {
            dataset,
            params: KNearestNeighborsParams::default(),
        }
    }
    pub fn set_algorithm(mut self, algorithm: SimilarityAlgorithm) -> Self {
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

    pub fn predict(&self, index: usize) -> KNearestNeighborsResult<F> {
        let similarity_fn = self.params.get_similarity_algorithm();
        let query = self.dataset.get_row(index);

        let mut results = KNearestNeighborsResult::new(self.params.num_neighbors);
        if let Some(threshold) = self.params.early_return_threshold {
            for (i, q) in self.dataset.rows() {
                let result = similarity_fn(&query, &q);
                if result > threshold {
                    results.push(result, i);
                    if results.is_full() {
                        break;
                    }
                };
            }
        } else {
            for (i, q) in self.dataset.rows() {
                let result = similarity_fn(&query, &q);
                results.push(result, i);
            }
        };

        results
    }
}

pub struct KNearestNeighborsResult<F: Numeric> {
    results: BinaryHeap<ItemResult<F>>,
}

impl<F: Numeric> KNearestNeighborsResult<F> {
    fn new(size: usize) -> Self {
        Self {
            results: BinaryHeap::with_capacity(size + 1),
        }
    }

    fn push(&mut self, result: F, index: usize) {
        self.results.push(ItemResult::new(result, index));
    }

    fn is_full(&self) -> bool {
        self.results.capacity() == self.results.len()
    }

    pub fn results(self) -> Vec<ItemResult<F>> {
        self.results.into_sorted_vec()
    }
}

pub struct KNearestNeighborsParams<F> {
    algorithm: SimilarityAlgorithm,
    num_neighbors: usize,
    early_return_threshold: Option<F>,
}

impl<F: Numeric> KNearestNeighborsParams<F> {
    fn default() -> Self {
        let threshold = F::from(0.93).unwrap();
        Self {
            algorithm: SimilarityAlgorithm::default(),
            num_neighbors: 10,
            early_return_threshold: Some(threshold),
        }
    }

    fn get_similarity_algorithm(
        &self,
    ) -> Box<dyn Fn(&ArrayView1<F>, &ArrayView1<F>) -> F> {
        self.algorithm.get_function()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_slices() {
        let a: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> =
            array![[3.0, 45.0, 7.0, 2.0], [3.0, 20.0, 7.0, 2.0]];

        let y: std::iter::Enumerate<
            ndarray::iter::AxisIter<'_, f64, ndarray::Dim<[usize; 1]>>,
        > = a.outer_iter().enumerate();

        println!("{:?}", y);
    }
}
