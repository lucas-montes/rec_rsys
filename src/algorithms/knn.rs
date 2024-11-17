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

    pub fn predict(&self, item: usize) -> KNearestNeighborsResult<F> {
        let similarity_fn = self.params.get_similarity_algorithm();
        let query = self.dataset.get_row(item);

        let mut results = KNearestNeighborsResult::new(self.params.num_neighbors);
        if let Some(t) = self.params.early_return_threshold {
            for (i, q) in self.dataset.rows() {
                let result = similarity_fn(&query, &q);
                results.push(result, i);

                if i % self.params.num_neighbors == 0 && results.is_full(&t) {
                    break;
                }
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

    fn is_full(&mut self, threshold: &F) -> bool {
        if self.results.capacity() == self.results.len() {
            self.results.retain(|i| &i.value() > threshold);
            return false;
        }
        true
    }

    pub fn results(self) -> Vec<ItemResult<F>> {
        let mut results = self.results.into_sorted_vec();
        results.reverse();
        results
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
