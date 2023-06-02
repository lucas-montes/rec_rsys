//! KNN
use std::collections::HashMap;

use crate::similarity::{cosine_similarity, euclidean_distance};
use crate::utils::sort_with_direction;

fn knn(
    new_input: Vec<f64>,
    references: HashMap<i8, Vec<f64>>,
    k: usize,
    formula: impl Fn(&Vec<f64>, &Vec<f64>) -> f64,
    reverse: bool,
) -> Vec<(f64, i8)> {
    sort_and_trucate(
        references
            .iter()
            .map(|(key, value)| (formula(&new_input, value), *key))
            .collect(),
        reverse,
        k,
    )
}

fn sort_and_trucate(mut best_matches: Vec<(f64, i8)>, reverse: bool, k: usize) -> Vec<(f64, i8)> {
    sort_with_direction(&mut best_matches, |(a, _), (b, _)| a.total_cmp(b), reverse);
    best_matches.truncate(k);
    best_matches
}

/// KNN algorithm using the euclidean distance
pub fn euclidean_knn(
    new_item: Vec<f64>,
    references: HashMap<i8, Vec<f64>>,
    k: usize,
) -> Vec<(f64, i8)> {
    knn(new_item, references, k, euclidean_distance, false)
}

/// KNN algorithm using the cosine similarity
pub fn cosine_knn(
    new_item: Vec<f64>,
    references: HashMap<i8, Vec<f64>>,
    k: usize,
) -> Vec<(f64, i8)> {
    knn(new_item, references, k, cosine_similarity, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_knn() {
        let mut refs: HashMap<i8, Vec<f64>> = HashMap::new();
        refs.insert(1, vec![0.9193, 0.9097, 0.4990, 0.3292, 0.8811]);
        refs.insert(2, vec![0.9826, 0.9977, 0.6924, 0.7509, 0.7644]);
        refs.insert(3, vec![0.4817, 0.7548, 0.1974, 0.2229, 0.1256]);
        refs.insert(4, vec![0.9376, 0.4734, 0.2254, 0.9728, 0.8401]);
        refs.insert(5, vec![0.7429, 0.3250, 0.5680, 0.2614, 0.4483]);
        refs.insert(6, vec![0.0686, 0.9531, 0.3464, 0.6426, 0.1746]);
        refs.insert(7, vec![0.2442, 0.3728, 0.3096, 0.1398, 0.8162]);
        refs.insert(8, vec![0.3682, 0.9574, 0.0486, 0.8852, 0.1986]);
        refs.insert(9, vec![0.3455, 0.2594, 0.7464, 0.0489, 0.4088]);
        refs.insert(10, vec![0.7193, 0.4097, 0.6990, 0.3292, 0.8811]);
        assert_eq!(
            cosine_knn(vec![0.9193, 0.9097, 0.4990, 0.3292, 0.8811], refs, 3),
            vec![(1.0, 1), (0.9696546846691771, 2), (0.9433796948413147, 10)],
        );
    }

    #[test]
    fn test_euclidean_knn() {
        let mut refs: HashMap<i8, Vec<f64>> = HashMap::new();
        refs.insert(1, vec![0.9193, 0.9097, 0.4990, 0.3292, 0.8811]);
        refs.insert(2, vec![0.9826, 0.9977, 0.6924, 0.7509, 0.7644]);
        refs.insert(3, vec![0.4817, 0.7548, 0.1974, 0.2229, 0.1256]);
        refs.insert(4, vec![0.9376, 0.4734, 0.2254, 0.9728, 0.8401]);
        refs.insert(5, vec![0.7429, 0.3250, 0.5680, 0.2614, 0.4483]);
        refs.insert(6, vec![0.0686, 0.9531, 0.3464, 0.6426, 0.1746]);
        refs.insert(7, vec![0.2442, 0.3728, 0.3096, 0.1398, 0.8162]);
        refs.insert(8, vec![0.3682, 0.9574, 0.0486, 0.8852, 0.1986]);
        refs.insert(9, vec![0.3455, 0.2594, 0.7464, 0.0489, 0.4088]);
        refs.insert(10, vec![0.7193, 0.4097, 0.6990, 0.3292, 0.8811]);

        assert_eq!(
            euclidean_knn(vec![0.9193, 0.9097, 0.4990, 0.3292, 0.8811], refs, 3),
            vec![(0.0, 1), (0.4905142505575144, 2), (0.5744562646538027, 10)],
        );
    }
}
