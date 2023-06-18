//! KNN
use crate::models::Item;
use crate::similarity::{
    adjusted_cosine_similarity, cosine_similarity, euclidean_distance, msd_similarity,
    pearson_baseline_similarity, pearson_correlation, spearman_correlation,
    SimilarityAlgos,
};
use crate::utils::sort_with_direction;

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
    pub fn vectors_comparaison(self, algo: SimilarityAlgos) -> Vec<Item> {
        let (formula, reverse) = KNN::get_formula(algo);
        let mut best_matches: Vec<Item> = Vec::new();
        self.references.iter().for_each(|item| {
            let cloned_item = item.clone();
            best_matches
                .push(cloned_item.result(formula(&self.new_item.values, &item.values)))
        });

        sort_and_trucate(best_matches, reverse, self.k)
    }
    fn get_formula(
        algo: SimilarityAlgos,
    ) -> (&'static dyn Fn(&Vec<f32>, &Vec<f32>) -> f32, bool) {
        match algo {
            SimilarityAlgos::Cosine => (&cosine_similarity, true),
            SimilarityAlgos::AdjustedCosine => (&adjusted_cosine_similarity, true),
            SimilarityAlgos::Euclidean => (&euclidean_distance, false),
            SimilarityAlgos::PearsonCorrelation => (&pearson_correlation, true),
            SimilarityAlgos::Spearman => (&spearman_correlation, true),
            SimilarityAlgos::MSD => (&msd_similarity, true),
        }
    }
}

/// KNN algorithm using the euclidean distance
pub fn euclidean_knn(new_item: Item, references: Vec<Item>, k: u8) -> Vec<Item> {
    knn(new_item, references, k, euclidean_distance, false)
}

/// KNN algorithm using the cosine similarity
pub fn cosine_knn(new_item: Item, references: Vec<Item>, k: u8) -> Vec<Item> {
    knn(new_item, references, k, cosine_similarity, true)
}

fn knn(
    new_input: Item,
    mut references: Vec<Item>,
    k: u8,
    formula: impl Fn(&Vec<f32>, &Vec<f32>) -> f32,
    reverse: bool,
) -> Vec<Item> {
    references
        .iter_mut()
        .for_each(|item| item.result = formula(&new_input.values, &item.values));

    sort_and_trucate(references, reverse, k)
}

fn sort_and_trucate(mut best_matches: Vec<Item>, reverse: bool, k: u8) -> Vec<Item> {
    sort_with_direction(
        &mut best_matches,
        |item_a, item_b| item_a.result.total_cmp(&item_b.result),
        reverse,
    );
    best_matches.truncate(k as usize);
    best_matches
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Item;

    fn mock(_: &Vec<f32>, m1: &Vec<f32>) -> f32 {
        m1[0]
    }

    #[test]
    fn test_knn() {
        let item1 = Item::new(1, vec![0.9193, 0.9097, 0.4990, 0.3292, 0.8811], Some(1.0));
        let item2 =
            Item::new(2, vec![0.9826, 0.9977, 0.6924, 0.7509, 0.7644], Some(0.33));
        let item3 = Item::new(3, vec![0.4817, 0.7548, 0.1974, 0.2229, 0.1256], Some(0.0));

        assert_eq!(
            knn(
                item1.clone(),
                vec![item1.clone(), item2.clone(), item3.clone()],
                6,
                mock,
                true
            ),
            vec![item2, item1, item3]
        );
    }

    #[test]
    fn test_sort_and_trucate() {
        let item1 = Item::new(1, vec![0.9193, 0.9097, 0.4990, 0.3292, 0.8811], Some(1.0));
        let item2 =
            Item::new(2, vec![0.9826, 0.9977, 0.6924, 0.7509, 0.7644], Some(0.33));
        let item3 = Item::new(3, vec![0.4817, 0.7548, 0.1974, 0.2229, 0.1256], Some(0.0));
        assert_eq!(
            sort_and_trucate(vec![item1.clone(), item2.clone(), item3], true, 2),
            vec![item1, item2]
        );
    }
}
