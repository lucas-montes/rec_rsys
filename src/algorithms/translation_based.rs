use std::collections::{HashMap, HashSet};

/// # Translation-based Recommendation
/// Translation-based recommendation is a technique used in recommendation systems to provide personalized recommendations by leveraging semantic relationships between items. It utilizes vector representations of items to measure similarity and make recommendations.
///
/// ## Parameters:
/// * `ratings`: A hashmap of user-item ratings, where the keys are user IDs and the values are sets of item IDs.
/// * `embedding_dim`: The dimensionality of item embeddings.
/// * `learning_rate`: The learning rate for the training process.
/// * `num_epochs`: The number of training epochs.
///
/// ## Returns:
/// * None
///
/// ## Examples:
/// ```rust
/// let mut model = TranslationBasedModel::new();
/// let ratings = hashmap! {
///     0 => hashset!{1, 2},
///     1 => hashset!{0, 2},
///     2 => hashset!{0, 1}
/// };
/// model.train(&ratings, 10, 0.01, 100);
/// let recommendations = model.recommend_items(&0, 5);
/// println!("Recommendations for user 0: {:?}", recommendations);
/// ```
///
/// ## Explanation:
/// Translation-based recommendation leverages the idea that similar items should have similar vector representations. It trains item embeddings using a translation-based approach, where each item's embedding is adjusted based on positive and negative samples. The trained embeddings are then used to calculate the similarity between user and item vectors, providing personalized recommendations.
///
/// ## Formula:
/// The similarity between two vectors, `embedding1` and `embedding2`, is calculated using the negative Euclidean distance as follows:
///
/// ```math
/// \text{similarity}(\text{embedding1}, \text{embedding2}) = e^{-\sum_{i=1}^{n} (embedding1_i - embedding2_i)^2}
/// ```
///
/// ### Where:
/// * `embedding1_i`: The i-th component of `embedding1`.
/// * `embedding2_i`: The i-th component of `embedding2`.
/// * `n`: The dimensionality of the embeddings.
///
/// The training process involves updating item embeddings using gradient descent. Positive and negative embeddings are sampled, and the embeddings are adjusted to minimize the difference between positive and negative embeddings.

struct TranslationBasedModel {
    item_embeddings: HashMap<usize, Vec<f32>>,
}

impl TranslationBasedModel {
    fn new() -> Self {
        TranslationBasedModel {
            item_embeddings: HashMap::new(),
        }
    }

    fn train(
        &mut self,
        ratings: &HashMap<usize, HashSet<usize>>,
        embedding_dim: usize,
        learning_rate: f32,
        num_epochs: usize,
    ) {
        // Generate initial embeddings randomly
        for item_id in ratings.keys() {
            let embedding = (0..embedding_dim)
                .map(|_| rand::random::<f32>())
                .collect::<Vec<f32>>();
            self.item_embeddings.insert(*item_id, embedding);
        }

        // Training loop
        for _ in 0..num_epochs {
            let item_embeddings_copy = self.item_embeddings.clone(); // Create a copy of item_embeddings
            for (user_id, items) in ratings {
                for item_id in items {
                    let pos_embedding = item_embeddings_copy.get(item_id).unwrap();
                    let neg_embedding = self.sample_negative_embedding(user_id, items);

                    // Update embeddings using gradient descent
                    for i in 0..embedding_dim {
                        let gradient = pos_embedding[i] - neg_embedding[i];
                        self.item_embeddings.get_mut(item_id).unwrap()[i] -=
                            learning_rate * gradient;
                    }
                }
            }
        }
    }

    fn sample_negative_embedding(&self, user_id: &usize, items: &HashSet<usize>) -> Vec<f32> {
        loop {
            let random_item = rand::random::<usize>() % self.item_embeddings.len();
            if !items.contains(&random_item) {
                return self.item_embeddings[&random_item].clone();
            }
        }
    }

    fn recommend_items(&self, user_id: &usize, top_n: usize) -> Vec<usize> {
        let user_embedding = self.calculate_user_embedding(user_id);
        let mut item_scores: Vec<(usize, f32)> = self
            .item_embeddings
            .iter()
            .map(|(item_id, item_embedding)| {
                let score = self.calculate_similarity(&user_embedding, item_embedding);
                (*item_id, score)
            })
            .collect();

        item_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        item_scores
            .into_iter()
            .take(top_n)
            .map(|(item_id, _)| item_id)
            .collect()
    }

    fn calculate_user_embedding(&self, user_id: &usize) -> Vec<f32> {
        let user_embedding = self
            .item_embeddings
            .values()
            .filter_map(|embedding| embedding.get(*user_id))
            .map(|&value| value)
            .collect::<Vec<f32>>();

        let num_items = self.item_embeddings.len() as f32;
        user_embedding
            .iter()
            .map(|&value| value / num_items)
            .collect()
    }

    fn calculate_similarity(&self, embedding1: &[f32], embedding2: &[f32]) -> f32 {
        let sum_squared_diff: f32 = embedding1
            .iter()
            .zip(embedding2.iter())
            .map(|(&value1, &value2)| (value1 - value2) * (value1 - value2))
            .sum();

        (-sum_squared_diff).exp()
    }
}
