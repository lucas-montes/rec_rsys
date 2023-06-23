//! # A collection of tools to compute the accuracy of the results
//! $y_i =$ actual rating
//! <br>
//! $x_i =$ predicted rating
//! <br>
//! $n =$ number of ratings

/// # RMSE (Root Mean Squared Error).
///
/// ## Parameters:
/// * `predicted`: The estimated rating.
/// * `actual`: The true rating.
///
/// ## Returns:
/// * The Root Mean Squared Error.
///
#[doc = include_str!("../docs/accuracy/rmse.md")]
pub fn rmse(predicted: &[f32], actual: &Vec<f32>) -> f32 {
    mse(predicted, actual).sqrt()
}

/// # Compute MSE (Mean Squared Error).
///
/// ## Parameters:
/// * `predicted`: The estimated rating.
/// * `actual`: The true rating.
///
/// ## Returns:
/// * The Mean Squared Error.
///
#[doc = include_str!("../docs/accuracy/mse.md")]
pub fn mse(predicted: &[f32], actual: &Vec<f32>) -> f32 {
    actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f32>()
        / actual.len() as f32
}

/// # Compute MAE (Mean Absolute Error).
///
/// ## Parameters:
/// * `predicted`: The estimated rating.
/// * `actual`: The true rating.
///
/// ## Returns:
/// * The Mean Absolute Error.
///
#[doc = include_str!("../docs/accuracy/mae.md")]
pub fn mae(predicted: &Vec<f32>, actual: &Vec<f32>) -> f32 {
    actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p).abs())
        .sum::<f32>()
        / actual.len() as f32
}

/// # Compute ARHR (Average reciprocal hit rate)
/// Rewards the recommendations that finish in top spots.
/// Meaning that they have been more useful for the user
///
/// ## Parameters:
/// * `hits_ranks`: A vector with the rank of each hit.
/// * `number_users`: The total number of users.
///
/// ## Returns:
/// * Average reciprocal hit rate.
///
#[doc = include_str!("../docs/accuracy/arhr.md")]
pub fn arhr(hits_ranks: Vec<u32>, number_users: u32) -> u32 {
    hits_ranks.iter().map(|rank: &u32| 1 / rank).sum::<u32>() / number_users
}

/// # Compute Hit Rate
///
/// ## Parameters:
/// * `number_hits`: The total number of hits.
/// * `number_users`: The total number of users.
///
/// ## Returns:
/// * Hit Rate.
///
#[doc = include_str!("../docs/accuracy/hit_rate.md")]
pub fn hit_rate(number_hits: u32, number_users: u32) -> u32 {
    number_hits / number_users
}

/// # Compute Hit Rate from vectors
///
/// ## Parameters:
/// * `hits`: A vector with hits.
/// * `users`: A vector with users.
///
/// ## Returns:
/// * Hit Rate.
///
#[doc = include_str!("../docs/accuracy/vec_hit_rate.md")]
pub fn vec_hit_rate(hits: Vec<u32>, users: Vec<u32>) -> u32 {
    hits.iter().sum::<u32>() / users.len() as u32
}

/// # Cumulative Hit Rate
/// Function to calculate the Cumulative Hit Rate (CHR) given a list of predicted items and a list of true items.
///
/// ## Parameters:
/// * `predicted_items`: A list of predicted items.
/// * `true_items`: A list of true items.
///
/// ## Returns:
/// * The Cumulative Hit Rate (CHR).
///
#[doc = include_str!("../docs/accuracy/cumulative_hit_rate.md")]
fn cumulative_hit_rate(predicted_items: &[u32], true_items: &[u32]) -> f32 {
    predicted_items
        .iter()
        .filter(|&item| true_items.contains(item))
        .count() as f32
        / true_items.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_cumulative_hit_rate() {
    //     assert_eq!(cumulative_hit_rate(), 1.0)
    // }

    // #[test]
    // fn test_rmse() {
    //     assert_eq!(rmse(), 1.0)
    // }

    // #[test]
    // fn test_mse() {
    //     assert_eq!(mse(), 1.0)
    // }

    // #[test]
    // fn test_mae() {
    //     assert_eq!(mae(), 1.0)
    // }

    // #[test]
    // fn test_arhr() {
    //     assert_eq!(arhr(), 1.0)
    // }

    #[test]
    fn test_hit_rate() {
        assert_eq!(hit_rate(8, 4), 2);
    }

    #[test]
    fn test_vec_hit_rate() {
        assert_eq!(vec_hit_rate(vec![9, 7, 8], vec![0, 0, 0]), 8);
    }
}
