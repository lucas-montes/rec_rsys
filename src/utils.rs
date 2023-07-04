//! # A collection of tools
//!

use crate::models::Item;

/// # Dot product
/// Calculates the dot product between two vectors.
///
/// ## Prameters:
/// * `x`: The first vector.
/// * `y`: The second vector.
///
/// ## Returns:
/// The dot product of the two vectors.
pub fn dot(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).map(|(&x, &y)| x * y).sum()
}

/// # Euclidean norm
/// Calculates the magnitude (Euclidean norm) of a vector.
///
/// ## Prameters:
/// * `x`: The vector.
///
/// ## Returns:
/// The magnitude of the vector.
pub fn euclidean_norm(x: &[f32]) -> f32 {
    x.iter().map(|&a| a * a).sum::<f32>().sqrt()
}

/// TODO
pub fn squared_diff_sum(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f32>()
}

/// TODO
pub fn local_sort(v: &mut [f32]) {
    v.sort_by(|x: &f32, y: &f32| x.total_cmp(y))
}

/// Function to calculate the ranks of the values in a vector.
///
/// ## Parameters:
/// * `x`: The vector of values.
///
/// ## Returns:
/// * Returns the indices that would sort an array.
///
/// ## Explanation:
/// Perform an indirect sort along the given axis (-1).
/// It returns an array of indices of the same shape as
/// `vector` that index data along the given axis in sorted order.
pub fn argsort(x: &[f32]) -> Vec<f32> {
    let mut indexed_vector: Vec<(usize, &f32)> = x.iter().enumerate().collect();
    indexed_vector.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
    indexed_vector.iter().map(|(i, _)| *i as f32).collect()
}

/// Sorts the elements in the given vector `vector` using the provided comparison function `compare_fn`,
/// with the option to reverse the sort order if `reverse` is `true`.
///
/// ## Parameters:
/// * `v`: The vector to be sorted.
/// * `compare_fn`: The comparison function that compares two elements and returns an `Ordering`.
///                  It should take two references to elements of type `T` and return an `Ordering` value.
/// * `reverse`: A flag indicating whether to sort the elements in reverse order.
///
/// ## Example
/// ```
/// use rec_rsys::utils::sort_with_direction;
/// let mut numbers = vec![4, 2, 8, 5, 1];
///
/// sort_with_direction(&mut numbers, |a, b| a.cmp(b), false);
/// println!("Ascending order: {:?}", numbers);
///
/// sort_with_direction(&mut numbers, |a, b| a.cmp(b), true);
/// println!("Descending order: {:?}", numbers);
/// ```
pub fn sort_with_direction<T, F>(v: &mut [T], compare_fn: F, reverse: bool)
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    if reverse {
        v.sort_by(|a, b| compare_fn(b, a));
    } else {
        v.sort_by(compare_fn);
    }
}

pub fn sort_and_trucate(mut best_matches: Vec<Item>, reverse: bool, k: u8) -> Vec<Item> {
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

    #[test]
    fn test_sort_with_direction_ascending() {
        let mut numbers: Vec<u8> = vec![4, 2, 8, 5, 1];
        sort_with_direction(&mut numbers, |a, b| a.cmp(b), false);
        assert_eq!(numbers, vec![1, 2, 4, 5, 8],);
    }

    #[test]
    fn test_sort_with_direction_descending() {
        let mut numbers: Vec<u8> = vec![4, 2, 8, 5, 1];
        sort_with_direction(&mut numbers, |a, b| a.cmp(b), true);
        assert_eq!(numbers, vec![8, 5, 4, 2, 1],);
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

    #[test]
    fn test_dot() {
        assert_eq!(
            dot(&[3.0, 45.0, 7.0, 2.0], &[2.0, 54.0, 13.0, 15.0]),
            2557.0,
        );
    }

    #[test]
    fn test_euclidean_norm() {
        assert_eq!(euclidean_norm(&[3.0, 45.0, 7.0, 2.0]), 45.683_697,);
    }

    #[test]
    fn test_squared_diff_sum() {
        assert_eq!(
            squared_diff_sum(&[3.0, 45.0, 7.0, 2.0], &[2.0, 54.0, 13.0, 15.0]),
            287.0,
        );
    }

    #[test]
    fn test_argsort() {
        assert_eq!(argsort(&[3.0, 45.0, 7.0, 2.0]), vec![3.0, 0.0, 2.0, 1.0],);
    }
}
