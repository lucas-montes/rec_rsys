//! # A collection of tools
//!

use std::ops::Sub;

use ndarray::Array1;
use num_traits::{float::TotalOrder, Float, FromPrimitive};

use crate::models::Item;

/// # Euclidean norm
/// Calculates the magnitude (Euclidean norm) of a vector.
///
/// ## Prameters:
/// * `x`: The vector.
///
/// ## Returns:
/// The magnitude of the vector.
pub fn euclidean_norm<A: Float + FromPrimitive + std::iter::Sum>(x: &Array1<A>) -> A {
    x.iter().map(|&a| a * a).sum::<A>().sqrt()
}

/// TODO: docs
pub fn squared_diff_sum<A: Float + FromPrimitive + std::iter::Sum>(
    x: &Array1<A>,
    y: &Array1<A>,
) -> A {
    x.iter()
        .zip(y.iter())
        .map(|(&a, &p)| (a - p).powi(2))
        .sum::<A>()
}

/// TODO
pub fn local_sort<A: Float + FromPrimitive + TotalOrder>(v: &mut [A]) {
    v.sort_by(|x: &A, y: &A| x.total_cmp(y))
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
pub fn argsort<A: Float + FromPrimitive>(x: &Array1<A>) -> Array1<A> {
    let mut indexed_vector: Vec<(usize, &A)> = x.iter().enumerate().collect();
    indexed_vector.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
    indexed_vector
        .iter()
        .map(|(i, _)| A::from(*i).unwrap())
        .collect()
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

pub fn sort_and_trucate(best_matches: &mut Vec<Item>, reverse: bool, k: usize) {
    sort_with_direction(
        best_matches,
        |item_a, item_b| item_a.result.total_cmp(&item_b.result),
        reverse,
    );
    best_matches.truncate(k);
}

#[cfg(test)]
mod tests {
    use ndarray::array;

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
        let mut initial = vec![item1.clone(), item2.clone(), item3];
        sort_and_trucate(&mut initial, true, 2);
        assert_eq!(initial, vec![item1, item2]);
    }

    #[test]
    fn test_euclidean_norm() {
        assert_eq!(
            euclidean_norm(&array![3.0, 45.0, 7.0, 2.0]),
            45.68369512200168
        );
    }

    #[test]
    fn test_squared_diff_sum() {
        assert_eq!(
            squared_diff_sum(
                &array![3.0, 45.0, 7.0, 2.0],
                &array![2.0, 54.0, 13.0, 15.0]
            ),
            287.0,
        );
    }

    #[test]
    fn test_argsort() {
        assert_eq!(
            argsort(&array![3.0, 45.0, 7.0, 2.0]),
            array![3.0, 0.0, 2.0, 1.0],
        );
    }
}
