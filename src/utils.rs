//! # A collection of tools
//!

/// # Dot product
/// Calculates the dot product between two vectors.
///
/// ## Prameters:
/// * `vec1`: The first vector.
/// * `vec2`: The second vector.
///
/// ## Returns:
/// The dot product of the two vectors.
pub fn dot(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
    vec1.iter().zip(vec2.iter()).map(|(&x, &y)| x * y).sum()
}

/// # Euclidean norm
/// Calculates the magnitude (Euclidean norm) of a vector.
///
/// ## Prameters:
/// * `vec`: The vector.
///
/// ## Returns:
/// The magnitude of the vector.
pub fn euclidean_norm(vec: &Vec<f64>) -> f64 {
    vec.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// TODO
pub fn squared_diff_sum(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
    vec1.iter()
        .zip(vec2.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>()
}

/// TODO
pub fn local_sort(v: &mut Vec<f64>) {
    v.sort_by(|x: &f64, y: &f64| x.total_cmp(y))
}

/// Function to calculate the ranks of the values in a vector.
///
/// ## Parameters:
/// * `vector`: The vector of values.
///
/// ## Returns:
/// * A vector containing the ranks of the values.
pub fn rank_vector(vector: &Vec<f64>) -> Vec<f64> {
    let mut indexed_vector: Vec<(usize, &f64)> = vector.iter().enumerate().collect();
    indexed_vector.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
    indexed_vector
        .iter()
        .map(|(i, _)| *i as f64 + 1.0)
        .collect()
}

/// Sorts the elements in the given vector `vector` using the provided comparison function `compare_fn`,
/// with the option to reverse the sort order if `reverse` is `true`.
///
/// ## Parameters:
/// * `vector`: The vector to be sorted.
/// * `compare_fn`: The comparison function that compares two elements and returns an `Ordering`.
///                  It should take two references to elements of type `T` and return an `Ordering` value.
/// * `reverse`: A flag indicating whether to sort the elements in reverse order.
///
/// ## Example
/// ```
/// let mut numbers = vec![4, 2, 8, 5, 1];
///
/// sort_with_direction(&mut numbers, |a, b| a.cmp(b), false);
/// println!("Ascending order: {:?}", numbers);
///
/// sort_with_direction(&mut numbers, |a, b| a.cmp(b), true);
/// println!("Descending order: {:?}", numbers);
/// ```
pub fn sort_with_direction<T, F>(vector: &mut [T], compare_fn: F, reverse: bool)
where
    F: Fn(&T, &T) -> std::cmp::Ordering,
{
    if reverse {
        vector.sort_by(|a, b| compare_fn(b, a));
    } else {
        vector.sort_by(compare_fn);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_with_direction_ascending() {
        let mut numbers: Vec<i32> = vec![4, 2, 8, 5, 1];
        sort_with_direction(&mut numbers, |a, b| a.cmp(b), false);
        assert_eq!(numbers, vec![1, 2, 4, 5, 8],);
    }

    #[test]
    fn test_sort_with_direction_descending() {
        let mut numbers: Vec<i32> = vec![4, 2, 8, 5, 1];
        sort_with_direction(&mut numbers, |a, b| a.cmp(b), true);
        assert_eq!(numbers, vec![8, 5, 4, 2, 1],);
    }

    #[test]
    fn test_dot() {
        assert_eq!(
            dot(&vec![3.0, 45.0, 7.0, 2.0], &vec![2.0, 54.0, 13.0, 15.0]),
            2557.0,
        );
    }

    #[test]
    fn test_euclidean_norm() {
        assert_eq!(
            euclidean_norm(&vec![3.0, 45.0, 7.0, 2.0]),
            45.68369512200168,
        );
    }

    #[test]
    fn test_squared_diff_sum() {
        assert_eq!(
            squared_diff_sum(&vec![3.0, 45.0, 7.0, 2.0], &vec![2.0, 54.0, 13.0, 15.0]),
            287.0,
        );
    }

    #[test]
    fn test_rank_vector() {
        assert_eq!(
            rank_vector(&vec![3.0, 45.0, 7.0, 2.0]),
            vec![1.0, 3.0, 2.0, 0.0],
        );
    }
}
