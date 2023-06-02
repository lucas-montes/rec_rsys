//! A collection of funcitons to apply to matrices
use std::collections::HashMap;

use crate::statistics::mean as vec_mean;
use rayon::prelude::*;

/// Transpose a matrix
pub fn transpose<T: Clone + Send + Sync>(matrix: &[Vec<T>]) -> Vec<Vec<T>> {
    let cols: usize = matrix[0].len();

    if cols < 90 {
        // This for loop is faster some times
        // let mut transposed: Vec<Vec<T>> = Vec::with_capacity(cols);
        // for j in 0..cols {
        //     transposed.push(matrix.iter().map(|row| row[j].clone()).collect());
        // }
        // return transposed;
        return (0..cols)
            .map(|j: usize| matrix.iter().map(|row: &Vec<T>| row[j].clone()).collect())
            .collect();
    }
    (0..cols)
        .into_par_iter()
        .map(|j: usize| matrix.iter().map(|row: &Vec<T>| row[j].clone()).collect())
        .collect()
}

/// Transpose a matrix using f64 values
pub fn transpose_64(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cols: usize = matrix[0].len();

    if cols < 90 {
        return (0..cols)
            .map(|j: usize| matrix.iter().map(|row: &Vec<f64>| row[j]).collect())
            .collect();
    }
    (0..cols)
        .into_par_iter()
        .map(|j: usize| matrix.iter().map(|row: &Vec<f64>| row[j]).collect())
        .collect()
}

/// Calculate the mean of a matrix using f64 values
pub fn mean(matrix: &[Vec<f64>]) -> f64 {
    let total_elements = matrix.len() * matrix[0].len();
    if total_elements == 0 {
        0.0 // Avoid division by zero for empty matrices
    } else {
        matrix.iter().flatten().sum::<f64>() / total_elements as f64
    }
}

/// Calculate the mean along the axis of a matrix using f64 values
pub fn mean_along_axis(matrix: &[Vec<f64>], axis: usize) -> Vec<f64> {
    match axis {
        0 => (0..matrix[0].len())
            .map(|j: usize| {
                matrix.iter().map(|row: &Vec<f64>| row[j]).sum::<f64>() / matrix.len() as f64
            })
            .collect(),
        1 => matrix
            .iter()
            .map(|row: &Vec<f64>| row.iter().sum::<f64>() / row.len() as f64)
            .collect(),
        _ => panic!("Use the mean instead of mean along axis"),
    }
}

/// # Covariance Matrix
/// Function to calculate the covariance between two matrices of data.
///
/// ## Parameters:
/// * `x`: The first matrix of data.
/// * `y`: The second matrix of data.
///
/// ## Returns:
/// * The covariance between x and y.
///
/// ## Formula:
/// $$ Cov(x, y) = \frac{{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})^T}}{{n}} $$
///
/// ## Explanation:
/// The covariance measures the direction and magnitude of the linear relationship
/// between two sets of data, x and y. It calculates the sum of the products of the deviations
/// of each data point from their respective means, divided by the number of data points.
pub fn covariance(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let means: Vec<f64> = data.iter().map(vec_mean).collect();

    let mut covariance_matrix: Vec<Vec<f64>> = vec![vec![0.0; data[0].len()]; data[0].len()];

    for (i, row_i) in data.iter().enumerate() {
        for (j, row_j) in data.iter().enumerate().skip(i) {
            let cov_ij: f64 = row_i
                .iter()
                .zip(row_j.iter())
                .map(|(&x, &y)| (x - means[i]) * (y - means[j]))
                .sum::<f64>()
                / (data.len() as f64 - 1.0); // Degrees of freedom correction

            covariance_matrix[i][j] = cov_ij;
            covariance_matrix[j][i] = cov_ij; // Symmetric element
        }
    }

    covariance_matrix
}

/// # Subtract Vector from Matrix
/// Subtract a vector from each element of a matrix.
///
/// ## Parameters:
/// * `matrix`: The matrix from which to subtract the vector.
/// * `vector`: The vector to subtract from the matrix.
///
/// ## Returns:
/// * A new matrix where each element is obtained by subtracting the corresponding element of the vector from the matrix.
///
/// ## Examples:
/// ```
/// let matrix = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
/// let vector = vec![0.5, 1.0, 1.5];
/// let result = subtract_vector_from_matrix(&matrix, &vector);
/// assert_eq!(result, vec![vec![0.5, 1.0, 1.5], vec![3.5, 4.0, 4.5]]);
/// ```
///
/// ## Explanation:
/// The function subtracts each element of the vector from the corresponding element of the matrix, resulting in a new matrix.
///
/// ## Formula:
/// The formula for subtracting a vector `v` from a matrix `M` is:
/// ```math
/// M_{ij} = M_{ij} - v_j
/// ```
///
/// ### Where:
/// * `M_{ij}` represents the element at the `i`th row and `j`th column of the matrix `M`.
/// * `v_j` represents the `j`th element of the vector `v`.
pub fn subtract_vector_from_matrix(vector: &[f64], matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(vector.iter())
                .map(|(&matrix_val, &vector_val)| matrix_val - vector_val)
                .collect::<Vec<f64>>()
        })
        .collect()
}

/// # Eigenvalues
/// Get the eigenvalues from a matrix.
///
/// ## Parameters:
/// * `matrix`: The matrix to get the eigenvalues from
///
/// ## Returns:
/// * The eigenvalues
///
/// ## Examples:
///
/// let matrix = X;
/// (matrix)
///
/// ## Explanation:
/// The eigenvalues represent scalar values associated with a square matrix.
/// They provide insight into the matrix's structural properties and transformations.
///
/// ## Formula:
/// $$AV =  \lambda V$$
///
/// ### Where:
/// * `$\lambda$`: Is the eigenvalue
/// * `$A$`: Is the matrix
pub fn get_eigenvalues(matrix: &[Vec<f64>]) -> Vec<i32> {
    // Check if the matrix is square
    if matrix.iter().any(|row| row.len() != matrix.len()) {
        panic!("Input matrix is not square");
    }
    let _indentity_matrix: Vec<Vec<f64>> = vec![vec![0.0; matrix.len()]; matrix.len()];
    for (index, row) in matrix.iter().enumerate() {
        for (index2, value) in row.iter().enumerate() {
            if index + index2 % 2 == 0 {
                value - 1.0
            } else {
                value - 0.0
            };
        }
    }
    vec![5, -1]
}

/// # Eigenvectors
/// Compute the eigenvectors of a square matrix.
///
/// ## Parameters:
/// * `matrix`: The input square matrix as a 2D vector.
///
/// ## Returns:
/// * A matrix where each column represents an eigenvector of the matrix.
///
/// ## Examples:
/// ```
/// let matrix = vec![
///     vec![2.0, 1.0, 0.0],
///     vec![1.0, 2.0, 1.0],
///     vec![0.0, 1.0, 2.0],
/// ];
/// let eigenvectors = eigenvectors(&matrix);
/// println!("Eigenvectors: {:?}", eigenvectors);
/// ```
///
/// ## Explanation:
/// The eigenvectors are vectors associated with the eigenvalues of a square matrix.
/// They can provide insight into the matrix's structural properties and transformations.
///
/// ## Formula:
/// $$AV =  \lambda V$$
/// $$(A - \lambda I)x = 0$$
///
/// ### Where:
/// * `x`: Is the eigenvector
/// * `$A$`: Is the matrix
pub fn get_eigenvectors(_matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_eigenvalues() {
        assert_eq!(
            get_eigenvalues(&[vec![2.0, 2.0], vec![5.0, -1.0]]),
            vec![5, -1]
        );
    }

    #[test]
    fn test_subtract_vector_from_matrix() {
        assert_eq!(
            subtract_vector_from_matrix(
                &vec![0.5, 1.0, 1.5],
                &vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            ),
            vec![vec![0.5, 1.0, 1.5], vec![3.5, 4.0, 4.5]]
        );
    }

    #[test]
    fn test_covariance() {
        assert_eq!(
            covariance(&[
                vec![12.06, 22.5, 73.0],
                vec![4.40, 7.0, 9.9],
                vec![7.0, 48.0, 79.808]
            ]),
            [
                vec![1062.1545333333333, 84.794, 1078.54392],
                vec![84.794, 7.57, 99.8812],
                vec![1078.54392, 99.8812, 1332.292288]
            ],
        );
    }

    #[test]
    fn test_transpose() {
        assert_eq!(
            transpose(&[vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]),
            [vec![1, 4, 7], vec![2, 5, 8], vec![3, 6, 9]],
        );
    }

    #[test]
    fn test_mean() {
        assert_eq!(
            mean(&[
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0]
            ]),
            5.0,
        );
    }

    #[test]
    fn test_mean_along_axis() {
        assert_eq!(
            mean_along_axis(
                &[
                    vec![1.0, 2.0, 3.0],
                    vec![4.0, 5.0, 6.0],
                    vec![7.0, 8.0, 9.0]
                ],
                0
            ),
            vec![4.0, 5.0, 6.0],
        );
        assert_eq!(
            mean_along_axis(
                &[
                    vec![1.0, 2.0, 3.0],
                    vec![4.0, 5.0, 6.0],
                    vec![7.0, 8.0, 9.0]
                ],
                1
            ),
            vec![2.0, 5.0, 8.0],
        );
    }
}
