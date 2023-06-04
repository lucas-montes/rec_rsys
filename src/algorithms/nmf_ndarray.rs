///! Fails to compile due to missing dependencies.
///! Using ndarray requires many dependencies, not only in Rust
extern crate ndarray;
extern crate rand;
extern crate rand_distr;

use ndarray::prelude::*;
use rand::prelude::*;
use rand_distr::Uniform;

/// # Non-Negative Matrix Factorization (NMF)
/// Decomposes a given matrix into two non-negative matrices using NMF.
///
/// ## Parameters:
/// * `matrix`: The input matrix to be factorized.
/// * `n_components`: The number of components (columns) in the factorized matrices.
/// * `max_iter`: The maximum number of iterations for the NMF algorithm.
///
/// ## Returns:
/// * A tuple `(W, H)` representing the factorized matrices `W` and `H`.
///
/// ## Examples:
/// ```
/// use ndarray::prelude::*;
///
/// // Create an example matrix
/// let matrix = arr2(&[
///     [1.0, 2.0, 3.0],
///     [4.0, 5.0, 6.0],
///     [7.0, 8.0, 9.0],
/// ]);
///
/// // Perform NMF with 2 components
/// let (W, H) = nmf(matrix, 2, 100);
///
/// println!("Factorized Matrix W:\n{:?}", W);
/// println!("Factorized Matrix H:\n{:?}", H);
/// ```
///
/// ## Explanation:
/// Non-Negative Matrix Factorization (NMF) factorizes a given matrix `V` into two non-negative matrices `W` and `H`.
/// It aims to find `W` and `H` such that their product approximates `V`:
///
/// ```
/// V ≈ W * H
/// ```
///
/// The factorized matrix `W` consists of basis vectors, and `H` contains coefficients for these basis vectors.
///
/// ## Formula:
/// NMF aims to minimize the Frobenius norm of the difference between `V` and `W * H` by updating `W` and `H` iteratively.
///
/// The update equations for `W` and `H` can be expressed as follows:
///
/// ```
/// W_{ij} = W_{ij} * ((V * H^T) / (W * H * H^T))
/// H_{ij} = H_{ij} * ((W^T * V) / (W^T * W * H))
/// ```
///
/// Where:
/// * `V` is the input matrix.
/// * `W` is the factorized matrix of size `(m, n_components)`.
/// * `H` is the factorized matrix of size `(n_components, n)`.
/// * `m` is the number of rows in `V`.
/// * `n` is the number of columns in `V`.
/// * `n_components` is the number of components (columns) in `W` and `H`.
/// * `^T` denotes matrix transpose.
///
pub fn nmf(
    matrix: Array2<f32>,
    n_components: usize,
    max_iter: usize,
) -> (Array2<f32>, Array2<f32>) {
    let (m, n) = matrix.dim();

    // Initialize W and H matrices with random non-negative values
    let mut rng = thread_rng();
    let uniform = Uniform::new(0.0, 1.0);
    let mut w = Array::from_shape_fn((m, n_components), |_| uniform.sample(&mut rng));
    let mut h = Array::from_shape_fn((n_components, n), |_| uniform.sample(&mut rng));

    let mut w_update = Array2::zeros((m, n_components));
    let mut h_update = Array2::zeros((n_components, n));

    for _ in 0..max_iter {
        // Update W
        let v_h_transpose = matrix.dot(&h.t());
        let w_h_h_transpose = w.dot(&h.dot(&h.t()));
        w_update.assign(&(&w * &v_h_transpose / &w_h_h_transpose));
        w.assign(&w_update);

        // Update H
        let w_transpose_v = w.t().dot(&matrix);
        let w_transpose_w_h = w.t().dot(&w.dot(&h));
        h_update.assign(&(&h * &w_transpose_v / &w_transpose_w_h));
        h.assign(&h_update);
    }

    (w, h)
}
