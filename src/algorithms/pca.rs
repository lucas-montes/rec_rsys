use crate::matrix::{covariance, mean_along_axis, subtract_vector_from_matrix, transpose};
/// # PCA (Principal Component Analysis)
/// PCA is a dimensionality reduction technique that finds the principal components in the data.
/// It identifies the directions (principal components) in which the data varies the most and projects the data onto those components,
/// resulting in a new set of uncorrelated variables called the principal components.
///
/// ## Parameters:
/// * `n_components`: The number of principal components to retain.
///
/// ## Examples:
/// ```
/// // Example usage
/// let mut pca = PCA::new(2);
/// pca.fit(&data);
/// let transformed_data = pca.transform(&data);
/// ```
///
/// ## Explanation:
/// PCA computes the principal components by performing the following steps:
/// 1. Mean centering the data.
/// 2. Calculating the covariance matrix.
/// 3. Finding the eigenvalues and eigenvectors of the covariance matrix.
/// 4. Sorting the eigenvectors based on their corresponding eigenvalues.
/// 5. Selecting the top `n_components` eigenvectors as the principal components.
///
/// ## Formula:
/// The formula for PCA is as follows:
///
/// ```katex
/// X_{\text{transformed}} = X - \bar{X} \cdot V^T
/// ```
///
/// ### Where:
/// * `X_{\text{transformed}}` represents the transformed data matrix.
/// * `X` is the original data matrix.
/// * `\bar{X}` represents the mean of each feature.
/// * `V` represents the matrix of eigenvectors (principal components).
pub struct PCA {
    n_components: usize,
    components: Option<Vec<Vec<f64>>>,
    mean: Option<Vec<f64>>,
    sorted_eigenvalues: Option<Vec<f64>>,
}

impl PCA {
    pub fn new(n_components: usize) -> Self {
        PCA {
            n_components,
            components: None,
            mean: None,
            sorted_eigenvalues: None,
        }
    }

    pub fn fit(&mut self, mut x: &[Vec<f64>]) {
        // Mean centering
        let centered_x = mean_along_axis(x, 0);
        x = &subtract_vector_from_matrix(&centered_x, &x);

        // Covariance
        let cov = covariance(&transpose(x));

        // Eigenvalues, Eigenvectors
        let (eigenvalues, eigenvectors) = eigen(cov);

        // Sort eigenvectors
        let mut eigenvecs = transpose(&eigenvectors);
        let mut idxs: Vec<usize> = (0..x[0].len()).collect();
        idxs.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());
        let sorted_eigenvalues = idxs.iter().map(|&i| eigenvalues[i]).collect::<Vec<f64>>();
        eigenvecs = idxs
            .iter()
            .map(|&i| eigenvecs[i].clone())
            .collect::<Vec<Vec<f64>>>();

        // Store first n eigenvectors
        let components = eigenvecs[..self.n_components].to_vec();

        self.mean = Some(centered_x);
        self.components = Some(components);
        self.sorted_eigenvalues = Some(sorted_eigenvalues);
    }

    pub fn transform(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let num_samples = x.len();
        let num_features = x[0].len();
        let mut transformed_x = vec![vec![0.0; self.n_components]; num_samples];

        for i in 0..num_samples {
            let mut centered_x = vec![0.0; num_features];
            for j in 0..num_features {
                centered_x[j] = x[i][j] - self.mean.as_ref().unwrap()[j];
            }

            for j in 0..self.n_components {
                for k in 0..num_features {
                    transformed_x[i][j] += centered_x[k] * self.components.as_ref().unwrap()[j][k];
                }
            }
        }

        transformed_x
    }
}

fn eigen(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut eigenvalues = vec![0.0; n];
    let mut eigenvectors = vec![vec![0.0; n]; n];

    for i in 0..n {
        eigenvectors[i][i] = 1.0;
    }

    for _ in 0..100 {
        let (mut p, mut q) = (0, 1);

        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > a[p][q].abs() {
                    p = i;
                    q = j;
                }
            }
        }

        if a[p][q].abs() < 1e-8 {
            break;
        }

        let theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            -1.0 / (-theta + (1.0 + theta * theta).sqrt())
        };

        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = c * t;

        for i in 0..n {
            let old_pi = a[i][p];
            let old_qi = a[i][q];
            a[i][p] = c * old_pi - s * old_qi;
            a[i][q] = s * old_pi + c * old_qi;

            let old_vi = eigenvectors[i][p];
            let old_wi = eigenvectors[i][q];
            eigenvectors[i][p] = c * old_vi - s * old_wi;
            eigenvectors[i][q] = s * old_vi + c * old_wi;
        }

        for j in 0..n {
            let old_pj = a[p][j];
            let old_qj = a[q][j];
            a[p][j] = c * old_pj - s * old_qj;
            a[q][j] = s * old_pj + c * old_qj;
        }
    }

    for i in 0..n {
        eigenvalues[i] = a[i][i];
    }

    (eigenvalues, eigenvectors)
}
