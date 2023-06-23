## Explanation:
PCA computes the principal components by performing the following steps:
1. Mean centering the data.
2. Calculating the covariance matrix.
3. Finding the eigenvalues and eigenvectors of the covariance matrix.
4. Sorting the eigenvectors based on their corresponding eigenvalues.
5. Selecting the top `n_components` eigenvectors as the principal components.
## Formula:
The formula for PCA is as follows:
$$
X_{\text{transformed}} = X - \bar{X} \cdot V^T
$$
### Where:
* `X_{\text{transformed}}` represents the transformed data matrix.
* `X` is the original data matrix.
* `\bar{X}` represents the mean of each feature.
* `V` represents the matrix of eigenvectors (principal components).