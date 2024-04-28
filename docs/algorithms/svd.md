# Singular Value Decomposition (SVD)

## Explanation:
Singular Value Decomposition (SVD) is a matrix factorization technique. It decomposes a matrix into three separate matrices to reveal the latent features or factors underlying the data. SVD is particularly effective in [reducing the dimensionality](#reducing-the-dimensionality) of [sparse](#sparse-data) or [noisy data](#noisey-data), allowing for efficient computation and improved recommendation accuracy.

SVD provides a factorization of the input matrix A into its constituent parts, enabling a low-rank approximation that captures the most important information within the original matrix.

SVD can be applied to recommender systems by utilizing the decomposed matrices to represent users and items in a latent space. The resulting low-dimensional representations facilitate efficient computations for generating recommendations based on user-item interactions or similarity measures.

By utilizing SVD, recommender systems can identify latent factors that influence user preferences or item characteristics, enabling accurate recommendations even for sparse or incomplete data.

## Formula:
$$ A = U \Sigma V^T $$

### Where:
* A is the input matrix to be decomposed.
* U is an orthogonal matrix that represents the left singular vectors, corresponding to the column space of A.
* Σ is a diagonal matrix that contains the singular values of A. The singular values represent the square roots of the eigenvalues of AA^T or A^TA.
* V^T is the transpose of an orthogonal matrix representing the right singular vectors, corresponding to the row space of A.

## Hints
### Reducing the dimensionality
Reducing the dimensionality of sparse or noisy data refers to the process of representing the data in a lower-dimensional space while retaining the most important information. 

### Sparse data
Sparse data refers to data where many entries are missing or have a value of zero. 

### Noisy data
Noisy data, on the other hand, contains random or irrelevant information that can obscure the underlying patterns or relationships.

## Steps
1.- Obtain the input matrix: Start with an input matrix A that you want to decompose using SVD. The matrix A should typically represent user-item interactions or any other relevant data for the recommender system.

2.- Compute the Singular Value Decomposition: Perform the SVD calculation on the input matrix A. This involves factorizing the matrix into three separate matrices: U, Σ, and V^T.

3.- Calculate the left singular vectors: Compute the matrix U, which represents the left singular vectors. These vectors capture the column space of the input matrix A and provide information about the relationships among users or items.

4.- Determine the singular values: Calculate the diagonal matrix Σ, which contains the singular values of the input matrix A. The singular values represent the square roots of the eigenvalues of AA^T or A^TA and provide information about the importance or significance of each singular vector.

5.- Calculate the right singular vectors: Compute the transpose of the matrix V, denoted as V^T, which represents the right singular vectors. These vectors capture the row space of the input matrix A and provide information about the characteristics or features of the items.

6.- Verify the reconstruction: Verify the reconstruction of the original matrix by multiplying U, Σ, and V^T back together. The result should be close to the original matrix A.