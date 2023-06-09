## Explanation:
Non-Negative Matrix Factorization (NMF) factorizes a given matrix `V` into two non-negative matrices `W` and `H`.
It aims to find `W` and `H` such that their product approximates `V`:
$$
V ≈ W * H
$$
The factorized matrix `W` consists of basis vectors, and `H` contains coefficients for these basis vectors.
## Formula:
NMF aims to minimize the Frobenius norm of the difference between `V` and `W * H` by updating `W` and `H` iteratively.
The update equations for `W` and `H` can be expressed as follows:
$$
W_{ij} = W_{ij} * ((V * H^T) / (W * H * H^T))
H_{ij} = H_{ij} * ((W^T * V) / (W^T * W * H))
$$
Where:
* `V` is the input matrix.
* `W` is the factorized matrix of size `(m, n_components)`.
* `H` is the factorized matrix of size `(n_components, n)`.
* `m` is the number of rows in `V`.
* `n` is the number of columns in `V`.
* `n_components` is the number of components (columns) in `W` and `H`.
* `^T` denotes matrix transpose.