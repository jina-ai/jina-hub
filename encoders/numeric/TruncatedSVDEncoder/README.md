# TruncatedSVDEncoder

`TruncatedSVDEncoder` encodes data from an `ndarray` of size `BatchSize x Dimension` into an `ndarray` of size `BatchSize x EmbeddingDimension` using [Truncated Singular Value Decomposition (SVD)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).  Truncated SVD is a dimension reduction technique which does not center the data before computation which makes it efficient for sparse matrices.

## Usage:

Initialise the encoder using the parameters mentioned below:

| `param name`    | `param_remarks`                              |
| --------------- | ---------------------------------------------|
| `output_dim`    | dimensionality of output embedding           |
| `algorithm`     | algorithm to be used to run SVD              |
| `max_iter`      | number of iterations for the algorithm to run|
