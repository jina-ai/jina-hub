# TruncatedSVDEncoder

`TruncatedSVDEncoder` encodes data from an `ndarray` of size `BatchSize x Dimension` into an `ndarray` of size `BatchSize x EmbeddingDimension` using [Truncated Singular Value Decomposition (SVD)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).

Truncated SVD is a dimensionality reduction technique which does not center the data before computation. This makes them useful on large input sparse data which cannot be centered without constraining the memory.


## Usage:

Initialise the encoder using the parameters mentioned below:

| `param name`    | `param_remarks`                                           |
| --------------- | ----------------------------------------------------------|
| `output_dim`    | dimensionality of output embedding                        |
| `algorithm`     | algorithm to be used to run SVD (`arpack` or `randomized`)|
| `max_iter`      | number of iterations for the algorithm to run             |
