# TruncatedSVDEncoder

`TruncatedSVDEncoder` encodes data from an `ndarray` of size `BatchSize x Dimension` into an `ndarray` of size `BatchSize x EmbeddingDimension` using [Truncated Singular Value Decomposition (SVD)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).

Truncated SVD is a dimensionality reduction technique which does not center the data before computation. This makes them useful on large input sparse data which cannot be centered without constraining the memory.


## Initialization:

Initialise the encoder using the parameters mentioned below:

| `param name`    | `param_remarks`                                           |
| --------------- | ----------------------------------------------------------|
| `output_dim`    | dimensionality of output embedding                        |
| `algorithm`     | algorithm to be used to run SVD (`arpack` or `randomized`)|
| `max_iter`      | number of iterations for the algorithm to run             |

## Usage

Users can use Pod images in several ways:

1. Run with docker:

```
docker run jinahub/pod.encoder.truncatedsvdencoder:MODULE_VERSION-JINA_VERSION
```

2. Run the Flow API:

```
 from jina.flow import Flow

 f = (Flow()
     .add(name='truncated_svd_encoder', uses='docker://jinahub/pod.encoder.truncatedsvdencoder:MODULE_VERSION-JINA_VERSION', port_in=55555, port_out=55556))
```

3. Run with Jina CLI

```
 jina pod --uses docker://jinahub/pod.encoder.truncatedsvdencoder:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
```

4. Conventional local usage with `uses` argument

```
jina pod --uses custom_folder/truncatedsvdencoder.yml --port-in 55555 --port-out 55556
```

**NOTE**:

- `MODULE_VERSION` is the version of the TruncatedSVDEncoder, in semver format. E.g. `0.0.1` and can be found in the `manifest.yml` file.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.1.2`
