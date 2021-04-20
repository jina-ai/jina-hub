# UMAPEncoder

`UMAPEncoder` encodes ``Document`` content from an `ndarray` of size `BatchSize x Dimension` into an `ndarray` of size `BatchSize x EmbeddingDimension` using [Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426).  UMAP is a dimension reduction technique that can be used for visualisation similarly to t-SNE, but also for general non-linear dimension reduction.

For more information, refer to [UMAP](https://github.com/lmcinnes/umap) documentation.

## Usage:

Initialise this Executor specifying parameters i.e.:

| `param name`    | `param_remarks`                          |
| --------------- | ---------------------------------------- |
| `model_path`    | path of pickled UMAP model to load       |
| `output_dim`    | dimensionality of encoder output         |
| `random_state`  | seed used by the random number generator |

### Snippets

Initialise UMAPEncoder:

`UMAPEncoder(model_path='pre-trained.model')`

Users can use Pod images in several ways:

- Run with Docker (`docker run`)
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION` and UMAPEncoder version as `MODULE_VERSION`.
  - ```bash
    docker run --network host docker://jinahub/pod.encoder.umapencoder:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```

- Flow API
  - ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='umap_encoder', uses='docker://jinahub/pod.encoder.umapencoder:MODULE_VERSION-JINA_VERSION')
    ```

- Jina CLI
  - ```bash
    jina pod --uses docker://jinahub/pod.encoder.umapencoder:MODULE_VERSION-JINA_VERSION
    ```

- Conventional local usage with `uses` argument 
  - ```bash
    jina pod --uses hub/example/config.yml --port-in 55555 --port-out 55556
    ```

**NOTE**:

- `MODULE_VERSION` is the version of the UMAPEncoder, in semver format. E.g. `0.0.13`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.10`
