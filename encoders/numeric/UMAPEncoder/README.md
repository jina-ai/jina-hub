# UMAPEncoder

Encodes data from an ndarray of size `B x T` into an ndarray of size `B x D` using _Uniform Manifold Approximation and Projection (UMAP)_.
UMAP is a dimension reduction technique that can be used for visualisation similarly to t-SNE, but also for general non-linear dimension reduction.

For more information, refer to [UMAP](https://github.com/lmcinnes/umap) documentation.

## Snippets:

Initialize UMAPEncoder:

| `param name`    | `param_remarks`                         |
| --------------- | --------------------------------------- |
| `model_path`    |path of pickled UMAP model to load       |
| `output_dim`    |dimensionality of encoder output         |
| `random_state`  |seed used by the random number generator |


**NOTE**:

- `MODULE_VERSION` is the version of the UMAPEncoder, in semver format. E.g. `0.0.13`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.1` 

- Flow API

  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-encoder', uses='docker://jinahub/pod.encoder.umapencoder:MODULE_VERSION-JINA_VERSION')
    ```
- Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: reducer
      uses: encoders/numeric/UMAPEncoder/config.yml
  ```
  
  and then in `umapencoder.yml`:
  ```yaml
  !UMAPEncoder
  metas:
    - py_modules:
        - __init__.py
  ```
- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.encoder.umapencoder:MODULE_VERSION-JINA_VERSION
  ```
- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses encoders/numeric/UMAPEncoder/config.yml --port-in 55555 --port-out 55556
  ```
- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.
  ```bash
    docker run --network host docker://jinahub/pod.encoder.umapencoder:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```