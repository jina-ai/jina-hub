# ZarrIndexer

ZarrIndexer is based on Zarr - a library that provides classes and functions for working with N-dimensional arrays that behave like NumPy arrays but whose data is divided into chunks and each chunk is compressed. If you are already familiar with HDF5 then Zarr arrays provide similar functionality, but with some additional flexibility.
For more information about Zarr, refer [Zarr documentation](https://zarr.readthedocs.io/en/stable/index.html).

## Snippets:

**NOTE**: 

- `MODULE_VERSION` is the version of the ZarrIndexer, in semver format. E.g. `0.0.13`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.6` 

- Flow API

  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-indexer', uses='docker://jinahub/pod.indexer.zarrindexer:MODULE_VERSION-JINA_VERSION')
    ```
- Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: ngt
      uses: indexers/vector/ZarrIndexer/config.yml
  ```
  
  and then in `zarr.yml`:
  ```yaml
  !ZarrIndexer
  with:
    hostname: yourdomain.com
    port: 6379
    db: 0
  ```
- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.indexer.zarrindexer:MODULE_VERSION-JINA_VERSION
  ```
- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses indexers/vector/ZarrIndexer/config.yml --port-in 55555 --port-out 55556
  ```
- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.
  ```bash
    docker run --network host docker://jinahub/pod.indexer.zarrindexer:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```