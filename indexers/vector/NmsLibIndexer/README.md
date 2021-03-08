# NmsLibIndexer

Non-Metric Space Library (NMSLIB) is an efficient cross-platform similarity search library and a toolkit for evaluation of similarity search methods. NMSLIB is possibly the first library with a principled support for non-metric space searching.
NmsLibIndexer is an `nmslib` powered vector indexer. For more information, refer to the [nmslib GitHub repo](https://github.com/nmslib/nmslib).

## Snippets:

Initialise NmsLibIndexer:

`NmsLibIndexer(space=space_str, method=meth_str, index_params={'post': 2}, print_progress=prog_bool, num_threads=thr_int  )`
Users can use Pod images in several ways:

**NOTE**: 

- `MODULE_VERSION` is the version of the NmsLibIndexer, in semver format. E.g. `0.0.16`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.7` 

- Flow API

  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-indexer', uses='docker://jinahub/pod.indexer.nmslibindexer:MODULE_VERSION-JINA_VERSION')
    ```
- Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: faiss
      uses: indexers/vector/NmsLibIndexer/config.yml
  ```
  
  and then in `nmslib.yml`:
  ```yaml
  !NmsLibIndexer
  with:
    hostname: yourdomain.com
    port: 6379
    db: 0
  ```
- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.indexer.nmslibindexer:MODULE_VERSION-JINA_VERSION
  ```
- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses indexers/vector/NmsLibIndexer/config.yml --port-in 55555 --port-out 55556
  ```
- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.
  ```bash
    docker run --network host docker://jinahub/pod.indexer.nmslibindexer:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```