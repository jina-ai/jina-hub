# NGTIndexer

Neighborhood Graph and Tree for Indexing High-dimensional Data
NGT provides commands and a library for performing high-speed approximate nearest neighbor searches against a large volume of data (several million to several 10 million items of data) in high dimensional vector data space (several ten to several thousand dimensions).
NGTIndexer is hence an NGT powered vector indexer
For more information, refer to [NGT's GitHub Repo](https://github.com/yahoojapan/NGT).

## Snippets:

Initialise NGTIndexer:

`NGTIndexer(num_threads=thread_int, metric='L1', epsilon=ep_flt)`

**NOTE**: 

- `MODULE_VERSION` is the version of the NGTIndexer, in semver format. E.g. `0.0.11`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.7` 

- Flow API

  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-indexer', uses='docker://jinahub/pod.indexer.ngtindexer:MODULE_VERSION-JINA_VERSION')
    ```
- Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: ngt
      uses: indexers/vector/NGTIndexer/config.yml
  ```
  
  and then in `leveldb.yml`:
  ```yaml
  !NGTIndexer
  with:
    hostname: yourdomain.com
    port: 6379
    db: 0
  ```
- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.indexer.ngtindexer:MODULE_VERSION-JINA_VERSION
  ```
- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses indexers/vector/NGTIndexer/config.yml --port-in 55555 --port-out 55556
  ```
- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.
  ```bash
    docker run --network host docker://jinahub/pod.indexer.ngtindexer:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```