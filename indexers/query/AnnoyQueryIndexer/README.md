# AnnoyQueryIndexer

Annoy (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings to search for points in space that are close to a given query point.
AnnoyQueryIndexer is thus an Annoy-powered vector indexer.
It differs from the AnnoyIndexer, as it is a write-once indexer, which receives a `dump_path` in its `metas`. It loads its data from this path.
For more information, refer to the GitHub repo for [Spotify's Annoy](https://github.com/spotify/annoy).

## Snippets:

Initialise AnnoyQueryIndexer:

`AnnoyQueryIndexer(metric='euclidean', n_trees=trees_int, search_k=search_int)`

Users can use Pod images in several ways:

**NOTE**: 

- `MODULE_VERSION` is the version of the AnnoyQueryIndexer, in semver format. E.g. `0.0.16`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.2` 

- Flow API
  
  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-indexer', uses='docker://jinahub/pod.indexer.annoyqueryindexer:MODULE_VERSION-JINA_VERSION')
    ```

- Flow YAML file

  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: annoy
      uses: indexers/vector/AnnoyQueryIndexer/config.yml
  ```
  
  and then in `leveldb.yml`:

  ```yaml
  !AnnoyQueryIndexer
  with:
    hostname: yourdomain.com
    port: 6379
    db: 0
  ```

- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.indexer.annoyqueryindexer:MODULE_VERSION-JINA_VERSION
  ```

- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses hub/example/config.yml --port-in 55555 --port-out 55556
  ```

- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.

  ```bash
    docker run --network host docker://jinahub/pod.indexer.annoyqueryindexer:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```
