# FaissIndexer

Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python/numpy. Some of the most useful algorithms are implemented on the GPU. It is developed by Facebook AI Research.
FaissIndexer is hence a Faiss powered vector indexer. For more information, refer [Facebook Research's Faiss](https://github.com/facebookresearch/faiss) GitHub repo.

## Snippets:

Initialise FaissIndexer:

`FaissIndexer(index_key=key_str, train_filepath=fpath_str, max_num_training_points=train_int, requires_training=train_bool, distance='l2', normalize=norm_bool, nprobe=probe_int  )`

Users can use Pod images in several ways:

**NOTE**: 

- `MODULE_VERSION` is the version of the FaissIndexer, in semver format. E.g. `0.0.15`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.0` 

- Flow API
  
  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-indexer', uses='docker://jinahub/pod.indexer.faissindexer:MODULE_VERSION-JINA_VERSION')
    ```

- Flow YAML file

  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: faiss
      uses: indexers/vector/FaissIndexer/config.yml
  ```
  
  and then in `leveldb.yml`:

  ```yaml
  !FaissIndexer
  with:
    hostname: yourdomain.com
    port: 6379
    db: 0
  ```

- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.indexer.faissindexer:MODULE_VERSION-JINA_VERSION
  ```

- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses indexers/vector/FaissIndexer/config.yml --port-in 55555 --port-out 55556
  ```

- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.

  ```bash
    docker run --network host docker://jinahub/pod.indexer.faissindexer:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```
