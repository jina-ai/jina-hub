# LevelDBIndexer

Key-value indexer wrapper around the LevelDB database for saving and querying protobuf documents. LevelDB is a fast key-value storage library written at Google that provides an ordered mapping from string keys to string values. You can read more about it [here](https://github.com/google/leveldb)

**NOTE** You will need to manually delete the keys in LevelDB if you choose to delete a workspace. 

## Snippets:

Initialise LevelDBIndexer:

`LevelDBIndexer(hostname=host_ip, port=host_port, db=db_int)`

Users can use Pod images in several ways:

**NOTE**: 

- `MODULE_VERSION` is the version of the LevelDBIndexer, in semver format. E.g. `0.0.15`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.2` 

- Flow API
  
  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-indexer', uses='docker://jinahub/pod.indexer.leveldbindexer:MODULE_VERSION-JINA_VERSION')
    ```

- Flow YAML file

  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: leveldb
      uses: indexers/keyvalue/LevelDBIndexer/config.yml
  ```
  
  and then in `leveldb.yml`:

  ```yaml
  !LevelDBIndexer
  with:
    hostname: yourdomain.com
    port: 6379
    db: 0
  ```

- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.indexer.leveldbindexer:MODULE_VERSION-JINA_VERSION
  ```

- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses hub/example/config.yml --port-in 55555 --port-out 55556
  ```

- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.

  ```bash
    docker run --network host docker://jinahub/pod.indexer.leveldbindexer:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```
