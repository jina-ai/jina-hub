# MongoDBIndexer

Key-value indexer wrapper around the Mongo database. Mongo is a no-SQL storage system. You can read more about it here: https://www.mongodb.com/


**NOTE** You will need to manually delete the entries in the database if you choose to delete a workspace. 

## Snippets:

Initialise MongoDBIndexer:

`MongoDBIndexer(hostname, port, username, password, database, collection)`

Users can use Pod images in several ways:

**NOTE**: 

- `MODULE_VERSION` is the version of the MongoDBIndexer, in semver format. E.g. `0.0.6`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.1` 

- YAML file
  
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: mongodb
      uses: docker://jinahub/pod.indexer.mongodbindexer:MODULE_VERSION-JINA_VERSION 
      uses_internal: mongodb.yml
  ```
  
  and then in `mongodb.yml`:
  ```yaml
  !MongoDBIndexer
  with:
    hostname: host.com
    # key-value arguments go here 
  ```

- Run with Docker (`docker run`)
  
  ```bash
    docker run jinahub/pod.indexer.mongodbindexer:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```

- Flow API
  
  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-indexer', uses='docker://jinahub/pod.indexer.mongodbindexer:MODULE_VERSION-JINA_VERSION', port_in=55555, port_out=55556)
    ```

- Jina CLI
  
  ```bash
  jina pod --uses jinahub/pod.indexer.mongodbindexer:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
  ```

- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses hub/example/config.yml --port-in 55555 --port-out 55556
  ```

- Docker command

  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.

  ```bash
  docker pull jinahub/pod.indexer.mongodbindexer:MODULE_VERSION-JINA_VERSION
  ```