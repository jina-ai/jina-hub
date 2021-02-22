# RedisDBIndexer

Key-value indexer wrapper around the Redis database. Redis is an in-memory key-value storage system. You can read more about it here: https://redis.io/

**NOTE** You will need to manually delete the keys in Redis if you choose to delete a workspace. 

## Snippets:

Initialise RedisDBIndexer:

`RedisDBIndexer(hostname=host_ip, port=host_port, db=db_int)`

Users can use Pod images in several ways:

- Run with Docker (`docker run`)
  
  ```bash
    docker run jinahub/pod.indexer.redisdbindexer:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```

- Flow API
  
  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-indexer', uses='docker://jinahub/pod.indexer.redisdbindexer:MODULE_VERSION-JINA_VERSION', port_in=55555, port_out=55556)
    ```

- Jina CLI
  
  ```bash
  jina pod --uses jinahub/pod.indexer.redisdbindexer:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
  ```

- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses hub/example/config.yml --port-in 55555 --port-out 55556
  ```

- Docker command

  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.

  ```bash
  docker pull jinahub/pod.indexer.redisdbindexer:MODULE_VERSION-JINA_VERSION
  ```