# RedisDBIndexer

Key-value indexer wrapper around the Redis database. Redis is an in-memory key-value storage system. You can read more about it here: https://redis.io/

**NOTE** You will need to manually delete the keys in Redis if you choose to delete a workspace. 

## Snippets:

Initialise RedisDBIndexer:

`RedisDBIndexer(hostname=host_ip, port=host_port, db=db_int)`

Users can use Pod images in several ways:

**NOTE**: 

- `MODULE_VERSION` is the version of the RedisDBIndexer, in semver format. E.g. `0.0.6`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.1` 

- Run with Docker (`docker run`)
  
  ```bash
    docker run --network host docker://jinahub/pod.indexer.redisdbindexer:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```

- Flow API
  
  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-indexer', uses='docker://jinahub/pod.indexer.redisdbindexer:MODULE_VERSION-JINA_VERSION')
    ```

- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.indexer.redisdbindexer:MODULE_VERSION-JINA_VERSION
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
  
- YAML file

  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: redis
      uses: docker://jinahub/pod.indexer.redisdbindexer:MODULE_VERSION-JINA_VERSION 
      uses_internal: redis.yml
  ```
  
  and then in `redis.yml`:

  ```yaml
  !RedisDBIndexer
  with:
    hostname: yourdomain.com
    port: 6379
    db: 0
  ```
