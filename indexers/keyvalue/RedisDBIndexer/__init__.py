__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, Iterator, Any

from jina.executors.indexers.keyvalue import BinaryPbIndexer


class RedisDBIndexer(BinaryPbIndexer):
    """
    :class:`RedisDBIndexer` Use Redis as a key-value indexer.
    """

    def __init__(self,
                 hostname: str = '0.0.0.0',
                 port: int = 63079,
                 db: int = 0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hostname = hostname
        self.port = port
        self.db = db

    def get_query_handler(self):
        """Get the database handler
        """
        import redis
        r = redis.Redis(host=self.hostname, port=self.port, db=self.db, socket_timeout=10)
        try:
            r.ping()
            return r
        except redis.exceptions.ConnectionError as r_con_error:
            self.logger.error('Redis connection error: ', r_con_error)

    def query(self, key: int, *args, **kwargs) -> Optional[Any]:
        """Find the protobuf chunk/doc using id
        :param key: ``id``
        :return: protobuf chunk or protobuf document
        """
        result = []
        with self.get_query_handler() as redis_handler:
            for _key in redis_handler.scan_iter(match=key):
                res = {
                    "key": _key,
                    "values": redis_handler.get(_key),
                }
                result.append(res)
        return result

    def add(self, keys: Iterator[int], values: Iterator[bytes], *args, **kwargs):
        """Add a JSON-friendly object to the indexer
        :param keys: keys to be added
        :param values: values to be added
        """
        redis_docs = [{'_id': i, 'values': j} for i, j in zip(keys, values)]

        with self.get_query_handler() as redis_handler:
            for k in redis_docs:
                redis_handler.set(k['_id'], k['values'])

    def update(self, keys: Iterator[int], values: Iterator[bytes], *args, **kwargs):
        """updates the keys if they exist
        """
        missed = []
        for key in keys:
            if len(self.query(key)) == 0:
                missed.append(key)
        if missed:
            raise KeyError(f'Key(s) {missed} were not found in redis')

        self.delete(keys)
        self.add(keys, values)

    def delete(self, keys: Iterator[int], *args, **kwargs):
        """deletes the keys in redis
        """
        with self.get_query_handler() as h:
            for k in keys:
                h.delete(k)
