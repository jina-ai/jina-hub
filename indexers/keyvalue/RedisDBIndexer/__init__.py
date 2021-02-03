__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, Iterator

from jina.executors.indexers.keyvalue import BinaryPbIndexer


class RedisDBIndexer(BinaryPbIndexer):
    """
    :class:`RedisDBIndexer` Use Redis as a key-value indexer.
    """

    def __init__(self,
                 hostname: str = '0.0.0.0',
                 # default port on linux
                 port: int = 6379,
                 db: int = 0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hostname = hostname
        self.port = port
        self.db = db

    def get_query_handler(self):
        """Get the database handler.
        """
        import redis
        r = redis.Redis(host=self.hostname, port=self.port, db=self.db, socket_timeout=10)
        try:
            r.ping()
            return r
        except redis.exceptions.ConnectionError as r_con_error:
            self.logger.error('Redis connection error: ', r_con_error)

    # TODO unify result interface
    def query(self, key: int, *args, **kwargs) -> Optional[bytes]:
        """Find the protobuf document via id.
        :param key: ``id``
        :return: matching document
        """
        results = []
        with self.get_query_handler() as redis_handler:
            for _key in redis_handler.scan_iter(match=key):
                res = {
                    "key": _key,
                    "values": redis_handler.get(_key),
                }
                results.append(res)
        if len(results) == 0:
            self.logger.warning(f'No matches for key {key} in {self.index_filename}')
            return None

        if len(results) > 1:
            self.logger.warning(
                f'More than 1 element retrieved from Redis with matching key {key}. Will return first...')
        return results[0]['values']

    def add(self, keys: Iterator[int], values: Iterator[bytes], *args, **kwargs):
        """Add JSON-friendly serialized documents to the index.

        :param keys: document ids
        :param values: JSON-friendly serialized documents
        """
        redis_docs = [{'_id': i, 'values': j} for i, j in zip(keys, values)]

        with self.get_query_handler() as redis_handler:
            for k in redis_docs:
                redis_handler.set(k['_id'], k['values'])

    def update(self, keys: Iterator[int], values: Iterator[bytes], *args, **kwargs):
        """Update JSON-friendly serialized documents on the index.

        :param keys: document ids to update
        :param values: JSON-friendly serialized documents
        """
        missed = []
        for key in keys:
            if self.query(key) is None:
                missed.append(key)
        if missed:
            raise KeyError(f'Key(s) {missed} were not found in redis')

        self.delete(keys)
        self.add(keys, values)

    def delete(self, keys: Iterator[int], *args, **kwargs):
        """Delete documents from the index.

        :param keys: document ids to delete
        """
        with self.get_query_handler() as h:
            for k in keys:
                h.delete(k)
