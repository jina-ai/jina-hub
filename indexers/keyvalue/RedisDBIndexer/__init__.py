__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, Iterable

from jina.executors.indexers.keyvalue import BinaryPbIndexer

if False:
    from redis import Redis


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

    def get_query_handler(self) -> 'Redis':
        """Get the database handler.
        """
        import redis
        try:
            r = redis.Redis(host=self.hostname, port=self.port, db=self.db, socket_timeout=10)
            r.ping()
            return r
        except redis.exceptions.ConnectionError as r_con_error:
            self.logger.error('Redis connection error: ', r_con_error)
            raise

    def query(self, key: str, *args, **kwargs) -> Optional[bytes]:
        """Find the protobuf document via id.
        :param key: ``id``
        :return: matching document
        """
        with self.get_query_handler() as redis_handler:
            return redis_handler.get(key)

    def add(self, keys: Iterable[str], values: Iterable[bytes], *args, **kwargs) -> None:
        """Add JSON-friendly serialized documents to the index.

        :param keys: document ids
        :param values: JSON-friendly serialized documents
        """
        redis_docs = [{'_id': i, 'values': j} for i, j in zip(keys, values)]

        with self.get_query_handler() as redis_handler:
            for k in redis_docs:
                redis_handler.set(k['_id'], k['values'])

    def update(self, keys: Iterable[str], values: Iterable[bytes], *args, **kwargs) -> None:
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

    def delete(self, keys: Iterable[str], *args, **kwargs) -> None:
        """Delete documents from the index.

        :param keys: document ids to delete
        """
        with self.get_query_handler() as h:
            for k in keys:
                h.delete(k)
