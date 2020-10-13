__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import json
from typing import Optional, Iterator

from jina.executors.indexers.keyvalue import BinaryPbIndexer
from google.protobuf.json_format import Parse
from jina.proto import jina_pb2


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

    def get_add_handler(self):
        """Get the database handler

        """
        import redis
        r = redis.Redis(host=self.hostname, port=self.port, db=self.db, socket_timeout=10)
        try:
            r.ping()
            return r
        except redis.exceptions.ConnectionError as r_con_error:
            self.logger.error('Redis connection error: ', r_con_error)

    def add(self, keys: Iterator[int], values: Iterator[bytes], *args, **kwargs):
        """Add a JSON-friendly object to the indexer

        :param objs: objects can be serialized into JSON format
        """
        redis_docs = [{'_id': i, 'values': j} for i, j in zip(keys, values)]

        with self.get_add_handler() as redis_handler:
            for k in redis_docs:
                redis_handler.set(k['_id'], k['values'])

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

    def query(self, key: int, *args, **kwargs) -> Optional[bytes]:
        """Find the protobuf chunk/doc using id
        :param key: ``id``
        :return: protobuf chunk or protobuf document
        """
        result = []

        with self.get_add_handler() as redis_handler:
            for _key in redis_handler.scan_iter(match=key):
                res = {
                  "key": _key,
                  "values": redis_handler.get(_key),
                }
                result.append(res)

        return result