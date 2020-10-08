__copyright__ = "Copyrsight (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import json
from typing import Optional

from jina.executors.indexers.keyvalue import BinaryPbIndexer
from google.protobuf.json_format import Parse
from jina.proto import jina_pb2


class RedisDBIndexer(BinaryPbIndexer):
    """
    :class:`RedisDBIndexer` Use Redis as a key-value indexer.
    """

    def get_add_handler(self):
        """Get the database handler

        """
        import redis
        r = redis.Redis(host='0.0.0.0', port=27017, db=0, password=None, socket_timeout=None)
        try:
            r.ping()
            print('Successfully connected to redis')
            return r
        except redis.exceptions.ConnectionError as r_con_error:
            print('Redis connection error')

    def add(self, objs):
        """Add a JSON-friendly object to the indexer

        :param objs: objects can be serialized into JSON format
        """
        r = self.get_add_handler()
        for k, obj in objs.items():
            key = k.encode('utf8')
            value = json.dumps(obj).encode('utf8')
            r.set(key, value)

    def get_query_handler(self):
        """Get the database handler

        """
        import redis
        r = redis.Redis(host='0.0.0.0', port=27017, db=0, password=None, socket_timeout=None)
        try:
            r.ping()
            print('Successfully connected to redis')
            return r
        except redis.exceptions.ConnectionError as r_con_error:
            print('Redis connection error')

    def query(self, key: str, *args, **kwargs) -> Optional['jina_pb2.Document']:
        """Find the protobuf chunk/doc using id

        :param key: ``id``
        :return: protobuf chunk or protobuf document
        """
        v = self.query_handler.get(key.encode('utf8'))
        value = None
        if v is not None:
            value = Parse(json.loads(v.decode('utf8')), jina_pb2.Document())
        return value
