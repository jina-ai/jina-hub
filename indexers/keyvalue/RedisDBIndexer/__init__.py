__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import json

from jina.executors.indexers import BaseIndexer
from google.protobuf.json_format import Parse
from jina.proto import jina_pb2

class RedisDBIndexer(BaseIndexer):
    """
    :class:`RedisDBIndexer` Use Redis as a key-value indexer.
    """

    def get_add_handler(self):
        """Get the database handler

        """
        import redis

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # your customized __init__ below
        raise NotImplementedError

    def query(self, keys, *args, **kwargs):
        raise NotImplementedError

    def add(self, keys, vectors, *args, **kwargs):
        raise NotImplementedError

    
