__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import json
from typing import Optional

from google.protobuf.json_format import Parse
from jina.executors.indexers.keyvalue import BinaryPbIndexer
from jina.proto import jina_pb2


class LevelDBIndexer(BinaryPbIndexer):
    """
    :class:`LevelDBIndexer` use `LevelDB` to save and query protobuf document.
    """

    def get_add_handler(self):
        """Get the database handler

        """
        import plyvel
        return plyvel.DB(self.index_abspath, create_if_missing=True)

    def get_create_handler(self):
        """Get the database handler

        """
        import plyvel
        return plyvel.DB(self.index_abspath, create_if_missing=True)

    def add(self, objs):
        """Add a JSON-friendly object to the indexer

        :param objs: objects can be serialized into JSON format
        """
        with self.write_handler.write_batch() as h:
            for k, obj in objs.items():
                key = k.encode('utf8')
                value = json.dumps(obj).encode('utf8')
                h.put(key, value)

    def get_query_handler(self):
        """Get the database handler

        """
        import plyvel
        return plyvel.DB(self.index_abspath, create_if_missing=True)

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
