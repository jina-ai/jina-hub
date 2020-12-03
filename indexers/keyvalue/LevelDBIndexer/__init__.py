__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import json
from typing import Optional, Iterator, Any

from google.protobuf.json_format import Parse
from jina import Document
from jina.executors.indexers.keyvalue import BinaryPbIndexer


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


    def add(self, keys: Iterator[int], values: Iterator[bytes], *args, **kwargs):
        """Add a JSON-friendly object to the indexer

        :param objs: objects can be serialized into JSON format
        """
        with self.write_handler.write_batch() as h:
            for k, v in zip(keys, values):
                key = bytes(k)
                value = json.dumps(v).encode('utf8')
                h.put(key, value)

    def get_query_handler(self):
        """Get the database handler

        """
        import plyvel
        return plyvel.DB(self.index_abspath, create_if_missing=True)

    def query(self, key: Any) -> Optional[Any]:
        """Find the protobuf chunk/doc using id

        :param key: ``id``
        :return: protobuf chunk or protobuf document
        """
        v = self.query_handler.get(bytes(key))
        value = None
        if v is not None:
            value = Parse(json.loads(v.decode('utf8')), Document())
        return value

    # TODO remove this method once https://github.com/jina-ai/jina/pull/1380 is merged
    def update(self, keys: Iterator[int], values: Iterator[bytes], *args, **kwargs):
        missed = []
        for key in keys:
            if self.query_handler.header.get(key) is None:
                missed.append(key)
        if missed:
            # FIXME get indexer name
            raise KeyError(f'Key(s) {missed} were not found in {self.save_abspath}')

        # hack
        self.query_handler.close()
        self.handler_mutex = False
        self.delete(keys)
        self.add(keys, values)
        return

    def delete(self, keys: Iterator[int], *args, **kwargs):
        with self.write_handler.write_batch() as h:
            for k in keys:
                key = bytes(k)
                h.delete(key)
