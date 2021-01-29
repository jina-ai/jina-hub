__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, Iterator, Any

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

    def get_query_handler(self):
        """Get the database handler
        """
        import plyvel
        return plyvel.DB(self.index_abspath, create_if_missing=True)

    def query(self, key: Any, *args, **kwargs) -> Optional[Any]:
        """Find the protobuf chunk/doc using id
        :param key: ``id``
        :return: protobuf chunk or protobuf document
        """
        from google.protobuf.json_format import Parse
        v = self.query_handler.get(bytes(key))
        value = None
        if v is not None:
            value = Parse(v.decode('utf8'), Document())
        return value

    def add(self, keys: Iterator[int], values: Iterator[bytes], *args, **kwargs):
        """Add JSON-friendly serialized documents to the indexer.

        :param keys: document ids
        :param values: JSON-friendly serialized documents
        """
        with self.query_handler.write_batch() as h:
            for k, v in zip(keys, values):
                h.put(bytes(k), v)

    def update(self, keys: Iterator[int], values: Iterator[bytes], *args, **kwargs):
        """Update JSON-friendly serialized documents on the indexer.

        :param keys: document ids to update
        :param values: JSON-friendly serialized documents
        """
        missed = []
        for key in keys:
            if self.query_handler.get(bytes(key)) is None:
                missed.append(key)
        if missed:
            raise KeyError(f'Key(s) {missed} were not found in {self.save_abspath}')

        self.delete(keys)
        self.add(keys, values)
        return

    def delete(self, keys: Iterator[int], *args, **kwargs):
        """Delete JSON-friendly serialized documents from the indexer.

        :param keys: document ids to delete
        """
        with self.query_handler.write_batch() as h:
            for k in keys:
                h.delete(bytes(k))
