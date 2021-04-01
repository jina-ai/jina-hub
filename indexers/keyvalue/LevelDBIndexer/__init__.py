__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, Iterable

from jina import Document
from jina.executors.indexers.keyvalue import BinaryPbIndexer

if False:
    from plyvel import DB


class LevelDBIndexer(BinaryPbIndexer):
    """
    :class:`LevelDBIndexer` use `LevelDB` to save and query protobuf document.
    """

    def get_add_handler(self) -> 'DB':
        """Get the database handler."""
        import plyvel
        return plyvel.DB(self.index_abspath, create_if_missing=True)

    def get_create_handler(self) -> 'DB':
        """Get the database handler."""
        import plyvel
        return plyvel.DB(self.index_abspath, create_if_missing=True)

    def get_query_handler(self) -> 'DB':
        """Get the database handler."""
        import plyvel
        return plyvel.DB(self.index_abspath, create_if_missing=True)

    def query(self, key: str, *args, **kwargs) -> Optional[bytes]:
        """Find the serialized protobuf documents via id.

        :param key: ``id``
        :return: serialized document
        """
        from google.protobuf.json_format import Parse
        v = self.query_handler.get(bytes(key))
        value = None
        if v is not None:
            value = Parse(v.decode('utf8'), Document())
        return value

    def add(self, keys: Iterable[str], values: Iterable[bytes], *args, **kwargs) -> None:
        """Add JSON-friendly serialized documents to the index.

        :param keys: document ids
        :param values: serialized documents
        """
        with self.query_handler.write_batch() as h:
            for k, v in zip(keys, values):
                h.put(bytes(k), v)

    def update(self, keys: Iterable[str], values: Iterable[bytes], *args, **kwargs) -> None:
        """Update serialized documents on the index.

        :param keys: document ids to update
        :param values: serialized documents
        """
        missed = []
        for key in keys:
            if self.query_handler.get(bytes(key)) is None:
                missed.append(key)
        if missed:
            raise KeyError(f'Key(s) {missed} were not found in {self.save_abspath}')

        self.add(keys, values)

    def delete(self, keys: Iterable[str], *args, **kwargs) -> None:
        """Delete documents from the index.

        :param keys: document ids to delete
        """
        with self.query_handler.write_batch() as h:
            for k in keys:
                h.delete(bytes(k))
