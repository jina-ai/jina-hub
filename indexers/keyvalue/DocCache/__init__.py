import hashlib
import os
import pickle
from typing import Tuple, Optional, Dict

from jina import Executor, DocumentArray, requests, Document
from jina.logging import JinaLogger


class _CacheHandler:
    """A handler for loading and serializing the in-memory cache of the DocCache.

    :param path: Path to the file from which to build the actual paths.
    :param logger: Instance of logger.
    """

    def __init__(self, path, logger):
        self.path = path
        try:
            self.id_to_cache_val = pickle.load(open(path + '.ids', 'rb'))
            self.cache_val_to_id = pickle.load(open(path + '.cache', 'rb'))
        except FileNotFoundError as e:
            logger.warning(
                f'File path did not exist : {path}.ids or {path}.cache: {e!r}. Creating new CacheHandler...'
            )
            self.id_to_cache_val = dict()
            self.cache_val_to_id = dict()

    def close(self):
        """Flushes the in-memory cache to pickle files."""
        pickle.dump(self.id_to_cache_val, open(self.path + '.ids', 'wb'))
        pickle.dump(self.cache_val_to_id, open(self.path + '.cache', 'wb'))


default_fields = ('text',)


class DocCache(Executor):
    """A cache Executor

    Checks if a Document has already been indexed.
    If it hasn't, it is kept
    If it has been indexed before, it will be removed from the list of Documents

    NOTE: The Traversal path used in processing the request is always root (`r`)
    """

    def __init__(
            self,
            fields: Optional[Tuple[str]] = None,
            default_traversal_paths: Tuple[str] = ('r',),
            tag: str = 'cache_hit',
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if fields is None:
            fields = default_fields
        self.fields = fields
        self.tag = tag
        self.logger = JinaLogger('DocCache')
        self.cache_handler = _CacheHandler(
            os.path.join(self.metas.workspace, 'cache'), self.logger
        )
        self.default_traversal_paths = default_traversal_paths

    @requests(on='/index')
    def cache(self, docs: DocumentArray, **kwargs):
        """Method to handle the index process for caching"""
        idx_to_remove = []
        for i, d in enumerate(docs):
            cache_value = DocCache.hash_doc(d, self.fields)
            exists = cache_value in self.cache_handler.cache_val_to_id.keys()
            if not exists:
                self.cache_handler.id_to_cache_val[d.id] = cache_value
                self.cache_handler.cache_val_to_id[cache_value] = d.id
            else:
                idx_to_remove.append(i)
        for i in sorted(idx_to_remove, reverse=True):
            del docs[i]

    def close(self) -> None:
        """Make sure to flush to file"""
        self.cache_handler.close()

    @staticmethod
    def hash_doc(doc: Document, fields: Tuple[str]) -> bytes:
        """Calculate hash by which we cache.

        :param doc: the Document
        :param fields: the list of fields
        :return: the hash value of the fields
        """
        values = doc.get_attributes(*fields)
        if not isinstance(values, list):
            values = [values]
        data = ''
        for field, value in zip(fields, values):
            data += f'{field}:{value};'
        digest = hashlib.sha256(bytes(data.encode('utf8'))).digest()
        return digest

    @property
    def size(self):
        return len(self.cache_handler.id_to_cache_val)
