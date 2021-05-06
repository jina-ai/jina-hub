import mmap

from typing import Optional, Dict
import os
from jina import Executor, requests, DocumentArray, Document
from jina.logging import JinaLogger

from jina.executors.indexers.dump import import_metas
from jina.helper import get_readable_size
from .binarypb import BinaryPbWriterMixin
from jina.executors.helper import physical_size

HEADER_NONE_ENTRY = (-1, -1, -1)


class BinaryPbQueryIndexer(Executor, BinaryPbWriterMixin):
    """
    A DBMS Indexer (no query method)

    :param index_filename: the name of the file for storing the index, when not given metas.name is used.
    :param key_length: the default minimum length of the key, will be expanded one time on the first batch
    :param args:  Additional positional arguments which are just used for the parent initialization
    :param kwargs: Additional keyword arguments which are just used for the parent initialization
    """

    def __init__(
        self,
        dump_path: str,
        index_filename: Optional[str] = None,
        key_length: int = 36,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.index_filename = index_filename or self.metas.name

        self.key_length = key_length
        self._size = 0

        self._start = 0
        self._page_size = mmap.ALLOCATIONGRANULARITY
        self.logger = JinaLogger(self.__class__.__name__)

        self._load_dump(dump_path)
        self.query_handler = self.get_query_handler()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def size(self) -> int:
        """
        The number of vectors or documents indexed.

        :return: size
        """
        return self._size

    def close(self):
        """Close all file-handlers and release all resources. """
        self.logger.info(
            f'indexer size: {self.size} physical size: {get_readable_size(physical_size(self.workspace))}'
        )
        self.query_handler.close()
        super().close()

    @property
    def index_abspath(self) -> str:
        """
        Get the file path of the index storage

        :return: absolute path
        """
        os.makedirs(self.workspace, exist_ok=True)
        return os.path.join(self.workspace, self.index_filename)

    def _load_dump(self, dump_path):
        """Load the dump at the path

        :param dump_path: the path of the dump"""
        ids, metas = import_metas(dump_path, str(self.metas.pea_id))
        with self.get_create_handler() as write_handler:
            self._add(list(ids), list(metas), write_handler)

    @requests(on='/search')
    def search(
        self, docs: DocumentArray, parameters: Dict = None, *args, **kwargs
    ) -> None:
        """Get a document by its id

        :param keys: the ids
        :param args: not used
        :param kwargs: not used
        :return: List of the bytes of the Documents (or None, if not found)
        """
        if parameters is None:
            parameters = {}
        to_be_searched_docs = docs.traverse_flatten(
            parameters.get('traversal_paths', ['r'])
        )
        self._search(to_be_searched_docs, parameters.get('is_update', True))

    def _search(self, docs, is_update):
        miss_idx = (
            []
        )  #: missed hit results, some search may not end with results. especially in shards

        serialized_docs = self._query([d.id for d in docs])

        for idx, (retrieved_doc, serialized_doc) in enumerate(
            zip(docs, serialized_docs)
        ):
            if serialized_doc:
                r = Document(serialized_doc)
                if is_update:
                    retrieved_doc.update(r)
                else:
                    retrieved_doc.CopyFrom(r)
            else:
                miss_idx.append(idx)

        # delete non-existed matches in reverse
        for j in reversed(miss_idx):
            del docs[j]
