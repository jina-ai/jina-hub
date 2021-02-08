from os import path
from typing import Optional, Iterable

import numpy as np

from jina.executors.decorators import as_update_method
from jina.executors.indexers.vector import NumpyIndexer
from jina.helper import cached_property

if False:
    import zarr


class ZarrIndexer(NumpyIndexer):
    """
    Indexing based on Zarr arrays.

    For more information about Zarr, please check
    https://zarr.readthedocs.io/en/stable/index.html

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_add_handler(self) -> 'zarr.hierarchy.Group':
        """
        Open an existing zarr group file for adding new vectors.

        :return: a zarr group file
        """
        import zarr
        return zarr.open(store=self.index_abspath, mode='a')

    def get_create_handler(self) -> 'zarr.hierarchy.Group':
        """
        Create a zarr group file for adding new vectors.

        :return: a zarr group file
        """
        import zarr
        return zarr.open(store=self.index_abspath, mode='w')

    @as_update_method
    def add(self, keys: Iterable[str], vectors: 'np.ndarray', *args, **kwargs) -> None:
        """
        Add the embeddings and document ids to the index.

        :param keys: a list of ``id``, i.e. ``doc.id`` in protobuf
        :param vectors: embeddings
        """
        np_keys = np.array(keys, (np.str_, self.key_length))

        self._add(np_keys, vectors)

    def _add(self, keys: 'np.ndarray', vectors: 'np.ndarray'):
        self._validate_key_vector_shapes(keys, vectors)
        if 'default' in self.write_handler.array_keys():
            self.write_handler['default'].append(data=vectors)
        else:
            self.write_handler.array(name='default', data=vectors)
        self.valid_indices = np.concatenate((self.valid_indices, np.full(len(keys), True)))
        self.key_bytes += keys.tobytes()
        self._size += keys.shape[0]

    @property
    def query_handler(self) -> Optional['zarr.core.Array']:
        """
        Get zarr file handler.

        :return: Zarr query handler
        """
        return self.get_query_handler()

    def get_query_handler(self) -> Optional['zarr.core.Array']:
        """
        Get zarr file handler.

        :return: Zarr query handler
        """
        import zarr
        if not (path.exists(self.index_abspath) or self.num_dim or self.dtype):
            return
        return zarr.open(store=f'{self.index_abspath}/default', mode='r',
                         shape=(self._size, self.num_dim), chunks=True)

    def query_by_key(self, keys: Iterable[str], *args, **kwargs) -> 'np.ndarray':
        """
        Get the vectors by keys.

        :param keys: list of document keys` as 1D-ndarray
        :return: subset of indexed vectors
        """
        filtered_keys = self._filter_nonexistent_keys(keys, self._ext2int_id.keys())
        int_ids = [self._ext2int_id[j] for j in filtered_keys]
        return self._raw_ndarray.get_orthogonal_selection(int_ids)

    @cached_property
    def _raw_ndarray(self):
        a = self.query_handler
        return a
