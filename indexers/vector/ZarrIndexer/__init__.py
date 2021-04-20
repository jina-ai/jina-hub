__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from os import path
from functools import wraps
from typing import Optional, Iterable, Tuple, Callable
from functools import lru_cache

import numpy as np
import copy

from jina.executors.decorators import as_update_method
from jina.executors.indexers.vector import NumpyIndexer, _ext_B, _euclidean, _norm, _cosine
from jina.helper import cached_property

if False:
    import zarr


def zarr_batching(
        func: Optional[Callable] = None,
        batch_size: Optional[int] = None,
):
    """Split the input of a function into small batches and call :func:`func` on each batch
    , collect the merged result and return. This is useful when the input is too big to fit into memory

    :param func: function to decorate
    :param batch_size: size of each batch
    :return: the merged result as if run :func:`func` once on the input.
    """

    def _zarr_batching(func):

        def batch_iterator(data, b_size):
            _l = data.shape[0]
            _d = data.ndim
            sl = [slice(None)] * _d
            for start in range(0, _l, b_size):
                end = min(_l, start + b_size)
                sl[0] = slice(start, end)
                _data_args = data[tuple(sl)]
                yield _data_args

        @wraps(func)
        def arg_wrapper(*args, **kwargs):
            # priority: decorator > class_attribute
            # by default data is in args[1] (self needs to be taken into account)
            data = args[2]
            b_size = batch_size or getattr(args[0], 'batch_size', None)

            # no batching if b_size is None
            if b_size is None or data is None:
                return func(*args, **kwargs)

            batch_args = list(copy.copy(args))

            results = []
            for _data_args in batch_iterator(data, b_size):
                batch_args[2] = _data_args
                r = func(*batch_args, **kwargs)
                if r is not None:
                    results.append(r)

            final_result = np.concatenate(results, 1)

            return final_result

        return arg_wrapper

    if func:
        return _zarr_batching(func)
    else:
        return _zarr_batching


class ZarrIndexer(NumpyIndexer):
    """
    Indexing based on Zarr arrays.

    For more information about Zarr, please check
    https://zarr.readthedocs.io/en/stable/index.html

    """

    @zarr_batching
    def _euclidean(self, cached_A, raw_B):
        data = _ext_B(raw_B)
        return _euclidean(cached_A, data)

    @zarr_batching
    def _cosine(self, cached_A, raw_B):
        data = _ext_B(_norm(raw_B))
        return _cosine(cached_A, data)

    @zarr_batching
    def _cdist(self, *args, **kwargs):
        with ImportExtensions(required=True):
            from scipy.spatial.distance import cdist
        return cdist(*args, **kwargs, metric=self.metric)

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
