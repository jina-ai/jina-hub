from os import path
from typing import List, Union, Optional

import numpy as np

from jina.executors.decorators import as_update_method
from jina.executors.indexers.vector import NumpyIndexer
from jina.helper import cached_property

if False:
    import zarr


class ZarrIndexer(NumpyIndexer):
    """Indexing based on Zarr arrays
    
    For more information about Zarr, please check
    https://zarr.readthedocs.io/en/stable/index.html
    
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_add_handler(self) -> 'zarr.hierarchy.Group':
        """Open an existing zarr group file for adding new vectors

        :return: a zarr group file
        """
        import zarr
        return zarr.open(store=self.index_abspath, mode='a')

    def get_create_handler(self) -> 'zarr.hierarchy.Group':
        """Create a zarr group file for adding new vectors

        :return: a zarr group file
        """
        import zarr
        return zarr.open(store=self.index_abspath, mode='w')

    @as_update_method
    def add(self, keys: 'np.ndarray', vectors: 'np.ndarray', *args, **kwargs) -> None:
        self._validate_key_vector_shapes(keys, vectors)
        if 'default' in self.write_handler.array_keys():
            self.write_handler['default'].append(data=vectors)
        else:
            self.write_handler.array(name='default', data=vectors)
        self.key_bytes += keys.tobytes()
        self.key_dtype = keys.dtype.name
        self._size += keys.shape[0]
        self.valid_indices = np.concatenate((self.valid_indices, np.full(len(keys), True)))
    
    @property
    def query_handler(self):
        return self.get_query_handler()
    
    def get_query_handler(self) -> Optional['zarr.core.Array']:
        import zarr
        if not (path.exists(self.index_abspath) or self.num_dim or self.dtype):
            return
        return zarr.open(store=f'{self.index_abspath}/default', mode='r',
                         shape=(self._size, self.num_dim), chunks=True)
    
    def query_by_id(self, ids: Union[List[int], 'np.ndarray'], *args, **kwargs) -> 'np.ndarray':
        ids = self._filter_nonexistent_keys(ids, self.ext2int_id.keys(), self.save_abspath)
        int_ids = [self.ext2int_id[j] for j in ids]
        return self.raw_ndarray.get_orthogonal_selection(int_ids)
    
    @cached_property
    def raw_ndarray(self):
        return self.query_handler
