__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Tuple, Optional, Dict, Any

import numpy as np
from jina.executors.indexers.vector import BaseNumpyIndexer
from jina.executors.decorators import batching


class NmsLibIndexer(BaseNumpyIndexer):

    batch_size = 512

    """nmslib powered vector indexer

    For documentation and explanation of each parameter, please refer to

        - https://nmslib.github.io/nmslib/quickstart.html
        - https://github.com/nmslib/nmslib/blob/master/manual/methods.md

    .. note::
        Nmslib package dependency is only required at the query time.
    """

    def __init__(self,
                 space: str = 'cosinesimil',
                 method: str = 'hnsw',
                 index_params: Optional[Dict[str, Any]] = {'post': 2},
                 print_progress: bool = False,
                 num_threads: int = 1,
                 *args, **kwargs):
        """
        Initialize an NmslibIndexer

        :param space: The metric space to create for this index
        :param method: The index method to use
        :param index_params: Dictionary of optional parameters to use in indexing
        :param num_threads: The number of threads to use
        :param print_progress: Whether or not to display progress bar when creating index
        :param args:
        :param kwargs:
        """
        super().__init__(*args, compress_level=0, **kwargs)
        self.space = space
        self.method = method
        self.index_params = index_params
        self.print_progress = print_progress
        self.num_threads = num_threads

    def build_advanced_index(self, vecs: 'np.ndarray'):
        """Build an advanced index structure from a numpy array.

        :param vecs: numpy array containing the vectors to index
        """
        import nmslib
        _index = nmslib.init(method=self.method, space=self.space)
        self._build_partial_index(vecs, slice(0, len(vecs)), _index)
        _index.createIndex(index_params=self.index_params, print_progress=self.print_progress)
        return _index

    @batching(ordinal_idx_arg=2)
    def _build_partial_index(self, vecs: 'np.ndarray', ord_idx: 'slice', _index):
        _index.addDataPointBatch(vecs.astype(np.float32), range(ord_idx.start, ord_idx.stop))

    def query(self, keys: 'np.ndarray', top_k: int, *args, **kwargs) -> Tuple['np.ndarray', 'np.ndarray']:
        """Find the top-k vectors with smallest ``metric`` and return their ids in ascending order.
        :param keys: numpy array containing vectors to search for
        :param top_k: upper limit of responses for each search vector
        """
        ret = self.query_handler.knnQueryBatch(keys, k=top_k, num_threads=self.num_threads)
        idx, dist = zip(*ret)
        return self._int2ext_id[np.array(idx)], np.array(dist)
