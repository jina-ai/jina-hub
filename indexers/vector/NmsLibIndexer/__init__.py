__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Tuple

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

    def __init__(self, space: str = 'cosinesimil',
                 method: str = 'hnsw',
                 print_progress: bool = False,
                 num_threads: int = 1,
                 *args, **kwargs):
        """
        Initialize an NmslibIndexer

        :param space: The metric space to create for this index
        :param method: The index method to use
        :param num_threads: The number of threads to use
        :param print_progress: Whether or not to display progress bar when creating index
        :param args:
        :param kwargs:
        """
        super().__init__(*args, compress_level=0, **kwargs)
        self.method = method
        self.space = space
        self.print_progress = print_progress
        self.num_threads = num_threads

    def build_advanced_index(self, vecs: 'np.ndarray'):
        import nmslib
        _index = nmslib.init(method=self.method, space=self.space)
        self.build_partial_index(vecs, slice(0, len(vecs)), _index)
        params = {'post': 2} if self.method not in ['brute_force', 'simple_invindx'] else None
        _index.createIndex(index_params=params, print_progress=self.print_progress)
        return _index

    @batching(ordinal_idx_arg=2)
    def build_partial_index(self, vecs: 'np.ndarray', ord_idx: 'slice', _index):
        _index.addDataPointBatch(vecs.astype(np.float32), range(ord_idx.start, ord_idx.stop))

    def query(self, keys: 'np.ndarray', top_k: int, *args, **kwargs) -> Tuple['np.ndarray', 'np.ndarray']:
        ret = self.query_handler.knnQueryBatch(keys, k=top_k, num_threads=self.num_threads)
        idx, dist = zip(*ret)
        return self.int2ext_id[np.array(idx)], np.array(dist)
