__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Tuple

import numpy as np
from jina.executors.indexers.vector import BaseNumpyIndexer


class NGTIndexer(BaseNumpyIndexer):
    """
    NGT powered vector indexer

    For more information about the NGT supported parameters and installation problems, please consult:
        - https://github.com/yahoojapan/NGT

    .. note::
        NGT package dependency is only required at the query time.
        Quick Install : pip install ngt
    """

    def __init__(self, num_threads: int = 2, metric: str = 'L2', epsilon: float = 0.1, *args, **kwargs):
        """
        Initialize an NGT Indexer.

        :param num_threads: Number of threads to build index
        :param metric: Should be one of {L1,L2,Hamming,Jaccard,Angle,Normalized Angle,Cosine,Normalized Cosine}
        :param epsilon: Toggle this variable for speed vs recall tradeoff.
                        Higher value of epsilon means higher recall
                        but query time will increase with epsilon
        """

        super().__init__(*args, **kwargs)
        self._metric = metric
        self._num_threads = num_threads
        self._epsilon = epsilon

    def post_init(self):
        """Setup workspace."""

        super().post_init()
        self.index_path = self.get_file_from_workspace('index')

    def build_advanced_index(self, vecs: 'np.ndarray'):
        """
        Build an advanced index structure from a numpy array.

        :param vecs: numpy array containing the vectors to index
        :return: advanced NGT index
        """

        import ngtpy
        ngtpy.create(path=self.index_path, dimension=self.num_dim, distance_type=self._metric)
        _index = ngtpy.Index(self.index_path)
        _index.batch_insert(vecs, num_threads=self._num_threads)
        return _index

    def query(self, keys: 'np.ndarray', top_k: int, *args, **kwargs) -> Tuple['np.ndarray', 'np.ndarray']:
        """
        Find the top-k vectors with smallest ``metric`` and return their ids in ascending order.

        :param keys: numpy array containing vectors to search for
        :param top_k: upper limit of responses for each search vector
        :return: best matching indices and distance values in two separate `np.array`
        """

        dist = []
        idx = []
        for key in keys:
            results = self.query_handler.search(key, size=top_k, epsilon=self._epsilon)
            if results:
                index_k, distance_k = zip(*results)
                idx.append(index_k)
                dist.append(distance_k)

        return self._int2ext_id[np.array(idx)], np.array(dist)
