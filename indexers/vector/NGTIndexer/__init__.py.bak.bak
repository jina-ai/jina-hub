__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Tuple

import numpy as np
from jina.executors.indexers.vector import BaseNumpyIndexer


class NGTIndexer(BaseNumpyIndexer):
    """NGT powered vector indexer

    For more information about the NGT supported parameters and installation problems, please consult:
        - https://github.com/yahoojapan/NGT

    .. note::
        NGT package dependency is only required at the query time.
        Quick Install : pip install ngt
    """

    def __init__(self, num_threads: int = 2, metric: str = 'L2', epsilon: float = 0.1, *args, **kwargs):
        """
        Initialize an NGT Indexer
        :param num_threads: Number of threads to build index
        :param metric: Should be one of {L1,L2,Hamming,Jaccard,Angle,Normalized Angle,Cosine,Normalized Cosine}
        :param epsilon: Toggle this variable for speed vs recall tradeoff.
                        Higher value of epsilon means higher recall
                        but query time will increase with epsilon
        """

        super().__init__(*args, **kwargs)
        self.metric = metric
        self.num_threads = num_threads
        self.epsilon = epsilon

    def post_init(self):
        super().post_init()
        self.index_path = os.path.join(self.workspace, 'index')

    def build_advanced_index(self, vecs: 'np.ndarray'):
        import ngtpy
        ngtpy.create(path=self.index_path, dimension=self.num_dim, distance_type=self.metric)
        _index = ngtpy.Index(self.index_path)
        _index.batch_insert(vecs, num_threads=1)
        return _index

    def query(self, keys: 'np.ndarray', top_k: int, *args, **kwargs) -> Tuple['np.ndarray', 'np.ndarray']:
        dist = []
        idx = []
        for key in keys:
            results = self.query_handler.search(key, size=top_k, epsilon=self.epsilon)
            if results:
                index_k, distance_k = zip(*results)
                idx.append(index_k)
                dist.append(distance_k)

        return self.int2ext_id[np.array(idx)], np.array(dist)
