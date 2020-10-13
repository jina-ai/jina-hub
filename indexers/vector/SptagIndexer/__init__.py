from typing import Tuple

import numpy as np
from jina.executors.indexers.vector import BaseNumpyIndexer


class SptagIndexer(BaseNumpyIndexer):
    """
    :class:`SptagIndexer` SPTAG powered vector indexer.

    For SPTAG installation and python API usage, please consult:

        - https://github.com/microsoft/SPTAG/blob/master/Dockerfile
        - https://github.com/microsoft/SPTAG/blob/master/docs/Tutorial.ipynb
        - https://github.com/microsoft/SPTAG

    .. note::
        sptag package dependency is only required at the query time.
    """

    def __init__(self, dist_calc_method: str = 'Cosine', method: str = 'BKT',
                 num_threads: int = 1,
                 *args, **kwargs):
        """
        Initialize an SptagIndexer

        :param dist_calc_method: the distance type, currently SPTAG only support Cosine and L2 distances.
        :param method: The index method to use, index Algorithm type (e.g. BKT, KDT), required.
        :param num_threads: The number of threads to use
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.method = method
        self.space = dist_calc_method
        self.num_threads = num_threads

    def build_advanced_index(self, vecs: 'np.ndarray'):
        import SPTAG

        _index = SPTAG.AnnIndex(self.method, 'Float', vecs.shape[1])

        # Set the thread number to speed up the build procedure in parallel
        _index.SetBuildParam("NumberOfThreads", str(self.num_threads))
        _index.SetBuildParam("DistCalcMethod", self.space)

        if _index.Build(vecs.astype('float32'), vecs.shape[0]):
            return _index

    def query(self, keys: 'np.ndarray', top_k: int, *args, **kwargs) -> Tuple['np.ndarray', 'np.ndarray']:
        idx = np.ones((keys.shape[0], top_k)) * (-1)
        dist = np.ones((keys.shape[0], top_k)) * (-1)
        for r_id, k in enumerate(keys):
            _idx, _dist, _ = self.query_handler.Search(k, top_k)
            idx[r_id, :] = self.int2ext_id[np.array(_idx)]
            dist[r_id, :] = np.array(_dist)
        return idx, dist
