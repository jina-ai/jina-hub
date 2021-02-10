from typing import Tuple

import numpy as np
from jina.executors.indexers.vector import BaseNumpyIndexer

if False:
    import SPTAG
class SptagIndexer(BaseNumpyIndexer):
    """
    :class:`SptagIndexer` SPTAG powered vector indexer.

    For SPTAG installation and python API usage, please consult:

        - https://github.com/microsoft/SPTAG/blob/master/Dockerfile
        - https://github.com/microsoft/SPTAG/blob/master/docs/Tutorial.ipynb
        - https://github.com/microsoft/SPTAG
        - https://github.com/microsoft/SPTAG/blob/master/docs/Parameters.md

    .. note::
        sptag package dependency is only required at the query time.
    """

    def __init__(self,
                 method: str = 'BKT',
                 samples: int = 1000,
                 tpt_number: int = 32,
                 tpt_leaf_size: int = 2000,
                 neighborhood_size: int = 32,
                 graph_neighborhood_size: int = 2,
                 cef: int = 100,
                 max_check_for_refined_graph: int = 10000,
                 num_threads: int = 1,
                 dist_calc_method: str = 'Cosine',
                 max_check: int = 8192,
                 bkt_number: int = 1,
                 bkt_meansk: int = 32,
                 kdt_number: int = 1,
                 *args, **kwargs):
        """
        Initialize an SptagIndexer
        :param method: The index method to use, index Algorithm type (e.g. BKT, KDT), required.
        :param samples: how many points will be sampled to do tree node split
        :param tpt_number: number of TPT trees to help with graph construction
        :param tpt_leaf_size: TPT tree leaf size
        :param neighborhood_size: number of neighbors each node has in the neighborhood graph
        :param graph_neighborhood_size: number of neighborhood size scale in the build stage
        :param cef: number of results used to construct RNG
        :param max_check_for_refined_graph: how many nodes each node will visit during graph refine in the build stage
        :param num_threads: The number of threads to use
        :param max_check: how many nodes will be visited for a query in the search stage
        :param dist_calc_method: the distance type, currently SPTAG only support Cosine and L2 distances.
        :param bkt_number: number of BKT trees (only used if method is BKT)
        :param bkt_meansk: how many childs each tree node has (only used if method is BKT)
        :param kdt_number: number of KDT trees (only used if method is BKT)
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.kdt_number = kdt_number
        self.bkt_meansk = bkt_meansk
        self.bkt_number = bkt_number
        self.max_check = max_check
        self.max_check_for_refined_graph = max_check_for_refined_graph
        self.cef = cef
        self.graph_neighborhood_size = graph_neighborhood_size
        self.neighborhood_size = neighborhood_size
        self.tpt_leaf_size = tpt_leaf_size
        self.tpt_number = tpt_number
        self.samples = samples
        self.method = method
        self.space = dist_calc_method
        self.num_threads = num_threads

    def build_advanced_index(self, vecs: 'np.ndarray') -> 'SPTAG.AnnIndex':
        """Build an advanced index structure from a numpy array.

        :param vecs: numpy array containing the vectors to index
        :return: advanced index
        """
        import SPTAG

        _index = SPTAG.AnnIndex(self.method, 'Float', vecs.shape[1])

        # Set the parameters
        _index.SetBuildParam("Samples", str(self.samples))
        _index.SetBuildParam("TPTNumber", str(self.tpt_number))
        _index.SetBuildParam("TPTLeafSize", str(self.tpt_leaf_size))
        _index.SetBuildParam("NeighborhoodSize", str(self.neighborhood_size))
        _index.SetBuildParam("GraphNeighborhoodScale", str(self.graph_neighborhood_size))
        _index.SetBuildParam("CEF", str(self.cef))
        _index.SetBuildParam("MaxCheckForRefineGraph", str(self.max_check_for_refined_graph))
        _index.SetBuildParam("NumberOfThreads", str(self.num_threads))
        _index.SetBuildParam("DistCalcMethod", self.space)
        _index.SetSearchParam("MaxCheck", str(self.max_check))
        _index.SetBuildParam("BKTNumber", str(self.bkt_number))
        _index.SetBuildParam("BKTMeansK", str(self.bkt_meansk))
        _index.SetBuildParam("KDTNumber", str(self.kdt_number))

        if _index.Build(vecs.astype('float32'), vecs.shape[0]):
            return _index

    def query(self, keys: 'np.ndarray', top_k: int, *args, **kwargs) -> Tuple['np.ndarray', 'np.ndarray']:
        """Find the top-k vectors with smallest ``metric`` and return their ids in ascending order.

        :param keys: numpy array containing vectors to search for
        :param top_k: upper limit of responses for each search vector
        :return: document ids and scores
        """
        idx = (np.ones((keys.shape[0], top_k)) * (-1)).astype(str)
        dist = np.ones((keys.shape[0], top_k)) * (-1)
        for r_id, k in enumerate(keys):
            _idx, _dist, _ = self.query_handler.Search(k, top_k)
            idx[r_id, :] = self._int2ext_id[np.array(_idx)]
            dist[r_id, :] = np.array(_dist)
        return idx, dist
