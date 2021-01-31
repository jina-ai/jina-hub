__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Tuple

import numpy as np
from jina.executors.indexers.vector import BaseNumpyIndexer


class AnnoyIndexer(BaseNumpyIndexer):
    """Annoy powered vector indexer

    For more information about the Annoy supported parameters, please consult:
        - https://github.com/spotify/annoy

    .. note::
        Annoy package dependency is only required at the query time.
    """

    def __init__(self, metric: str = 'euclidean', n_trees: int = 10, search_k: int = -1, *args, **kwargs):
        """
        Initialize an AnnoyIndexer

        :param metric: Metric can be "angular", "euclidean", "manhattan", "hamming", or "dot"
        :param n_trees: builds a forest of n_trees trees. More trees gives higher precision when querying.
        :param search_k: At query time annoy will inspect up to search_k nodes which defaults to
            n_trees * k if not provided (set to -1)
        :param args:
        :param kwargs:
        """
        super().__init__(*args, compress_level=0, **kwargs)
        self.metric = metric
        self.n_trees = n_trees
        self.search_k = search_k

    def build_advanced_index(self, vecs: 'np.ndarray'):
        """Build an advanced index structure from a numpy array.

        :param vecs: numpy array containing the vectors to index
        """
        from annoy import AnnoyIndex
        _index = AnnoyIndex(self.num_dim, self.metric)
        for idx, v in enumerate(vecs):
            _index.add_item(idx, v.astype(np.float32))
        _index.build(self.n_trees)
        return _index

    def query(self, keys: 'np.ndarray', top_k: int, *args, **kwargs) -> Tuple['np.ndarray', 'np.ndarray']:
        """Find the top-k vectors with smallest ``metric`` and return their ids in ascending order.
        :param keys: numpy array containing vectors to search for
        :param top_k: upper limit of responses for each search vector
        """
        all_idx = []
        all_dist = []
        for k in keys:
            idx, dist = self.query_handler.get_nns_by_vector(k, top_k, self.search_k, include_distances=True)
            all_idx.append(self.int2ext_id[self.valid_indices][idx])
            all_dist.append(dist)
        return np.array(all_idx), np.array(all_dist)
