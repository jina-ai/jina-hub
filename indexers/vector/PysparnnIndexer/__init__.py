import scipy
import numpy as np
import pysparnn.cluster_index as ci

from jina.executors.indexers.vector import BaseVectorIndexer


def check_indexer(func):
    def checker(self, *args, **kwargs):
        if self.multi_cluster_index:
            raise ValueError('Not possible query while indexing')
        else:
            return func(self, *args, **kwargs)
    return checker


class PysparnnIndexer(BaseVectorIndexer):
    """
    :class:`PysparnnIndexer` Approximate Nearest Neighbor Search for Sparse Data in Python using PySparNN.
    """

    def __init__(self, k_clusters=2, metric: str = 'cosine', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.index = {}
        self.metric = metric  #TODO add metric in search
        self.k_clusters = k_clusters
        self.multi_cluster_index = None

    def build_advanced_index(self):
        keys = []
        indexed_vectors = []
        for key, vector in self.index.items():
            keys.append(key)
            indexed_vectors.append(vector)

        self.multi_cluster_index = ci.MultiClusterIndex(scipy.sparse.vstack(indexed_vectors), keys)

    def query(self, vectors, top_k, *args, **kwargs):
        """Find the top-k vectors with smallest ``metric`` and return their ids in ascending order.

        :return: a tuple of two ndarrays.
            The first array contains indices, the second array contains distances.
            If `n_vectors = vector.shape[0]` both arrays have shape `n_vectors x top_k`

        :param vectors: the vectors with which to search
        :param args: not used
        :param kwargs: not used
        :param top_k: number of results to return
        :return: tuple of arrays of the form `(indices, distances`
        """

        if not self.multi_cluster_index:
            self.build_advanced_index()

        index_distance_pairs = self.multi_cluster_index.search(vectors,
                                                               k=top_k,
                                                               k_clusters=self.k_clusters,
                                                               return_distance=True)
        distances = []
        indices = []
        for record in index_distance_pairs:
            distances_to_record, indices_to_record = zip(*record)
            distances.append(distances_to_record)
            indices.append(indices_to_record)

        return np.array(indices), np.array(distances)

    @check_indexer
    def add(self, keys, vectors, *args, **kwargs):
        """Add keys and vectors to the indexer.

        :param keys: keys associated to the vectors
        :param vectors: vectors with which to search
        :param args: not used
        :param kwargs: not used

        """
        for key, vector in zip(keys, vectors):
            self.index[key] = vector

    @check_indexer
    def update(
            self, keys, vectors, *args, **kwargs
    ) -> None:
        """Update the embeddings on the index via document ids (keys).

        :param keys: keys associated to the vectors
        :param vectors: vectors with which to search
        :param args: not used
        :param kwargs: not used
        """

        for key, vector in zip(keys, vectors):
            self.index[key] = vector

    @check_indexer
    def delete(self, keys, *args, **kwargs) -> None:
        """Delete the embeddings from the index via document ids (keys).

        :param keys: a list of ids
        :param args: not used
        :param kwargs: not used
        """
        for key in keys:
            del self.index[key]
