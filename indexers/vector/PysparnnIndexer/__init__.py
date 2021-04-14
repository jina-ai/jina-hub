import scipy
import numpy as np

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

    def __init__(self,
                 k_clusters: int = 2,
                 metric: str = 'cosine',
                 num_indexes: int = 2,
                 prefix_filename: str = 'pysparnn_index',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = self._assign_distance_class(metric)
        self.k_clusters = k_clusters
        self.num_indexes = num_indexes
        self.multi_cluster_index = None
        self.index_filename = prefix_filename + '_index.npz'
        self.indices_filename = prefix_filename + '_indices.npy'

    def get_add_handler(self):
        pass

    def get_create_handler(self):
        pass

    def get_write_handler(self):
        pass

    def get_query_handler(self):
        pass

    def post_init(self):
        import os
        super().post_init()
        self.embeddings = []
        self.indices = []
        self.index = {}
        if os.path.exists(self.index_abspath):
            self._load_index_from_disk()
        else:
            self.indices = []

    def close(self) -> None:
        """
        Release the resources as executor is destroyed, need to be overridden
        """
        self._store_index_to_disk()
        super().close()

    def _assign_distance_class(self, metric: str):
        from pysparnn import matrix_distance

        if metric == 'cosine':
            class_metric = matrix_distance.CosineDistance
        elif metric == 'unit_cosine':
            class_metric = matrix_distance.UnitCosineDistance
        elif metric == 'euclidean':
            class_metric = matrix_distance.SlowEuclideanDistance
        elif metric == 'dense_cosine':
            class_metric = matrix_distance.DenseCosineDistance
        else:
            raise ValueError(f'metric={metric} is not a valid metric')

        return class_metric

    def build_advanced_index(self):
        import pysparnn.cluster_index as ci
        import scipy

        if not self.index:
            raise ValueError('Index is empty, please add data into the indexer using `add` method.')
        keys = []
        indexed_vectors = []
        for key, vector in self.index.items():
            keys.append(key)
            indexed_vectors.append(vector)

        self.multi_cluster_index = ci.MultiClusterIndex(
            features=scipy.sparse.vstack(indexed_vectors),
            records_data=keys,
            distance_type=self.metric,
            num_indexes=self.num_indexes)

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
            self.index.pop(key)

    def _store_index_to_disk(self):
        """Store self.index to disk"""
        import scipy
        scipy.sparse.save_npz(self.index_abspath, scipy.sparse.vstack(self.index.values()))

        with open(self.get_file_from_workspace(self.indices_filename), 'wb') as f:
            np.save(f, list(self.index.keys()))

    def _load_index_from_disk(self):
        """Load self.index from disk"""
        import scipy
        vectors = scipy.sparse.load_npz(self.index_abspath)

        with open(self.get_file_from_workspace(self.indices_filename), 'rb') as f:
            indices = np.load(f)

        self.index = {ind: vec for ind, vec in zip(indices, vectors)}
