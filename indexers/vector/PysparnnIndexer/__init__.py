__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import typing
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, bsr_matrix, csc_matrix
from pysparnn.matrix_distance import (
    CosineDistance,
    UnitCosineDistance,
    SlowEuclideanDistance,
    DenseCosineDistance,
)

from jina.executors.indexers.vector import BaseVectorIndexer

SparseMatrixType = typing.Union[csr_matrix, coo_matrix, bsr_matrix, csc_matrix]


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

    For more information about the Pysparnn supported parameters and installation please consult:
        - https://github.com/facebookresearch/pysparnn

    """

    embedding_cls_type = 'scipy_csr'

    def __init__(
        self,
        k_clusters: int = 2,
        metric: str = 'cosine',
        num_indexes: int = 2,
        prefix_filename: str = 'pysparnn_index',
        *args,
        **kwargs,
    ):
        """Initializes a PysparnnIndexer Indexer

        :param k_clusters: number of clusters to be used in the multi_cluster_index from Pysparnn.
        :param metric: type of metric used for query, one from ['cosine', 'unit_cosine'. 'euclidean','dense_cosine']
        :param num_indexes: number of indexses used in the multi_cluster_index from Pysparnn.
        :param prefix_filename: prefix used when storing indices to disk
        :param args: additional parameters.
        :param kwargs: additional positional parameters.
        """
        super().__init__(*args, **kwargs)
        self.index = {}
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

    def post_init(self) -> None:
        """Load index if exist."""
        import os

        super().post_init()
        if os.path.exists(self.index_abspath):
            self._load_index_from_disk()

    def close(self) -> None:
        """
        Release the resources as executor is destroyed, need to be overridden
        """
        self._store_index_to_disk()
        super().close()

    def _assign_distance_class(self, metric: str):

        if metric == 'cosine':
            class_metric = CosineDistance
        elif metric == 'unit_cosine':
            class_metric = UnitCosineDistance
        elif metric == 'euclidean':
            class_metric = SlowEuclideanDistance
        elif metric == 'dense_cosine':
            class_metric = DenseCosineDistance
        else:
            raise ValueError(f'metric={metric} is not a valid metric')

        return class_metric

    def build_advanced_index(self) -> None:
        """Build the index using pysparnn `cluster_index` and stores it in `multi_cluster_index` """

        import pysparnn.cluster_index as ci
        import scipy

        if not self.index:
            raise ValueError(
                'Index is empty, please add data into the indexer using `add` method.'
            )
        keys = []
        indexed_vectors = []
        for key, vector in self.index.items():
            keys.append(key)
            indexed_vectors.append(vector)

        self.multi_cluster_index = ci.MultiClusterIndex(
            features=scipy.sparse.vstack(indexed_vectors),
            records_data=keys,
            distance_type=self.metric,
            num_indexes=self.num_indexes,
        )

    def query(self, vectors: SparseMatrixType, top_k: int, *args, **kwargs):
        """Find the top-k vectors with smallest ``metric`` and return their ids in ascending order.

        :return: a tuple of two ndarrays.
            The first array contains indices, the second array contains distances.
            If `n_vectors = vector.shape[0]` both arrays have shape `n_vectors x top_k`

        :param vectors: the vectors with which to search
        :param top_k: number of results to return
        :param args: additional positional parameters.
        :param kwargs: additional positional parameters.
        :return: tuple of arrays of the form `(indices, distances`
        """

        if not self.multi_cluster_index:
            self.build_advanced_index()

        index_distance_pairs = self.multi_cluster_index.search(
            vectors, k=top_k, k_clusters=self.k_clusters, return_distance=True
        )
        distances = []
        indices = []
        for record in index_distance_pairs:
            distance_list, index_list = zip(*record)
            distances.append(distance_list)
            indices.append(index_list)
        return np.array(indices), np.array(distances).astype(np.float)

    @check_indexer
    def add(
        self, keys: typing.List, vectors: SparseMatrixType, *args, **kwargs
    ) -> None:
        """Add keys and vectors to the indexer.

        :param keys: keys associated to the vectors
        :param vectors: vectors with which to search
        :param args: additional positional parameters.
        :param kwargs: additional positional parameters.
        """
        for key, vector in zip(keys, vectors):
            self.index[key] = vector

    @check_indexer
    def update(
        self, keys: typing.List, vectors: SparseMatrixType, *args, **kwargs
    ) -> None:
        """Update the embeddings on the index via document ids (keys).

        :param keys: keys associated to the vectors
        :param vectors: vectors with which to search
        :param args: additional positional parameters.
        :param kwargs: additional positional parameters.
        """

        for key, vector in zip(keys, vectors):
            self.index[key] = vector

    @check_indexer
    def delete(self, keys: typing.List, *args, **kwargs) -> None:
        """Delete the embeddings from the index via document ids (keys).

        :param keys: a list of ids
        :param args: additional positional parameters.
        :param kwargs: additional positional parameters.
        """
        for key in keys:
            self.index.pop(key)

    def _store_index_to_disk(self):
        """Store self.index to disk"""
        import scipy
        self.logger.info(f'Storing vectors to {self.index_abspath}')
        self.logger.info(f'Storing indices from {self.get_file_from_workspace(self.indices_filename)}')

        scipy.sparse.save_npz(
            self.index_abspath, scipy.sparse.vstack(self.index.values())
        )

        with open(self.get_file_from_workspace(self.indices_filename), 'wb') as f:
            np.save(f, list(self.index.keys()))

    def _load_index_from_disk(self):
        """Load self.index from disk"""
        import scipy
        self.logger.info(f'Loading vectors from {self.index_abspath}')
        self.logger.info(f'Loading indices from {self.get_file_from_workspace(self.indices_filename)}')

        vectors = scipy.sparse.load_npz(self.index_abspath)

        with open(self.get_file_from_workspace(self.indices_filename), 'rb') as f:
            indices = np.load(f)

        self.index = {ind: vec for ind, vec in zip(indices, vectors)}
