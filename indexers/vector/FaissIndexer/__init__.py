__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Tuple, Optional

import numpy as np
from jina.executors.devices import FaissDevice
from jina.executors.indexers.vector import BaseNumpyIndexer
from jina.executors.decorators import batching


class FaissIndexer(FaissDevice, BaseNumpyIndexer):
    batch_size = 512

    """Faiss powered vector indexer

    For more information about the Faiss supported parameters and installation problems, please consult:
        - https://github.com/facebookresearch/faiss

    .. note::
        Faiss package dependency is only required at the query time.
    """

    def __init__(self,
                 index_key: str,
                 train_filepath: Optional[str] = None,
                 distance: str = 'l2',
                 normalize: bool = False,
                 nprobe: int = 1,
                 *args,
                 **kwargs):
        """
        Initialize an Faiss Indexer

        :param index_key: index type supported by ``faiss.index_factory``
        :param train_filepath: the training data file path, e.g ``faiss.tgz`` or `faiss.npy`. The data file is expected
            to be either `.npy` file from `numpy.save()` or a `.tgz` file from `NumpyIndexer`.
        :param distance: 'l2' or 'inner_product' accepted. Determines which distances to optimize by FAISS. l2...smaller is better, inner_product...larger is better
        :param normalize: whether or not to normalize the vectors e.g. for the cosine similarity https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-index-vectors-for-cosine-similarity
        :param nprobe: Number of clusters to consider at search time.

        .. highlight:: python
        .. code-block:: python
            # generate a training file in `.tgz`
            import gzip
            import numpy as np
            from jina.executors.indexers.vector.faiss import FaissIndexer

            train_filepath = 'faiss_train.tgz'
            train_data = np.random.rand(10000, 128)
            with gzip.open(train_filepath, 'wb', compresslevel=1) as f:
                f.write(train_data.astype('float32'))
            indexer = FaissIndexer('PCA64,FLAT', train_filepath)

            # generate a training file in `.npy`
            train_filepath = 'faiss_train'
            np.save(train_filepath, train_data)
            indexer = FaissIndexer('PCA64,FLAT', train_filepath)
        """
        super().__init__(*args, **kwargs)
        self.index_key = index_key
        self.train_filepath = train_filepath
        self.distance = distance
        self.normalize = normalize
        self.nprobe = nprobe

    def build_advanced_index(self, vecs: 'np.ndarray'):
        """Load all vectors (in numpy ndarray) into Faiss indexers """
        import faiss

        metric = faiss.METRIC_L2
        if self.distance == 'inner_product':
            metric = faiss.METRIC_INNER_PRODUCT
        if self.distance not in {'inner_product', 'l2'}:
            self.logger.warning('Invalid distance metric for Faiss index construction. Defaulting to l2 distance')

        index = self.to_device(index=faiss.index_factory(self.num_dim, self.index_key, metric))
        if self.train_filepath:
            train_data = self._load_training_data(self.train_filepath)
            if train_data is None:
                self.logger.warning('loading training data failed. some faiss indexes require previous training.')
            else:
                train_data = train_data.astype(np.float32)
                if self.normalize:
                    faiss.normalize_L2(train_data)
                self._train(index, train_data)
        
        self.build_partial_index(vecs, index)
        index.nprobe = self.nprobe
        return index

    @batching
    def build_partial_index(self, vecs: 'np.ndarray', index):
        vecs = vecs.astype(np.float32)
        if self.normalize:
            from faiss import normalize_L2
            normalize_L2(vecs)
        index.add(vecs)

    def query(self, vecs: 'np.ndarray', top_k: int, *args, **kwargs) -> Tuple['np.ndarray', 'np.ndarray']:
        if self.normalize:
            from faiss import normalize_L2
            normalize_L2(vecs)
        dist, ids = self.query_handler.search(vecs, top_k)
        keys = self.int2ext_id[self.valid_indices][ids]
        return keys, dist

    def _train(self, index, data: 'np.ndarray', *args, **kwargs) -> None:
        _num_samples, _num_dim = data.shape
        if not self.num_dim:
            self.num_dim = _num_dim
        if self.num_dim != _num_dim:
            raise ValueError('training data should have the same number of features as the index, {} != {}'.format(
                self.num_dim, _num_dim))
        index.train(data)

    def _load_training_data(self, train_filepath: str) -> 'np.ndarray':
        result = None
        try:
            result = self._load_gzip(train_filepath)
        except Exception as e:
            self.logger.error('loading training data from gzip failed, filepath={}, {}'.format(train_filepath, e))

        if result is None:
            try:
                result = np.load(train_filepath)
                if isinstance(result, np.lib.npyio.NpzFile):
                    self.logger.warning('.npz format is not supported. Please save the array in .npy format.')
                    result = None
            except Exception as e:
                self.logger.error('loading training data with np.load failed, filepath={}, {}'.format(train_filepath, e))

        if result is None:
            try:
                # Read from binary file:
                with open(train_filepath, 'rb') as f:
                    result = f.read()
            except Exception as e:
                self.logger.error('loading training data from binary file failed, filepath={}, {}'.format(train_filepath, e))
        return result
