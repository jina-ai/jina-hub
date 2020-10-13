__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Tuple

import numpy as np
from jina.executors.indexers import BaseVectorIndexer


class MilvusIndexer(BaseVectorIndexer):
    """Milvus powered vector indexer

        For more information about Milvus:
            - https://github.com/milvus-io/milvus/
    """
    def __init__(self, host: str = '0.0.0.0', port: int = 19530,
                 collection_name: str = 'default', index_type: str = 'IVF,Flat',
                 index_params=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if index_params is None:
            index_params = dict({'nlist': 10})
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.index_type = index_type
        self.index_params = index_params

    def post_init(self):
        from milvusdbhandler import MilvusDBHandler
        super().post_init()
        self.milvus = MilvusDBHandler(self.host, self.port, self.collection_name)

    def get_query_handler(self):
        db_handler = self.milvus.connect()
        db_handler.build_index(self.index_type, self.index_params)
        return db_handler

    def get_add_handler(self):
        return self.milvus.connect()

    def get_create_handler(self):
        return self.milvus.connect()

    @property
    def query_handler(self):
        return self.get_query_handler()

    def add(self, keys: 'np.ndarray', vectors: 'np.ndarray', *args, **kwargs):
        self._validate_key_vector_shapes(keys, vectors)
        self.write_handler.insert(keys, vectors)

    def query(self, keys: 'np.ndarray', top_k: int, *args, **kwargs) -> Tuple['np.ndarray', 'np.ndarray']:
        dist, ids = self.query_handler.search(keys, top_k, *args, **kwargs)
        return np.array(ids), np.array(dist)
