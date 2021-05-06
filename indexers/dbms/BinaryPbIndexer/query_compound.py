import copy
from typing import Dict

from jina import requests, DocumentArray


class CompoundIndexer:
    def __init__(self, vector_indexer_class, kv_indexer_class, *args, **kwargs):
        self._vec_indexer = vector_indexer_class(*args, **kwargs)
        self._kv_indexer = kv_indexer_class(*args, **kwargs)

    @requests(on='/search')
    def search(self, docs: 'DocumentArray', parameters: Dict = None, **kwargs):
        self._vec_indexer.search(docs, parameters)
        kv_parameters = copy.deepcopy(parameters)

        kv_parameters['traversal_paths'] = [
            path + 'm' for path in kv_parameters.get('traversal_paths', ['r'])
        ]

        self._kv_indexer.search(docs, kv_parameters)
