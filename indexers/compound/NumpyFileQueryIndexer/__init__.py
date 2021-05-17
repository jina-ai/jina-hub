__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import copy
from typing import Dict

from jina import requests, DocumentArray, Executor
from jina.hub.indexers.keyvalue.FileQueryIndexer import FileQueryIndexer
from jina.hub.indexers.vector.NumpyIndexer import NumpyIndexer


class NumpyFileQueryIndexer(Executor):
    def __init__(self, source_path, *args, **kwargs):
        self._vec_indexer = NumpyIndexer(source_path=source_path, *args, **kwargs)
        self._kv_indexer = FileQueryIndexer(source_path=source_path, *args, **kwargs)

    @requests(on='/search')
    def search(self, docs: 'DocumentArray', parameters: Dict = None, **kwargs):
        self._vec_indexer.search(docs, parameters)
        kv_parameters = copy.deepcopy(parameters)

        kv_parameters['traversal_paths'] = [
            path + 'm' for path in kv_parameters.get('traversal_paths', ['r'])
        ]

        self._kv_indexer.search(docs, kv_parameters)
