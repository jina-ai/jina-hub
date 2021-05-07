import numpy as np

from jina import Document, DocumentArray
from jina.hub.indexers.vector.NumpyIndexer import NumpyIndexer


def test_query_vector(tmpdir):
    metas = {'workspace': str(tmpdir), 'name': 'dbms', 'pea_id': 0, 'replica_id': 0}
    indexer = NumpyIndexer(dump_path='tests/dump1', index_filename='dbms', metas=metas)
    docs = DocumentArray([Document(embedding=np.random.random(7))])
    TOP_K = 5
    indexer.search(docs, {'top_k': TOP_K})
    assert len(docs) == 1
    assert len(docs[0].matches) == TOP_K
    assert len(docs[0].matches[0].embedding) == 7
