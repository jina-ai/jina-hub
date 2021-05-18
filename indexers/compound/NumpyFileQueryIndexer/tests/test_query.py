import numpy as np

from jina import Document, DocumentArray
from .. import NumpyFileQueryIndexer


def test_query_compound(tmpdir):
    metas = {'workspace': str(tmpdir), 'name': 'dbms', 'pea_id': 0, 'replica_id': 0}
    indexer = NumpyFileQueryIndexer(
        source_path='tests/dump1',
        metas=metas,
    )
    docs = DocumentArray([Document(embedding=np.random.random(7))])
    TOP_K = 5
    indexer.search(docs, {'top_k': TOP_K})
    assert len(docs) == 1
    assert len(docs[0].matches) == TOP_K
    assert len(docs[0].matches[0].embedding) == 7
    assert docs[0].matches[0].text[:11] == 'hello world'
