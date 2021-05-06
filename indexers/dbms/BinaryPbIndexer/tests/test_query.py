import numpy as np

from jina import Document, DocumentArray
from ..query_kv import BinaryPbQueryIndexer
from ..query_vec import NumpyQueryIndexer
from ..query_compound import CompoundIndexer


def test_query_keyvalue(tmpdir):
    metas = {'workspace': str(tmpdir), 'name': 'dbms', 'pea_id': 0, 'replica_id': 0}
    with BinaryPbQueryIndexer(
        dump_path='tests/dump1', index_filename='dbms', metas=metas
    ) as indexer:
        docs = DocumentArray([Document(id=1), Document(id=42)])
        indexer.search(docs)
        assert len(docs) == 1
        assert docs[0].text == 'hello world 1'


def test_query_vector(tmpdir):
    metas = {'workspace': str(tmpdir), 'name': 'dbms', 'pea_id': 0, 'replica_id': 0}
    indexer = NumpyQueryIndexer(
        dump_path='tests/dump1', index_filename='dbms', metas=metas
    )
    docs = DocumentArray([Document(embedding=np.random.random(7))])
    TOP_K = 5
    indexer.search(docs, {'top_k': TOP_K})
    assert len(docs) == 1
    assert len(docs[0].matches) == TOP_K
    assert len(docs[0].matches[0].embedding) == 7


def test_query_compound(tmpdir):
    metas = {'workspace': str(tmpdir), 'name': 'dbms', 'pea_id': 0, 'replica_id': 0}
    indexer = CompoundIndexer(
        vector_indexer_class=NumpyQueryIndexer,
        kv_indexer_class=BinaryPbQueryIndexer,
        dump_path='tests/dump1',
        index_filename='dbms',
        metas=metas,
    )
    docs = DocumentArray([Document(embedding=np.random.random(7))])
    TOP_K = 5
    indexer.search(docs, {'top_k': TOP_K})
    assert len(docs) == 1
    assert len(docs[0].matches) == TOP_K
    assert len(docs[0].matches[0].embedding) == 7
    assert docs[0].matches[0].text[:11] == 'hello world'
