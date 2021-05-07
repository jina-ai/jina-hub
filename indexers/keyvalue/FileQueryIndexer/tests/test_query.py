from jina import Document, DocumentArray
from .. import FileQueryIndexer


def test_query_keyvalue(tmpdir):
    metas = {'workspace': str(tmpdir), 'name': 'dbms', 'pea_id': 0, 'replica_id': 0}
    with FileQueryIndexer(
        dump_path='tests/dump1', index_filename='dbms', metas=metas
    ) as indexer:
        docs = DocumentArray([Document(id=1), Document(id=42)])
        indexer.search(docs)
        print(docs)
        assert len(docs) == 1
        assert docs[0].text == 'hello world 1'
