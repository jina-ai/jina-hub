from jina import Document
from .. import PostgreSQLDBMSIndexer

from tests import get_documents


def doc_without_embedding(d):
    new_doc = Document()
    new_doc.CopyFrom(d)
    new_doc.ClearField('embedding')
    return new_doc


def test_postgress():
    with PostgreSQLDBMSIndexer(username='default_name', password='default_pwd', database='default_db',
            table='jina_index') as postgres_indexer:
        docs = list(get_documents(chunks=0, same_content=False))
        info = [
            (doc.id, doc.embedding, doc_without_embedding(doc).SerializeToString())
            for doc in docs
        ]
        if info:
            ids, vecs, metas = zip(*info)

            added = postgres_indexer.add(ids, vecs, metas)
            updated = postgres_indexer.update(ids[0], vecs[1], metas[1])
            deleted = postgres_indexer.delete(ids[0])
            assert len(added) == 10
            assert len(updated) == 10
            assert len(deleted) == 9
