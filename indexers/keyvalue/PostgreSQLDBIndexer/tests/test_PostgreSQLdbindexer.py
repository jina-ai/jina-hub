from jina import Document
from .. import PostgreSQLDBMSIndexer

from tests import get_documents


def doc_without_embedding(d):
    new_doc = Document()
    new_doc.CopyFrom(d)
    new_doc.ClearField('embedding')
    return new_doc


def test_postgress():
    with PostgreSQLDBMSIndexer(username='user', password='pwd', database='python_test',
            table='sql') as postgres_indexer:
        docs = list(get_documents(chunks=0, same_content=False))
        info = [
            (doc.id, doc.embedding, doc_without_embedding(doc).SerializeToString())
            for doc in docs
        ]
        if info:
            ids, vecs, metas = zip(*info)

            postgres_indexer.add(ids, vecs, metas)
            postgres_indexer.update(ids[0], vecs[1], metas[1])
            postgres_indexer.delete(ids[0])
