import pickle

import numpy as np
from jina import Document

from .. import PostgreSQLDBMSIndexer


d_embedding = np.array([1, 1, 1, 1, 1, 1, 1])
c_embedding = np.array([2, 2, 2, 2, 2, 2, 2])


def get_documents(chunks, same_content, nr=10, index_start=0, same_tag_content=None):
    next_chunk_id = nr + index_start
    for i in range(index_start, nr + index_start):
        with Document() as d:
            d.id = i
            if same_content:
                d.text = 'hello world'
                d.embedding = d_embedding
            else:
                d.text = f'hello world {i}'
                d.embedding = np.random.random(d_embedding.shape)
            if same_tag_content:
                d.tags['field'] = 'tag data'
            elif same_tag_content is False:
                d.tags['field'] = f'tag data {i}'
            for j in range(chunks):
                with Document() as c:
                    c.id = next_chunk_id
                    if same_content:
                        c.text = 'hello world from chunk'
                        c.embedding = c_embedding
                    else:
                        c.text = f'hello world from chunk {j}'
                        c.embedding = np.random.random(d_embedding.shape)
                    if same_tag_content:
                        c.tags['field'] = 'tag data'
                    elif same_tag_content is False:
                        c.tags['field'] = f'tag data {next_chunk_id}'
                next_chunk_id += 1
                d.chunks.append(c)
        yield d


def doc_without_embedding(d):
    new_doc = Document()
    new_doc.CopyFrom(d)
    new_doc.ClearField('embedding')
    return new_doc


def validate_db_side(postgres_indexer, expected_data):
    ids, vecs, metas = zip(*expected_data)
    postgres_indexer.handler.connect()
    postgres_indexer.handler.cursor.execute(f'SELECT ID, VECS, METAS from {postgres_indexer.table} ORDER BY ID')
    record = postgres_indexer.handler.cursor.fetchall()
    for i in range(len(expected_data)):
        assert ids[i] == str(record[i][0])
        assert (np.array(vecs[i]) == np.array(pickle.loads(record[i][1]))).all()
        assert metas[i] == pickle.loads(record[i][2])


def test_postgress():
    postgres_indexer = PostgreSQLDBMSIndexer()
    docs = list(get_documents(chunks=0, same_content=False))
    info = [
        (doc.id, doc.embedding, doc_without_embedding(doc).SerializeToString())
        for doc in docs
    ]
    if info:
        ids, vecs, metas = zip(*info)

        added = postgres_indexer.add(ids, vecs, metas)
        assert len(added) == 10
        validate_db_side(postgres_indexer, info)

        updated = postgres_indexer.update(ids[0], vecs[1], metas[1])
        expected_info = [(ids[0], vecs[1], metas[1])]
        assert len(updated) == 10
        validate_db_side(postgres_indexer, expected_info)

        deleted = postgres_indexer.delete(ids[0])
        assert len(deleted) == 9
        validate_db_side(postgres_indexer, info[1:])
