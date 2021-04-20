import os

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
                d.embedding = np.random.random(d_embedding.shape)
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
                        c.embedding = np.random.random(c_embedding.shape)
                    else:
                        c.text = f'hello world from chunk {j}'
                        c.embedding = np.random.random(c_embedding.shape)
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
        np.testing.assert_equal(vecs[i], np.frombuffer(record[i][1]))
        assert metas[i] == bytes(record[i][2])


def test_postgress(tmpdir):
    postgres_indexer = PostgreSQLDBMSIndexer()
    postgres_indexer.handler.connect()

    original_docs = list(get_documents(chunks=0, same_content=False))
    info_original_docs = [
        (doc.id, doc.embedding, doc_without_embedding(doc).SerializeToString())
        for doc in original_docs
    ]
    ids, vecs, metas = zip(*info_original_docs)

    added = postgres_indexer.handler.add(ids, vecs, metas)
    assert added == 10
    validate_db_side(postgres_indexer, info_original_docs)

    new_docs = list(get_documents(chunks=False, nr=10, same_content=True))
    info_new_docs = [
        (doc.id, doc.embedding, doc_without_embedding(doc).SerializeToString())
        for doc in new_docs
    ]
    ids, vecs, metas = zip(*info_new_docs)

    updated = postgres_indexer.handler.update(ids, vecs, metas)
    expected_info = [(ids[0], vecs[0], metas[0])]
    assert updated == 10
    validate_db_side(postgres_indexer, expected_info)

    deleted = postgres_indexer.handler.delete(ids)
    assert deleted == 10
