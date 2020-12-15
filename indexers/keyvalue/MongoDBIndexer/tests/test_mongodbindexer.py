import random
from collections import Iterator

import numpy as np
from jina import Document

from .. import MongoDBIndexer


def random_docs(num_docs, chunks_per_doc=5, embed_dim=10, jitter=1):
    c_id = 3 * num_docs  # avoid collision with docs
    for j in range(num_docs):
        with Document() as d:
            d.tags['id'] = j
            d.text = b'hello world doc id %d' % j
            d.embedding = np.random.random([embed_dim + np.random.randint(0, jitter)])
            for k in range(chunks_per_doc):
                with Document() as c:
                    c.text = 'i\'m chunk %d from doc %d' % (c_id, j)
                    c.embedding = np.random.random([embed_dim + np.random.randint(0, jitter)])
                    c.tags['id'] = c_id
                    c.tags['parent_id'] = j
                    c_id += 1
                    c.parent_id = d.id
                d.chunks.add(c)
        yield d


def test_mongodbindexer():
    num_docs = 5
    docs = list(random_docs(num_docs=num_docs,
                            chunks_per_doc=3))
    keys: Iterator[int] = iter([doc.id for doc in docs])
    values = [doc.SerializeToString() for doc in docs]

    query_index = random.randint(0, num_docs - 1)
    # adding type annotations to remove further warnings
    query_key: int = docs[query_index].id
    query_text = docs[query_index].text

    # add
    with MongoDBIndexer() as mongo_indexer:
        mongo_indexer.add(keys=keys, values=values)

    with MongoDBIndexer() as mongo_query:
        result = mongo_query.query(key=query_key)
        assert result['_id'] == query_key
        d = Document()
        d.ParseFromString(result['values'])
        assert d.text == query_text

    # update
    new_docs = list(random_docs(num_docs=num_docs, chunks_per_doc=3))
    # what we insert
    new_values = [doc.SerializeToString() for doc in new_docs]
    # what we assert
    new_texts = [doc.text for doc in new_docs]

    with MongoDBIndexer() as mongo_indexer:
        mongo_indexer.update(keys=keys, values=new_values)

    with MongoDBIndexer() as mongo_query:
        for key, new_value in zip(keys, new_texts):
            result = mongo_query.query(key)
            assert result['_id'] == key
            new_doc = Document()
            new_doc.ParseFromString(result['values'])
            assert new_value == new_doc.text

    # delete
    with MongoDBIndexer() as mongo_query:
        mongo_query.delete(keys)

    with MongoDBIndexer() as mongo_query:
        for key in keys:
            result = mongo_query.query(key)
            assert result is None
