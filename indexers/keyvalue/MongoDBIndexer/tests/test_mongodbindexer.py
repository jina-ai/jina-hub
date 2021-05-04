import random
from collections import Iterable

import numpy as np
from jina import Document

from .. import MongoDBIndexer


def _random_docs(num_docs, chunks_per_doc=5, embed_dim=10, jitter=1):
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
    docs = list(_random_docs(num_docs=num_docs,
                            chunks_per_doc=3))
    keys: Iterable[str] = [doc.id for doc in docs]
    values = [doc.SerializeToString() for doc in docs]

    query_index = random.randint(0, num_docs - 1)
    # adding type annotations to remove further warnings
    query_key: str = docs[query_index].id
    query_text = docs[query_index].text

    # add
    with MongoDBIndexer() as mongo_indexer:
        mongo_indexer.add(keys=keys, values=values)

    with MongoDBIndexer() as mongo_query:
        results = mongo_query.query(keys=[query_key])
        result = results[0]
        assert result['_id'] == query_key
        d = Document()
        d.ParseFromString(result['values'])
        assert d.text == query_text

    # documents to be updated
    new_docs = list(_random_docs(num_docs=num_docs, chunks_per_doc=3))
    # documents for insertion
    new_values = [doc.SerializeToString() for doc in new_docs]
    # documents for assertion
    new_texts = [doc.text for doc in new_docs]

    with MongoDBIndexer() as mongo_indexer:
        mongo_indexer.update(keys=keys, values=new_values)

    with MongoDBIndexer() as mongo_query:
        results = mongo_query.query(keys)

        for key, new_value, result in zip(keys, new_texts, results):
            assert result['_id'] == key
            new_doc = Document()
            new_doc.ParseFromString(result['values'])
            assert new_value == new_doc.text

    # delete
    with MongoDBIndexer() as mongo_query:
        mongo_query.delete(keys)

    with MongoDBIndexer() as mongo_query:
        assert mongo_query.query(keys) == [None] * len(keys)
