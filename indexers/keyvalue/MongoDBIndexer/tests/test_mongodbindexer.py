import random

import numpy as np

from jina.proto import jina_pb2
from jina.types.document import uid
from jina.proto.ndarray.generic import GenericNdArray
from .. import MongoDBIndexer


def random_docs(num_docs, chunks_per_doc=5, embed_dim=10, jitter=1):
    c_id = 3 * num_docs  # avoid collision with docs
    for j in range(num_docs):
        d = jina_pb2.Document()
        d.tags['id'] = j
        d.text = b'hello world doc id %d' % j
        GenericNdArray(d.embedding).value = np.random.random([embed_dim + np.random.randint(0, jitter)])
        d.id = uid.new_doc_id(d)
        for k in range(chunks_per_doc):
            c = d.chunks.add()
            c.text = 'i\'m chunk %d from doc %d' % (c_id, j)
            GenericNdArray(c.embedding).value = np.random.random([embed_dim + np.random.randint(0, jitter)])
            c.tags['id'] = c_id
            c.tags['parent_id'] = j
            c_id += 1
            c.parent_id = d.id
            c.id = uid.new_doc_id(c)
        yield d


def test_mongodbindexer():
    keys = []
    values = []

    num_docs = 5
    docs = list(random_docs(num_docs=num_docs,
                            chunks_per_doc=3))
    keys = [uid.id2hash(doc.id) for doc in docs]
    values = [doc.SerializeToString() for doc in docs]

    query_index = random.randint(0, num_docs - 1)
    query_id = docs[query_index].id
    query_key = uid.id2hash(query_id)
    query_text = docs[query_index].text

    with MongoDBIndexer() as mongo_indexer:
        mongo_indexer.add(keys=keys, values=values)

    with MongoDBIndexer() as mongo_query:
        query_results = mongo_query.query(key=query_key)
        assert query_results is not None

        for result in query_results:
            assert result['_id'] == query_key
            d = jina_pb2.Document()
            d.ParseFromString(result['values'])
            assert d.text == query_text


if __name__ == "__main__":
    test_mongodbindexer()
