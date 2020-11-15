import os
import random

import numpy as np
import pytest

from jina.executors.metas import get_default_metas
from jina.proto import jina_pb2
from jina.types.document import uid
from jina.proto.ndarray.generic import GenericNdArray
from .. import RedisDBIndexer


@pytest.fixture(scope='function', autouse=True)
def metas(tmpdir):
    os.environ['TEST_WORKSPACE'] = str(tmpdir)
    metas = get_default_metas()
    metas['workspace'] = os.environ['TEST_WORKSPACE']
    yield metas
    del os.environ['TEST_WORKSPACE']


def random_docs(num_docs, chunks_per_doc=5, embed_dim=10, jitter=1):
    c_id = 3 * num_docs  # avoid collision with docs
    for j in range(num_docs):
        d = jina_pb2.Document()
        d.tags['id'] = j
        d.text = b'hello world doc id %d' % j
        GenericNdArray(d.embedding).value = np.random.random([embed_dim + np.random.randint(0, jitter)])
        d.id = uid.new_doc_id(d)
        yield d


def test_redis_db_indexer(metas):
    num_docs = 5
    docs = list(random_docs(num_docs=num_docs,
                            chunks_per_doc=3))
    keys = [uid.id2hash(doc.id) for doc in docs]
    values = [doc.SerializeToString() for doc in docs]

    query_index = random.randint(0, num_docs - 1)
    query_id = docs[query_index].id
    query_key = uid.id2hash(query_id)
    query_text = docs[query_index].text

    with RedisDBIndexer(metas=metas) as idx:
        idx.add(keys=keys, values=values)

    with RedisDBIndexer(metas=metas) as redis_query:
        query_results = redis_query.query(key=query_key)
        for result in query_results:
            assert result is not None
            assert result['key'] == str(query_key).encode()
            d = jina_pb2.Document()
            d.ParseFromString(result['values'])
            assert d.text == query_text
