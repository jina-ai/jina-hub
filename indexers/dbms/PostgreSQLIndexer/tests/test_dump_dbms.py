import os
from pathlib import Path

import numpy as np
import pytest

from jina import Flow, Document
from jina.drivers.index import DBMSIndexDriver
from jina.executors import BaseExecutor
from jina.executors.indexers.dump import import_vectors, import_metas
from jina.executors.indexers.query import BaseQueryIndexer
from jina.executors.indexers.query.compound import CompoundQueryExecutor
from jina.logging.profile import TimeContext


def doc_without_embedding(d):
    new_doc = Document()
    new_doc.CopyFrom(d)
    new_doc.ClearField('embedding')
    return new_doc


def get_documents(nr=10, index_start=0, emb_size=7):
    for i in range(index_start, nr + index_start):
        with Document() as d:
            d.id = i
            d.text = f'hello world {i}'
            d.embedding = np.random.random(emb_size)
            d.tags['field'] = f'tag data {i}'
        yield d


def assert_dump_data(dump_path, docs, shards, pea_id):
    size_shard = len(docs) // shards
    size_shard_modulus = len(docs) % shards
    ids_dump, vectors_dump = import_vectors(
        dump_path,
        str(pea_id),
    )
    if pea_id == shards - 1:
        docs_expected = docs[
            (pea_id) * size_shard : (pea_id + 1) * size_shard + size_shard_modulus
        ]
    else:
        docs_expected = docs[(pea_id) * size_shard : (pea_id + 1) * size_shard]
    print(f'### pea {pea_id} has {len(docs_expected)} docs')

    # TODO these might fail if we implement any ordering of elements on dumping / reloading
    ids_dump = list(ids_dump)
    vectors_dump = list(vectors_dump)
    np.testing.assert_equal(ids_dump, [d.id for d in docs_expected])
    np.testing.assert_allclose(vectors_dump, [d.embedding for d in docs_expected])

    _, metas_dump = import_metas(
        dump_path,
        str(pea_id),
    )
    metas_dump = list(metas_dump)
    np.testing.assert_equal(
        metas_dump,
        [
            DBMSIndexDriver._doc_without_embedding(d).SerializeToString()
            for d in docs_expected
        ],
    )

    # assert with Indexers
    # TODO currently metas are only passed to the parent Compound, not to the inner components
    with TimeContext(f'### reloading {len(docs_expected)}'):
        # noinspection PyTypeChecker
        cp: CompoundQueryExecutor = BaseQueryIndexer.load_config(
            'indexer_query.yml',
            pea_id=pea_id,
            metas={
                'workspace': os.path.join(dump_path, 'new_ws'),
                'dump_path': dump_path,
            },
        )
    for c in cp.components:
        assert c.size == len(docs_expected)

    # test with the inner indexers separate from the Compound
    for i, indexer_file in enumerate(['query_np.yml', 'query_kv.yml']):
        indexer = BaseQueryIndexer.load_config(
            indexer_file,
            pea_id=pea_id,
            metas={
                'workspace': os.path.realpath(os.path.join(dump_path, f'new_ws-{i}')),
                'dump_path': dump_path,
            },
        )
        assert indexer.size == len(docs_expected)


def path_size(dump_path):
    dir_size = (
            sum(f.stat().st_size for f in Path(dump_path).glob('**/*') if f.is_file()) / 1e6
    )
    return dir_size


@pytest.mark.parametrize('shards', [1, 3, 7])
@pytest.mark.parametrize('nr_docs', [10])
@pytest.mark.parametrize('emb_size', [10])
def test_dump(tmpdir, nr_docs, emb_size, shards):
    docs = list(get_documents(nr=nr_docs, index_start=0, emb_size=emb_size))
    assert len(docs) == nr_docs

    dump_path = os.path.join(str(tmpdir), 'dump_dir')
    os.environ['DBMS_WORKSPACE'] = os.path.join(str(tmpdir), 'index_ws')
    print('DBMS_WORKSPACE ', os.environ['DBMS_WORKSPACE'])
    with Flow.load_config('flow_dbms.yml') as flow_dbms:
        with TimeContext(f'### indexing {len(docs)} docs'):
            flow_dbms.index(docs)

        with TimeContext(f'### dumping {len(docs)} docs'):
            flow_dbms.dump('indexer_dbms', dump_path, shards=shards, timeout=-1)

        dir_size = path_size(dump_path)
        print(f'### dump path size: {dir_size} MBs')

    with BaseExecutor.load(os.path.join(os.environ['DBMS_WORKSPACE'], 'psql-0', 'psql.bin')) as idx:
        assert idx.size == nr_docs

    # assert data dumped is correct
    for pea_id in range(shards):
        assert_dump_data(dump_path, docs, shards, pea_id)

    # required to pass next tests
    with BaseExecutor.load(os.path.join(os.environ['DBMS_WORKSPACE'], 'psql-0', 'psql.bin')) as idx:
        idx.delete([d.id for d in docs])


def _in_docker():
    """ Returns: True if running in a Docker container, else False """
    with open('/proc/1/cgroup', 'rt') as ifh:
        if 'docker' in ifh.read():
            print('in docker, skipping benchmark')
            return True
        return False

# benchmark only
@pytest.mark.skipif(
    _in_docker() or ('GITHUB_WORKFLOW' in os.environ), reason='skip the benchmark test on github workflow or docker'
)
def test_benchmark(tmpdir):
    nr_docs = 100000
    return test_dump(
        tmpdir, nr_docs=nr_docs, emb_size=128, shards=1
    )
