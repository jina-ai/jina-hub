import os

import pytest

import numpy as np
from scipy.sparse import csr_matrix

from jina.executors.metas import get_default_metas

from .. import PysparnnIndexer


@pytest.fixture(scope='function', autouse=True)
def metas(tmpdir):
    os.environ['TEST_WORKSPACE'] = str(tmpdir)
    metas = get_default_metas()
    metas['workspace'] = os.environ['TEST_WORKSPACE']
    metas['name'] = 'name'
    yield metas
    del os.environ['TEST_WORKSPACE']


@pytest.fixture
def indexer(metas):
    return PysparnnIndexer(metas=metas, metric='cosine')


@pytest.fixture
def features():
    vectors = np.random.binomial(1, 0.01, size=(50, 100))
    return csr_matrix(vectors)


def test_invalid_metric():
    with pytest.raises(ValueError):
        PysparnnIndexer(metric='cosin')


def test_add(indexer, features):
    indexer.add(keys=list(range(0, 50)), vectors=features)
    assert indexer.index
    assert indexer.index[0].shape == (1, 100)
    assert isinstance(indexer.index[0], csr_matrix)


def test_update(indexer, features):
    indexer.update(keys=list(range(0, 50)), vectors=features)
    assert indexer.index
    assert indexer.index[0].shape == (1, 100)
    assert isinstance(indexer.index[0], csr_matrix)


def test_delete(indexer, features):
    indexer.add(keys=list(range(0, 50)), vectors=features)
    assert indexer.index[0].shape == (1, 100)
    indexer.delete([0])
    with pytest.raises(KeyError):
        indexer.index[0].shape


def test_immutable_with_index_built(indexer, features):
    indexer.add(keys=list(range(0, 50)), vectors=features)
    assert indexer.index[0].shape == (1, 100)
    indexer.build_advanced_index()
    assert indexer.multi_cluster_index
    with pytest.raises(ValueError):
        indexer.add(keys=list(range(50, 100)), vectors=features)
    with pytest.raises(ValueError):
        indexer.update(keys=list(range(0, 50)), vectors=features)
    with pytest.raises(ValueError):
        indexer.delete([0])


def test_query(indexer, features):
    indexer.add(keys=list(range(0, 50)), vectors=features)
    indices, distances = indexer.query(vectors=features[:5], top_k=1)
    assert indices[0] == [0]


def test_close_load(indexer, features):
    indexer.add(keys=list(range(0, 50)), vectors=features)
    assert indexer.index[0].shape == (1, 100)
    indexer.close()
    indexer_query = PysparnnIndexer(metas=indexer.metas, metric='cosine')
    assert indexer.index[0].shape == (1, 100)
    assert indexer_query.index[0].shape == (1, 100)
    assert (indexer.index[0] != indexer_query.index[0]).nnz == 0


def test_delete_close_load(indexer, features):
    keys_to_remove = [5, 10, 15]
    indexer.add(keys=list(range(0, 50)), vectors=features)
    indexer.delete(keys=keys_to_remove)
    indexer.close()
    indexer_from_file = PysparnnIndexer(metas=indexer.metas, metric='cosine')
    assert len(indexer_from_file.index) == len(indexer.index)
