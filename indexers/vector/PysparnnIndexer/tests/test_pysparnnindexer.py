
import pytest
import numpy as np
from scipy.sparse import csr_matrix

from .. import PysparnnIndexer


@pytest.fixture
def indexer():
    return PysparnnIndexer()


@pytest.fixture
def features():
    vectors = np.random.binomial(1, 0.01, size=(50, 100))
    return csr_matrix(vectors)


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


def test_query():
    pass


def test_save_load(indexer, features):
    indexer.add(keys=list(range(0, 50)), vectors=features)
    assert indexer.index[0].shape == (1, 100)
    index_before_save = indexer.index[0]
    indexer.build_advanced_index()
    indexer.save(filename='abc')
    indexer = PysparnnIndexer.load(filename='abc')
    assert indexer.index[0].shape == (1, 100)
    assert (indexer.index[0] != index_before_save).nnz == 0

