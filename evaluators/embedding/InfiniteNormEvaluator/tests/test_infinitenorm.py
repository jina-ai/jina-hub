import numpy as np
import pytest

from .. import InfiniteNormEvaluator


@pytest.mark.parametrize(
    'doc_embedding, gt_embedding, expected',
    [
        ([0, 0], [0, 0], 0.0),
        ([0, 0], [0, 1], 1.0),
        ([0, 0], [1, 0], 1.0),
        ([0, 0], [1, 1], 2.0),
        ([0, 1], [0, 0], 1.0),
        ([0, 1], [0, 1], 0.0),
        ([0, 1], [1, 0], 2.0),
        ([0, 1], [1, 1], 1.0),
        ([1, 0], [0, 0], 1.0),
        ([2, 0], [0, 1], 3.0),
        ([1, 0], [1, 0], 0.0),
        ([1, 0], [1, 1], 1.0),
        ([3, 1], [1, 0], 3.0),
        ([1, 1], [0, 1], 1.0),
        ([1, 1], [1, 0], 1.0),
        ([1, 1], [1, 1], 0.0)
    ]
)
def test_infinitenorm_evaluator(doc_embedding, gt_embedding, expected):
    evaluator = InfiniteNormEvaluator()
    assert evaluator.evaluate(actual=doc_embedding, desired=gt_embedding) == expected
    np.testing.assert_almost_equal(evaluator.mean, expected)
