import numpy as np
import pytest

from .. import L1NormEvaluator


@pytest.mark.parametrize(
    'doc_embedding, gt_embedding, expected',
    [
        ([0, 0], [0, 0], 0.0),
        ([0, 0], [0, 1], 1.0),
        ([0, 0], [1, 0], 1.0),
        ([0, 0], [1, 1], 1.0),
        ([0, 1], [0, 0], 1.0),
        ([0, 1], [0, 1], 0.0),
        ([0, 1], [1, 0], 1.0),
        ([0, 1], [1, 1], 1.0),
        ([1, 0], [0, 0], 1.0),
        ([1, 0], [0, 1], 1.0),
        ([1, 0], [1, 0], 0.0),
        ([1, 0], [1, 1], 1.0),
        ([1, 1], [0, 0], 1.0),
        ([1, 1], [0, 1], 1.0),
        ([1, 1], [1, 0], 1.0),
        ([1, 1], [1, 1], 0.0),
        ([1], [1, 1], 0.0),
    ]
)
def test_l1norm_evaluator(doc_embedding, gt_embedding, expected):
    evaluator = L1NormEvaluator()
    assert evaluator.evaluate(actual=doc_embedding, desired=gt_embedding) == expected
    np.testing.assert_almost_equal(evaluator.mean, expected)
