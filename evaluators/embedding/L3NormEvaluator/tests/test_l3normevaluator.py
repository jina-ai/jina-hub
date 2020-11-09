from math import sqrt

import numpy as np
import pytest

from .. import L3NormEvaluator


@pytest.mark.parametrize(
    'doc_embedding, gt_embedding, expected',
    [
        ([0, 0], [0, 0], 0.0),
        ([0, 0], [0, 1], 1.0),
        ([0, 0], [1, 0], 1.0),
        ([0, 1], [0, 1], 0.0),
        ([0, 1], [1, 1], 1.0),
        ([0, 0], [1, 1], 2 ** (1/3)),
        ([0, -1], [1, 0], 2 ** (1/3))
    ]
)


def test_l3norm_evaluator(doc_embedding, gt_embedding, expected):

    evaluator = L3NormEvaluator()
    assert evaluator.evaluate(actual=doc_embedding, desired=gt_embedding) == pytest.approx(expected)
    assert evaluator._running_stats._n == 1
    np.testing.assert_almost_equal(evaluator.mean, expected)