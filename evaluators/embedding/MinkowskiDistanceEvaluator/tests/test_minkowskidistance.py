from math import sqrt

import numpy as np
import pytest

from .. import MinkowskiDistanceEvaluator


@pytest.mark.parametrize(
    'doc_embedding, gt_embedding, expected, order',
    [
        ([0, 0], [0, 0], 0.0, 3),
        ([0, 0], [0, 1], 1.0, 3),
        ([0, 0], [1, 0], 1.0, 3),
        ([0, 1], [0, 1], 0.0, 3),
        ([0, 1], [1, 1], 1.0, 3),
        ([0, 0], [1, 1], 2 ** (1/3), 3),
        ([0, -1], [1, 0], 2 ** (1/3), 3)
    ]
)

def test_minkowski_distance_order_three(doc_embedding, gt_embedding, expected, order):
    evaluator = MinkowskiDistanceEvaluator(order)
    assert evaluator.evaluate(actual=doc_embedding, desired=gt_embedding) == pytest.approx(expected)
    assert evaluator._running_stats._n == 1


@pytest.mark.parametrize(
    'doc_embedding, gt_embedding, expected, order',
    [
        ([0, 0], [0, 0], 0.0, 1.5),
        ([0, 0], [0, 1], 1.0, 1.5),
        ([0, 0], [1, 0], 1.0, 1.5),
        ([0, 1], [0, 1], 0.0, 1.5),
        ([0, 2], [0, 0], 2.0, 1.5),
        ([0, 0], [1, 1], 2 ** (2/3), 1.5),
        ([0, -1], [1, 0], 2 ** (2/3), 1.5)
    ]
)


def test_minkowski_distance_order_one_and_half(doc_embedding, gt_embedding, expected, order):
    evaluator = MinkowskiDistanceEvaluator(order)
    assert evaluator.evaluate(actual=doc_embedding, desired=gt_embedding) == pytest.approx(expected)
    assert evaluator._running_stats._n == 1


@pytest.mark.parametrize(
    'doc_embedding, gt_embedding, expected, order',
    [
        ([0, 0], [0, 0], 0.0, -1)
    ]
)

def test_minkowski_distance_order_negative(doc_embedding, gt_embedding, expected, order):
    evaluator = MinkowskiDistanceEvaluator(order)
    with pytest.raises(ValueError):
        evaluator.evaluate(actual=doc_embedding, desired=gt_embedding)