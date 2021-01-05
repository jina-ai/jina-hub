import numpy as np
import pytest

from .. import AveragePrecisionEvaluator


@pytest.mark.parametrize(
    'matches_ids, desired_ids, expected',
    [
        ([0, 1, 2, 3], [0, 1, 2, 3], 1.0),
        ([0, 1, 4, 5], [0, 1, 2, 3], 0.5),
        ([4, 5, 6, 7], [0, 1, 2, 3], 0.0),
        ([0, 1, 1, 1], [0, 1, 2, 3], 0.5),
        ([0, 1], [0, 1, 2, 3], 0.5),
        ([0, 1, 4, 2], [0, 1, 2, 3], 0.5),
        ([0, 1, 3], [0, 1, 2, 3], 0.677083),
        ([0, 1, 3, 2], [0, 1, 2, 3], 0.677083),
    ]
)
def test_average_precision_evaluator(matches_ids, desired_ids, expected):

    evaluator = AveragePrecisionEvaluator()
    output = evaluator.evaluate(actual=matches_ids, desired=desired_ids)
    np.testing.assert_array_almost_equal(output, expected)


def test_precision_evaluator_no_groundtruth():
    matches_ids = [0, 1, 2, 3, 4]
    desired_ids = []

    evaluator = AveragePrecisionEvaluator()
    assert evaluator.evaluate(actual=matches_ids, desired=desired_ids) == 0.0


def test_precision_evaluator_no_actuals():
    matches_ids = []
    desired_ids = [1, 2]

    evaluator = AveragePrecisionEvaluator()
    assert evaluator.evaluate(actual=matches_ids, desired=desired_ids) == 0.0
