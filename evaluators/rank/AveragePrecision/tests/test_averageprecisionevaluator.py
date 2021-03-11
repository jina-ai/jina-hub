import pytest

from .. import AveragePrecisionEvaluator


@pytest.mark.parametrize(
    'matches_ids, desired_ids, expected',
    [
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 3, 6, 9, 10], 0.6222),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [7, 5, 2], 0.4428),
        ([0, 1, 2, 3], [0, 1, 2, 3], 1.0),
        ([0, 1], [0, 1, 2, 3], 0.5),
        ([0, 1, 4, 5], [0, 1, 2, 3], 0.5),
        ([4, 5, 6, 7], [0, 1, 2, 3], 0.0),
        ([0, 1, 4, 2], [0, 1, 2, 3], 0.6875),
        ([0, 1, 3], [0, 1, 2, 3], 0.75),
        ([0, 1, 3, 2], [0, 1, 2, 3], 1.0),
    ]
)
def test_average_precision_evaluator(matches_ids, desired_ids, expected):

    evaluator = AveragePrecisionEvaluator()
    output = evaluator.evaluate(actual=matches_ids, desired=desired_ids)
    assert output == pytest.approx(expected, 0.001)


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
