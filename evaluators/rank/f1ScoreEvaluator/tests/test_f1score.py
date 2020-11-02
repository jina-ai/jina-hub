import pytest
import numpy as np
from .. import f1ScoreEvaluator


@pytest.mark.parametrize(
    'eval_at, expected',
    [
        (0, 0.0),
        (2, 0.5714),
        (5, 0.4),
        (100, 0.4)
    ]
)
def test_f1score_evaluator(eval_at, expected):
    matches_ids = [0, 1, 2, 3, 4]

    desired_ids = [1, 0, 20, 30, 40]

    evaluator = f1ScoreEvaluator(eval_at=eval_at)
    assert evaluator.evaluate(actual=matches_ids, desired=desired_ids) == expected
    np.testing.assert_almost_equal(evaluator.mean, expected)



