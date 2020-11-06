import pytest

from .. import MeanReciprocalRankEvaluator


@pytest.mark.parametrize('actual, desired, score',
[
    ([],[], 0.0),
    ([[1, 2, 3, 4],[]], [[], []], 0.0),
    ([[], []], [[1, 2, 3, 4],[]], 0.0),
    ([[1, 2, 3, 4]],[[1, 2, 3, 4]], 1.0),
    ([[1, 2, 3, 4]],[[11, 1, 2, 3]], 0.0),
    ([[4, 2, 3, 1]],[[1, 2, 3, 4]], 0.0),
    ([[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [2, 1, 3, 4]], (1+0.5)/2)
]
)
def test_meanreciprocalrankevaluator(actual, desired, score):
    evaluator = MeanReciprocalRankEvaluator(eval_at=3)
    assert evaluator.evaluate(actual, desired) == score