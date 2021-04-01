import pytest

from .. import ReciprocalRankEvaluator


@pytest.mark.parametrize('actual, desired, score',
                         [
                             ([], [], 0.0),
                             ([1, 2, 3, 4], [], 0.0),
                             ([], [1, 2, 3, 4], 0.0),
                             ([1, 2, 3, 4], [1, 2, 3, 4], 1.0),
                             ([1, 2, 3, 4], [2, 1, 3, 4], 0.5),
                             ([1, 2, 3, 4], [11, 1, 2, 3], 0.0),
                             ([4, 2, 3, 1], [1, 2, 3, 4], 0.25),
                             ([2, 1, 3, 4, 5, 6, 7, 8, 9, 10], [1, 3, 6, 9, 10], 0.5)
                         ]
                         )
def test_reciprocalrankevaluator(actual, desired, score):
    evaluator = ReciprocalRankEvaluator()
    assert evaluator.evaluate(actual, desired) == score
