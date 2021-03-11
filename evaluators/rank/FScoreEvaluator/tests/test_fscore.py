import pytest

from .. import FScoreEvaluator


@pytest.mark.parametrize(
    'eval_at, beta, expected',
    [
        (None, 1.0, 0.4),
        (0, 1.0, 0.0),
        (2, 1.0, 0.5714),
        (2, 0.32, 0.8777),
        (5, 1.0, 0.4),
        (100, 1.0, 0.4),
        (5, 0.5, 0.4),
        (100, 4.0, 0.4)
    ]
)
def test_fscore_evaluator(eval_at, beta, expected):
    matches_ids = ['0', '1', '2', '3', '4']

    desired_ids = ['1', '0', '20', '30', '40']

    evaluator = FScoreEvaluator(eval_at=eval_at, beta=beta)
    assert evaluator.evaluate(actual=matches_ids, desired=desired_ids) == pytest.approx(expected, 0.001)


def test_fscore_evaluator_invalid_beta():
    with pytest.raises(AssertionError):
        FScoreEvaluator(beta=0)
