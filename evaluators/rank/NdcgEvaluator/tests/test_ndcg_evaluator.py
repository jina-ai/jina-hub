import pytest

from .. import NDCGEvaluator

@pytest.fixture(scope='function')
def evaluator():
    return NDCGEvaluator(eval_at=3)

@pytest.mark.parametrize('actual, evaluator, use_traditional_formula, expected',
    [
        ([.4, .1, .8], NDCGEvaluator(eval_at=3), True, 0.795),
        ([.0, .1, .4], NDCGEvaluator(eval_at=3), True, 0.279),
        ([.4, .1, .0], NDCGEvaluator(eval_at=3), True, 0.396),
        ([.4, .1, .8], NDCGEvaluator(eval_at=3), False, 0.751),
        ([.0, .1, .4], NDCGEvaluator(eval_at=3), False, 0.209),
        ([.4, .1, .0], NDCGEvaluator(eval_at=3),  False, 0.373),
        ([.0, .0, .0], NDCGEvaluator(eval_at=3), True, 0.0), # no match
        ([.0, .0, .0], NDCGEvaluator(eval_at=3), False, 0.0),
        ([.8, .4, .1], NDCGEvaluator(eval_at=3), True, 1.0), # all match
        ([.8, .4, .1], NDCGEvaluator(eval_at=3), False, 1.0),
        ([.4, .1, .8], NDCGEvaluator(eval_at=2), False, 0.387),
        ([.4, .1, .8], NDCGEvaluator(eval_at=2), True, 0.417),
        ([.4, .1, .8], NDCGEvaluator(eval_at=5), True, 0.795),
    ]
)
def test_encode(actual, evaluator, use_traditional_formula, expected):
    assert evaluator.evaluate(
        actual=actual,
        desired=[.8, .4, .1, 0],
        use_traditional_formula=use_traditional_formula
    ) == pytest.approx(expected, 0.01)

@pytest.mark.parametrize('actual', [
    ([]), ([0.8])
])
def test_encode_fail(evaluator, actual):
    with pytest.raises(ValueError):
        evaluator.evaluate(actual=actual, desired=[.8, .4, .1, 0])