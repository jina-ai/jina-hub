import pytest

from .. import NDCGEvaluator

@pytest.fixture(scope='function')
def evaluator():
    return NDCGEvaluator(eval_at=3)

@pytest.mark.parametrize('actual, use_traditional_formula, expected',
    [
        ([.4, .1, .8], True, 0.795),
        ([.0, .1, .4], True, 0.279),
        ([.4, .1, .0], True, 0.396),
        ([.4, .1, .8], False, 0.795),
        ([.0, .1, .4], False, 0.210),
        ([.4, .1, .0], False, 0.396),
        ([.0, .0, .0], True, 0.0), # no match
        ([.0, .0, .0], False, 0.0),
        ([.8, .4, .1], True, 1.0), # all match
        ([.8, .4, .1], False, 1.0),
    ]
)
def test_encode(evaluator, actual, use_traditional_formula, expected):
    assert evaluator.evaluate(actual=actual, desired=[.8, .4, .1, 0], use_traditional_formula=use_traditional_formula) == pytest.approx(expected, 0.1)

def test_encode_fail(evaluator):
    with pytest.raises(ValueError):
        evaluator.evaluate(actual=[0.8], desired=[.8, .4, .1, 0])