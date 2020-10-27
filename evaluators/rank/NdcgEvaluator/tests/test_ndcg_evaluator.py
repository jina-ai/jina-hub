import pytest

from .. import NDCGEvaluator

@pytest.fixture(scope='function')
def evaluator():
    return NDCGEvaluator(eval_at=3)

@pytest.fixture(scope='function')
def desired():
    return [.8, .4, .1]

@pytest.mark.parametrize('actual, expected',[
    ([.4, .1, .8], 0.795)
    ([.0, .1, .4], 0.279)
    ([.4, .1, .0], 0.396)
])
def test_encode_success(evaluator, actual, desired, expected):
    assert evaluator.evaluate(actual=actual, desired=desired) == pytest.approx(expected)

def test_encode_fail(evaluator, desired):
    assert pytest.raises(ValueError, evaluator.evaluate(actual=[0.8], desired=desired))