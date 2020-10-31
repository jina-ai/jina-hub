import pytest

from .. import RougeEvaluator


@pytest.mark.parametrize('actual, desired, score',
    [
        ('hello', 'hello', 1.0),
        ('hello', 'Hello', 0.0),
        ('hello world', 'hello', 1.0),
        ('hello world', 'Hello', 0.0),
        ('hello', 'hello world', 0.5),
        ('hello', 'Hello world', 0.0),
        ('', '', 0.0),
        ('hello', 'helo', 0.0),
        ('hello', '', 0.0),
        ('', 'hello', 0.0)
    ]
)
def test_rougeevaluator(actual, desired, score):
    evaluator = RougeEvaluator(metrics='rouge-1', stats='r')
    assert evaluator.evaluate(actual, desired) == score
