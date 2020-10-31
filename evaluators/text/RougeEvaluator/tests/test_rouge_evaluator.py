import pytest

from .. import RougeEvaluator


@pytest.mark.parametrize('actual, desired, score',
    [
        (['hello'], ['hello'], 1.0),
        (['hello world'], ['hello'], 1.0),
        (['hello'], ['hello world'], 0.5),
        ([''], [''], 0.0),
        ([], [], 0.0),
        (['hello'], ['helo'], 0.0),
        (['hello'], [''], 0.0),
        ([''], ['hello'], 0.0),
        (['hello', 'hello'], ['hello world', 'helo'], 0.5)
    ]
)
def test_rougeevaluator(actual, desired, score):
    evaluator = RougeEvaluator(metrics='rouge-1', stats='r')
    assert evaluator.evaluate(actual, desired) == score
