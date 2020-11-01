import pytest

from .. import RougeEvaluator


@pytest.mark.parametrize('metrics, stats, actual, desired, score',
    [
        ('rouge-1', 'r', 'hello', 'hello', 1.0),
        ('rouge-1', 'r', 'hello', 'Hello', 0.0),
        ('rouge-1', 'r', 'hello world', 'hello', 1.0),
        ('rouge-1', 'r', 'hello world', 'Hello', 0.0),
        ('rouge-1', 'r', 'hello', 'hello world', 0.5),
        ('rouge-1', 'r', 'hello', 'Hello world', 0.0),
        ('rouge-1', 'r', 'hello world jina was here', 'Hello world jina is not here', 0.5),
        ('rouge-1', 'r', 'Hello world jina is not here', 'hello world jina was here', 0.6),
        ('rouge-1', 'r', 'Hello world jina is not here', 'hello world jina was here!', 0.4),
        ('rouge-1', 'r', '', '', 0.0),
        ('rouge-1', 'r', 'hello', 'helo', 0.0),
        ('rouge-1', 'r', 'hello', '', 0.0),
        ('rouge-1', 'r', '', 'hello', 0.0),

        ('ROUGE-1', 'R', 'hello', 'hello', 1.0),

        ('rouge-1', 'p', 'hello', 'hello', 1.0),
        ('rouge-1', 'p', 'hello', 'Hello', 0.0),
        ('rouge-1', 'p', 'hello world', 'hello', 1.0),
        ('rouge-1', 'p', 'hello world', 'Hello', 0.0),
        ('rouge-1', 'p', 'hello', 'hello world', 0.5),
        ('rouge-1', 'p', 'hello', 'Hello world', 0.0),
        ('rouge-1', 'p', 'hello world jina was here', 'Hello world jina is not here', 0.5),
        ('rouge-1', 'p', 'Hello world jina is not here', 'hello world jina was here', 0.6),
        ('rouge-1', 'p', 'Hello world jina is not here', 'hello world jina was here!', 0.4),
        ('rouge-1', 'p', '', '', 0.0),
        ('rouge-1', 'p', 'hello', 'helo', 0.0),
        ('rouge-1', 'p', 'hello', '', 0.0),
        ('rouge-1', 'p', '', 'hello', 0.0),

        ('rouge-1', 'f', 'hello', 'hello', 1.0),
        ('rouge-1', 'f', 'hello', 'Hello', 0.0),
        ('rouge-1', 'f', 'hello world', 'hello', 1.0),
        ('rouge-1', 'f', 'hello world', 'Hello', 0.0),
        ('rouge-1', 'f', 'hello', 'hello world', 0.5),
        ('rouge-1', 'f', 'hello', 'Hello world', 0.0),
        ('rouge-1', 'f', 'hello world jina was here', 'Hello world jina is not here', 0.5),
        ('rouge-1', 'f', 'Hello world jina is not here', 'hello world jina was here', 0.6),
        ('rouge-1', 'f', 'Hello world jina is not here', 'hello world jina was here!', 0.4),
        ('rouge-1', 'f', '', '', 0.0),
        ('rouge-1', 'f', 'hello', 'helo', 0.0),
        ('rouge-1', 'f', 'hello', '', 0.0),
        ('rouge-1', 'f', '', 'hello', 0.0),

        ('rouge-2', 'r', 'hello', 'hello', 1.0),
        ('rouge-2', 'r', 'hello', 'Hello', 0.0),
        ('rouge-2', 'r', 'hello world', 'hello', 1.0),
        ('rouge-2', 'r', 'hello world', 'Hello', 0.0),
        ('rouge-2', 'r', 'hello', 'hello world', 0.5),
        ('rouge-2', 'r', 'hello', 'Hello world', 0.0),
        ('rouge-2', 'r', 'hello world jina was here', 'Hello world jina is not here', 0.5),
        ('rouge-2', 'r', 'Hello world jina is not here', 'hello world jina was here', 0.6),
        ('rouge-2', 'r', 'Hello world jina is not here', 'hello world jina was here!', 0.4),
        ('rouge-2', 'r', '', '', 0.0),
        ('rouge-2', 'r', 'hello', 'helo', 0.0),
        ('rouge-2', 'r', 'hello', '', 0.0),
        ('rouge-2', 'r', '', 'hello', 0.0),

        ('rouge-l', 'r', 'hello', 'hello', 1.0),
        ('rouge-l', 'r', 'hello', 'Hello', 0.0),
        ('rouge-l', 'r', 'hello world', 'hello', 1.0),
        ('rouge-l', 'r', 'hello world', 'Hello', 0.0),
        ('rouge-l', 'r', 'hello', 'hello world', 0.5),
        ('rouge-l', 'r', 'hello', 'Hello world', 0.0),
        ('rouge-l', 'r', 'hello world jina was here', 'Hello world jina is not here', 0.5),
        ('rouge-l', 'r', 'Hello world jina is not here', 'hello world jina was here', 0.6),
        ('rouge-l', 'r', 'Hello world jina is not here', 'hello world jina was here!', 0.4),
        ('rouge-l', 'r', '', '', 0.0),
        ('rouge-l', 'r', 'hello', 'helo', 0.0),
        ('rouge-l', 'r', 'hello', '', 0.0),
        ('rouge-l', 'r', '', 'hello', 0.0)
    ]
)
def test_rougeevaluator(metrics, stats, actual, desired, score):
    evaluator = RougeEvaluator(metrics=metrics, stats=stats)
    assert evaluator.evaluate(actual, desired) == score
