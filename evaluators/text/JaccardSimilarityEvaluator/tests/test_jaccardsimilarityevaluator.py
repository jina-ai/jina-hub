import pytest

from .. import JaccardSimilarityEvaluator


@pytest.mark.parametrize('actual, desired, distance',
    [
        ('hello world', 'hello world', 1.0),
        ('HELLO WORLD', 'hello world', 1.0),
        ('hello world', 'hello      ', 0.5),
        ('hello      ', 'hello world', 0.5),
        ('gdkkm      ', 'hello      ', 0.0),
        ('hey yup man', 'whats up man', 0.2),
        ('hello', '', 0.0),
        ('', 'hello', 0.0),
        ('', '', 0.0)
    ]
)
def test_jaccardsimilarityevaluator(actual, desired, distance):
    evaluator = JaccardSimilarityEvaluator()
    assert evaluator.evaluate(actual, desired) == pytest.approx(distance)
