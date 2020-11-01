import pytest

from .. import ReciprocalRankEvaluator


@pytest.mark.parametrize('actual, desired, score',
[
    ([],[], 0.0),
    (['hey', 'hello', 'hi', 'heya'],[], 0.0),
    ([],['hey', 'hello', 'hi', 'heya'], 0.0),
    (['hey', 'hello', 'hi', 'heya'],['hey', 'hello', 'hi', 'heya'], 1.0),
    (['hey', 'hello', 'hi', 'heya'],['hello', 'hi', 'hey', 'heya'], 0.5),
    (['hey', 'hello', 'hi', 'heya'],['helloo', 'hi', 'hey', 'heya'], 0.0),
]
)
def test_reciprocalrankevaluator(actual, desired, score):
    evaluator = ReciprocalRankEvaluator(eval_at=3)
    assert evaluator.evaluate(actual, desired) == score