import pytest

from .. import HammingDistanceEvaluator


@pytest.mark.parametrize('actual, desired, distance',
    [
        ('hello', 'hello', 0.0),
        ('Hello', 'hello', 1.0),
        ('eello', 'hello', 1.0),
        ('he!@#', 'hello', 3.0),
        ('gdkkm', 'hello', 5.0),
        ('g s m', ' s j ', 5.0),
        ('^@*&$', ':<>?)', 5.0),
        ('@,.&$', '}|.?)', 4.0),
        ('', '', 0.0)
    ]
)
def test_hammingdistanceevaluator(actual, desired, distance):
    evaluator = HammingDistanceEvaluator()
    assert evaluator.evaluate(actual, desired) == distance

@pytest.mark.parametrize('actual, desired',
    [
        ('', 'hello'),       # actual is empty
        ('hello', ''),       # desired is empty
        ('hello', 'helloo')  # not equal length
    ]
)

def test_hammingdistanceevaluator_fail(actual, desired):
    evaluator = HammingDistanceEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(actual, desired)