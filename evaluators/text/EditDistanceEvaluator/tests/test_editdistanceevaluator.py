import pytest

from .. import EditDistanceEvaluator


@pytest.mark.parametrize('actual, desired, distance',
                         [
                             ('hello', 'hello', 0.0),
                             ('Hello', 'hello', 1.0),
                             ('1Hello1', 'hello', 3.0),
                             ('hello!@#', 'hello', 3.0),
                             ('hell!o', 'hello', 1.0),
                             ('', '', 0.0),
                             ('hello', 'helo', 1.0),
                             ('hello', '', 5.0),
                             ('', 'hello', 5.0)
                         ]
                         )
def test_editdistanceevaluator(actual, desired, distance):
    evaluator = EditDistanceEvaluator()
    assert evaluator.evaluate(actual, desired) == distance
