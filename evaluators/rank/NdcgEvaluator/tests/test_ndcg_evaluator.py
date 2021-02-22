import pytest
import random

from .. import NDCGEvaluator


@pytest.fixture(scope='function')
def evaluator():
    return NDCGEvaluator(eval_at=3)


@pytest.mark.repeat(10)
@pytest.mark.parametrize('string_keys', [False, True])
@pytest.mark.parametrize('actual, evaluator, expected',
                         [
                             ([(1, .9), (3, .8), (4, .7), (2, 0.)], NDCGEvaluator(eval_at=3, power_relevance=False), 1.0),
                             ([(1, .9), (3, .8), (4, .7), (2, 0.)], NDCGEvaluator(eval_at=3, power_relevance=True), 1.0),
                             ([(10, .9), (30, .8), (40, .7), (20, 0.)], NDCGEvaluator(eval_at=3, power_relevance=False), 0.0),
                             ([(10, .9), (30, .8), (40, .7), (20, 0.)], NDCGEvaluator(eval_at=3, power_relevance=True), 0.0),
                             ([(1, .9), (3, .8), (4, .7), (2, 0.)], NDCGEvaluator(eval_at=3, power_relevance=False,
                                                                                  is_relevance_score=False), 0.278),
                             ([(1, .9), (3, .8), (4, .7), (2, 0.)], NDCGEvaluator(eval_at=3, power_relevance=True,
                                                                                  is_relevance_score=False), 0.209),
                             ([(1, .0), (3, .1), (4, .2), (2, 0.3)], NDCGEvaluator(eval_at=3, power_relevance=False), 0.278),
                             ([(1, .0), (3, .1), (4, .2), (2, 0.3)], NDCGEvaluator(eval_at=3, power_relevance=True), 0.209),
                             ([(1, .0), (3, .1), (4, .2), (2, 0.3)], NDCGEvaluator(eval_at=3, power_relevance=False,
                                                                                   is_relevance_score=False), 1.0),
                             ([(1, .0), (3, .1), (4, .2), (2, 0.3)], NDCGEvaluator(eval_at=3, power_relevance=True,
                                                                                   is_relevance_score=False), 1.0)
                         ]
                         )
def test_evaluate(actual, evaluator, expected, string_keys):
    def _key_to_str(x):
        return str(x[0]), x[1]

    desired = [(1, .8), (3, .4), (4, .1), (2, 0.)]
    if string_keys:
        desired = list(map(_key_to_str, desired))
        actual = list(map(_key_to_str, actual))
    random.shuffle(actual)
    random.shuffle(desired)
    assert evaluator.evaluate(
        actual=actual,
        desired=desired,
    ) == pytest.approx(expected, 0.01)


@pytest.mark.parametrize('actual, desired', [
    ([], [(1, .8), (2, .4), (3, .1), (4, 0)]),  # actual is empty
    ([(1, .4), (2, .1), (3, .8)], []),  # desired is empty
    ([(1, .4), (2, .1), (3, .8)], [(1, .4), (2, .1), (3, -5)]),  # desired has negative value
])
def test_evaluate_fail(evaluator, actual, desired):
    with pytest.raises(ValueError):
        evaluator.evaluate(actual=actual, desired=desired)
