import pytest

from .. import NDCGEvaluator


@pytest.fixture(scope='function')
def evaluator():
    return NDCGEvaluator(eval_at=3)


@pytest.mark.parametrize('actual, evaluator, expected',
                         [
                             ([.4, .1, .8], NDCGEvaluator(eval_at=3, power_relevance=False), 0.795),
                             ([.0, .1, .4], NDCGEvaluator(eval_at=3, power_relevance=False), 0.279),
                             ([.4, .1, .0], NDCGEvaluator(eval_at=3, power_relevance=False), 0.396),
                             ([.4, .1, .8], NDCGEvaluator(eval_at=3, power_relevance=True), 0.751),
                             ([.0, .1, .4], NDCGEvaluator(eval_at=3, power_relevance=True), 0.209),
                             ([.4, .1, .0], NDCGEvaluator(eval_at=3, power_relevance=True), 0.373),
                             ([.0, .0, .0], NDCGEvaluator(eval_at=3, power_relevance=False), 0.0),  # no match
                             ([.0, .0, .0], NDCGEvaluator(eval_at=3, power_relevance=False), 0.0),
                             ([.8, .4, .1], NDCGEvaluator(eval_at=3, power_relevance=False), 1.0),  # all match
                             ([.8, .4, .1], NDCGEvaluator(eval_at=3, power_relevance=True), 1.0),
                             ([.4, .1, .8], NDCGEvaluator(eval_at=2, power_relevance=True), 0.387),
                             ([.4, .1, .8], NDCGEvaluator(eval_at=2, power_relevance=False), 0.417),
                             ([.4, .1, .8], NDCGEvaluator(eval_at=5, power_relevance=False), 0.795),
                         ]
                         )
def test_encode(actual, evaluator, expected):
    assert evaluator.evaluate(
        actual=actual,
        desired=[.8, .4, .1, 0],
    ) == pytest.approx(expected, 0.01)


@pytest.mark.parametrize('actual, desired', [
    ([], [.8, .4, .1, 0]),  # actual is empty
    ([.4, .1, .8], []),  # desired is empty
    ([.4, .1, -1], []),  # actual has negative value
    ([], [.8, .4, .1, -1]),  # desired has negative value
])
def test_encode_fail(evaluator, actual, desired):
    with pytest.raises(ValueError):
        evaluator.evaluate(actual=actual, desired=desired)
