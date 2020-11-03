from .. import BleuEvaluator
import pytest


@pytest.mark.parametrize('actual, desired, score',
    [
        ('All cats are so super beautiful', 'All cats are so super beautiful', 1.0),
        ('All cats are so super beautiful', '', 0.0),
        ('', 'All cats are so super beautiful', 0.0),
        ('All cats are so super beautiful', 'ALL CATS ARE SO SUPER BEAUTIFUL', 1.0), 
        ('All cats are so super beautiful', 'Some dogs are also cute memes', 0.16),
        ('All cats are so super beautiful', 'Why is there an unicorn here', 0.0),
        ('A small cat', 'A small cat', 1.0),
        ('Even smaller', 'Even smaller', 1.0)
    ])


def test_bleu_evaluator(actual, desired, score):
    evaluator = BleuEvaluator()
    assert evaluator.evaluate(actual, desired) == pytest.approx(score, 0.1)

    # Without checking n-gram the results are not always correct
    # for this example, the score should be less than 1
    # since the two sentences are not the same
    # but without reseting the weights, the result is bigger than 1.0
    fail_actual = 'All cats are so super beautiful'
    fail_desired = 'Some dogs are also cute memes'
    assert evaluator.get_nltk_bleu_score([fail_desired.lower().split()], fail_actual.lower().split(), 0) < 1.0 






