from .. import BleuEvaluator
import pytest


@pytest.mark.parametrize('actual, desired, score',
    [
        ('All cats are so super beautiful', 'All cats are so super beautiful', 1.0),
        ('All cats are so super beautiful', '', 0.0),
        ('', 'All cats are so super beautiful', 0.0),
        ('All cats are so super beautiful', 'ALL CATS ARE SO SUPER BEAUTIFUL', 1.0), 
        ('All cats are so super beautiful', 'Some dogs are also cute memes', 0.16),
        ('All cats are so super beautiful', 'Why is there an unicorn here', 0.0)
    ])


def test_bleu_evaluator(actual, desired, score):
    evaluator = BleuEvaluator()
    assert evaluator.evaluate(actual, desired) == pytest.approx(score, 0.1)





