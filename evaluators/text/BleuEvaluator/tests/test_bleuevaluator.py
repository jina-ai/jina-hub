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

def test_without_reset_weight():
    '''
    This 2 sentences are not very similar, they should have a score close to 0 but bigger than 0
    If we set the weights correctly, we have a score of around 0.16 as in the past test
    but if we don'd do that and harcode the n-gram to 4 or more
    the score we get is smaller than 0
    This shouldn't be the case since the scale in NLTK is from 0 - 1
    but since the weights are not set correctly
    we get a warning invovling "ZeroDivisionError: Fraction(0, 0)"
    '''
    evalualtor = BleuEvaluator()
    actual_list = 'All cats are so super beautiful'.lower().split()
    desired_list = 'Some dogs are also cute memes'.lower().split()
    assert evalualtor.get_score([desired_list], actual_list, 4) < 0




