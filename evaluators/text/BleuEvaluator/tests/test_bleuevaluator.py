__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import pytest

from .. import BleuEvaluator


@pytest.mark.parametrize('actual, desired, score',
                         [
                             ('All cats are so super beautiful', 'All cats are so super beautiful', 1.0),
                             ('All cats are so super beautiful', '', 0.0),
                             ('', 'All cats are so super beautiful', 0.0),
                             ('All cats are so super beautiful', 'ALL CATS ARE SO SUPER BEAUTIFUL', 1.0),
                             ('All cats are so super beautiful', 'All dogs are so super beautiful', 0.5),
                             ('All cats are so super beautiful', 'Why is there an unicorn here', 0.0),
                             ('A small cat', 'A small cat', 1.0),
                             ('Even smaller', 'Even smaller', 1.0)
                         ])
def test_bleu_evaluator(actual, desired, score):
    evaluator = BleuEvaluator()
    assert evaluator.evaluate(actual, desired) == pytest.approx(score, 0.1)


def test_without_reset_weight():
    """
    This 2 sentences are not very similar, they should have a score close to, but bigger than 0
    If we set the weights correctly, we have a score of around 0.5 as in the past test
    but if we don'd do that and harcode the n-gram to 4 or more
    the score is around 0.5 which shouldn't be since they are the same sentence.
    Also for some cases, it can have values less than 0
    This shouldn't be the case since the scale in NLTK is from 0 - 1
    but since the weights are not set correctly
    we get a warning invovling ZeroDivisionError: Fraction(0, 0)
    We need to reset the scores because the n-gram is 3 in this case.
    """
    evaluator = BleuEvaluator()
    actual_list = 'A small cat'.lower().split()
    desired_list = 'A small cat'.lower().split()
    assert evaluator.get_score(actual_list, [desired_list], 4) < 1
