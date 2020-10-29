from .. import BleuEvaluator
import pytest


def test_bleu_evaluator():
    reference = [ ['All', 'cats', 'are', 'beautiful'], ['All', 'cats', 'are', 'cute'] ]
    hypothesis = ['All', 'cats', 'are', 'beautiful']
    BLUE_score = BleuEvaluator().evaluate(hypothesis, reference, True)
    assert BLUE_score == 1.0







