from .. import BleuEvaluator
import pytest


def test_bleu_evaluator():
    desired = 'All cats are so super beautiful'
    actual = 'All cats are so super beautiful'
    assert BleuEvaluator().evaluate(desired, actual) == 1.0






