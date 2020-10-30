from .. import BleuEvaluator
import pytest


def test_bleu_evaluator():
    desired = 'All cats are so super beautiful'
    actual = 'All cats are so super beautiful'
    assert BleuEvaluator().evaluate(actual, desired) == 1.0

def test_bleu_evaluator_actual_emtpy():
    desired = 'All cats are so super beautiful'
    actual = ''
    assert BleuEvaluator().evaluate(actual, desired) == 0.0

def test_bleu_evaluator_desired_emtpy():
    desired = ''
    actual = 'All cats are so super beautiful'
    assert BleuEvaluator().evaluate(actual, desired) == 0.0

def test_bleu_evaluator_caps():
    desired = 'All cats are so super beautiful'
    actual = 'ALL CATS ARE SO SUPER BEAUTIFUL'
    assert BleuEvaluator().evaluate(actual, desired) == 1.0

def test_bleu_evaluator_fail():
    desired = 'All cats are so super beautiful'
    actual = 'Some dogs are also cute memes'
    with pytest.raises(Exception):
        assert BleuEvaluator().evaluate(actual, desired) == 1.0

