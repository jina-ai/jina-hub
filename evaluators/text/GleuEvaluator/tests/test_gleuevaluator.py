from .. import GleuEvaluator
import pytest


@pytest.mark.parametrize(
    "actual, desired, score",
    [
        ("I like watching soccer on tv", "I like watching soccer on tv", 1.0),
        ("I like watching soccer on tv", "Who let the dogs out?", 0.0),
        ("I like watching soccer on tv", "", 0.0),
        ("", "I like watching soccer on tv", 0.0),
        ("All cats are so super beautiful", "ALL CATS ARE SO SUPER BEAUTIFUL", 1.0),
        ("i like to watch cricket", "i like to watch soccer", 0.7),
        ("i like to play soccer", "i like to watch cricket", 0.4),
        ("All cats are so super beautiful", "Why is there an unicorn here", 0.0),
        ("A small cat", "A small cat", 1.0),
        ("Even smaller", "Even smaller", 1.0),
    ],
)
def test_gleu_evaluator(actual, desired, score):
    evaluator = GleuEvaluator()
    assert evaluator.evaluate(actual, desired) == pytest.approx(score, 0.1)

