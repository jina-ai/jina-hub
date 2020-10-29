from .. import BleuEvaluator



def test_bleuevaluator():
    BLEU_eval = BleuEvaluator()
    hypothesis = ['All', 'cats', 'are', 'beautiful']
    reference = ['All', 'cats', 'are', 'beautiful']
    BLUE_score = BLEU_eval.evaluate(hypothesis, reference, True)
    assert BLUE_score == 1.0
    





