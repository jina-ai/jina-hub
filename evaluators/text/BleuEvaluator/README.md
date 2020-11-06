# BleuEvaluator

Bilingual Evaluation Understudy Score. Evaluates the generated sentence against a desired sentence. 
A perfect match will score 1.0 and a complete mismatch will score 0.0

BLEU works well with n-grams of at least 4, if less than that, it's neccessary to use a smoothing function and reset the weights.
We use accumulative-error to check the n-grams and set the weights accordingly.

To read more about this please check this websites:
https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
https://towardsdatascience.com/bleu-bilingual-evaluation-understudy-2b4eab9bcfd1
https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213
