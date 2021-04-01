__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import List

from jina.executors.evaluators.text import BaseTextEvaluator


class BleuEvaluator(BaseTextEvaluator):
    """
    :class:`BleuEvaluator`Bilingual Evaluation Understudy Score. 
    Evaluates the generated sentence (actual) against a desired sentence. 
    It will use the Bleu on NLTK package.
    A perfect match will score 1.0 and a complete mismatch will score 0.0

    The NLTK library can score n-gram individually or cummulative.
    Here we use the cumulative as it is more precise.
    https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_score(actual_list: List[str], desired_list: List[List[str]], n_gram: int):
        """
        :param desired_list: A List of a List of all possible sentences ex: [['cats are cute'], ['dogs are cute']]
        :param actual_list: A list of the sentences to be scored ex: ['cats are cute']
        :param n_gram: is the gram size ex: ['cats are cute'] -> n_gram = 3
        return gives a float of the sentence-level BLEU score
        Cumulative score is the calculation of individual n-grams
        from 1 to n-order, and then weights them with the geometric mean
        It will check the n-gram size, if the n-gram is smaller than 4,
        which is the standard for NLTK, it is necessary to reset the weights
        """
        import nltk.translate.bleu_score as bleu
        from nltk.translate.bleu_score import SmoothingFunction

        if n_gram == 1:
            return bleu.sentence_bleu(desired_list, actual_list, weights=(1.0, 0, 0, 0),
                                      smoothing_function=SmoothingFunction().method1)
        elif n_gram == 2:
            return bleu.sentence_bleu(desired_list, actual_list, weights=(0.5, 0.5, 0, 0),
                                      smoothing_function=SmoothingFunction().method2)
        elif n_gram == 3:
            return bleu.sentence_bleu(desired_list, actual_list, weights=(0.33, 0.33, 0.33, 0),
                                      smoothing_function=SmoothingFunction().method3)
        else:
            return bleu.sentence_bleu(desired_list, actual_list)  # if the ngram is at least 4, use the standard

    def evaluate(self,
                 actual,
                 desired,
                 *args,
                 **kwargs) -> float:
        """"
        :param desired: the expected text given by user as groundtruth.
        :param actual: the text predicted by the search system.
        :return the evaluation metric value for the request document.

        NLTK expectes an array of strings, 
        so the incoming string needs to be tokenized first.
        They will be stored in a desired_list and actual_list accordingly
        """

        # set everything to undercase and tokenize
        desired_list = desired.lower().split()
        actual_list = actual.lower().split()

        return self.get_score(actual_list, [desired_list], len(actual_list))
