from typing import List

from jina.executors.evaluators.text import BaseTextEvaluator


class GleuEvaluator(BaseTextEvaluator):
    """
    :class:`GleuEvaluator` Evaluate GLEU score between acutal and ground truth.
    It will use the Gleu on NLTK package.
    A perfect match will score 1.0 and a complete mismatch will score 0.0
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_score(actual_list: List[str], desired_list: List[List[str]], n_gram: int):
        """
        :param desired_list: A List of a List of all possible sentences ex: [['cats are cute'], ['dogs are cute']]
        :param actual_list: A list of the sentences to be scored ex: ['cats are cute']
        :param n_gram: is the gram size ex: ['cats are cute'] -> n_gram = 3
        return gives a float of the sentence-level GLEU score
        """
        import nltk.translate.gleu_score as gleu

        if n_gram <= 4:
            return gleu.sentence_gleu(desired_list, actual_list, max_len=n_gram)
        else:
            return gleu.sentence_gleu(desired_list, actual_list)  # if the ngram is at least 5, use the standard

    def evaluate(self, actual, desired, *args, **kwargs) -> float:
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
