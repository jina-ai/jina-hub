from jina.executors.evaluators import BaseEvaluator
from typing import Sequence, Any
import nltk.translate.bleu_score as bleu
import numpy as np
from nltk import ngrams
from collections import Counter



class BleuEvaluator(BaseEvaluator):
    """
    :class:`BleuEvaluator`Bilingual Evaluation Understudy Score. 
    Evaluates the generated sentence (hypothesis) against a reference sentence. 
    It will use the Bleu on NLTK package.
    A perfect match will score 1.0 and a perfect unmatch will score 0.0
    """
    
    
    def get_nltk_bleu_score(self, hypothesis, reference):
        return bleu.sentence_bleu(reference, hypothesis)


    def evaluate(self,
            hypothesis,
            reference,
            nltk_default = True,
            *args,
            **kwargs) -> float:
        """"
        :param actual: the scores predicted by the search system.
        :param desired: the expected score given by user as groundtruth.
        :return the evaluation metric value for the request document.
        """

        BLEUScore = self.get_nltk_bleu_score(hypothesis, reference)

        return BLEUScore
    
