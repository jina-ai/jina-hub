import nltk.translate.bleu_score as bleu

from jina.executors.evaluators import BaseEvaluator

class BleuEvaluator(BaseEvaluator):
    """
    :class:`BleuEvaluator`Bilingual Evaluation Understudy Score. 
    Evaluates the generated sentence (actual) against a desired sentence. 
    It will use the Bleu on NLTK package.
    A perfect match will score 1.0 and a perfect unmatch will score 0.0
    """
    
    
    def get_nltk_bleu_score(self, desired, actual):
        return bleu.sentence_bleu(desired, actual)


    def evaluate(self,
            desired,
            actual, 
            *args,
            **kwargs) -> float:
        """"
        :param desired: the expected score given by user as groundtruth.
        :param actual: the scores predicted by the search system.
        :return the evaluation metric value for the request document.

        NLTK expectes an array of strings, 
        so the incoming string needs to be tokenized first.
        They will be stored in a desired_list
        """
        
        # tokenize
        desired_list = desired.split()
        actual_list = actual.split()
        
        return self.get_nltk_bleu_score([desired_list], actual_list)
    
