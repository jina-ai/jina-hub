from jina.executors.evaluators.text import BaseTextEvaluator

class BleuEvaluator(BaseTextEvaluator):
    """
    :class:`BleuEvaluator`Bilingual Evaluation Understudy Score. 
    Evaluates the generated sentence (actual) against a desired sentence. 
    It will use the Bleu on NLTK package.
    A perfect match will score 1.0 and a perfect unmatch will score 0.0
    """
    
    def get_nltk_bleu_score(self, desired, actual):
        import nltk.translate.bleu_score as bleu    
        from nltk.translate.bleu_score import SmoothingFunction

        return bleu.sentence_bleu(desired, actual, smoothing_function=SmoothingFunction().method4)


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
        
        return self.get_nltk_bleu_score([desired_list], actual_list)
    
