from jina.executors.evaluators.text import BaseTextEvaluator

class BleuEvaluator(BaseTextEvaluator):
    """
    :class:`BleuEvaluator`Bilingual Evaluation Understudy Score. 
    Evaluates the generated sentence (actual) against a desired sentence. 
    It will use the Bleu on NLTK package.
    A perfect match will score 1.0 and a perfect unmatch will score 0.0
    """

    def count_ngram(self, text, max_ngram):
        """Gets all n_grams from a given text
            - text: the text to be analyzed
            - max_ngram: max n_gram from the text
            - counter: dictionary of n-gram and its count
        """
        from collections import Counter

        counter = Counter()

        for n_order in range(1, max_ngram):
            for i in range(len(text) - n_order):
                ngram = tuple(text[i:i+n_order])
                counter[ngram] += 1
        
        return counter

    
    def get_nltk_bleu_score(self, desired, actual, overlap):
        import nltk.translate.bleu_score as bleu    
        from nltk.translate.bleu_score import SmoothingFunction

        #Check the n-gram and reset the weights accordingly
        overlaps = sum(overlap.values())
        if overlaps == 1:
            return bleu.sentence_bleu(desired, actual, weights = (1, 0, 0, 0), smoothing_function=SmoothingFunction().method4)
        if overlaps == 2:
            return bleu.sentence_bleu(desired, actual, weights = (0, 1, 0, 0), smoothing_function=SmoothingFunction().method4)
        if overlaps == 3:
            return bleu.sentence_bleu(desired, actual, weights = (0, 0, 1, 0), smoothing_function=SmoothingFunction().method4)
        if overlaps == 4:
            return bleu.sentence_bleu(desired, actual, weights = (0, 0, 0, 1), smoothing_function=SmoothingFunction().method4)
        else:
            return bleu.sentence_bleu(desired, actual) #otherwise use standard weights (0.25, 0.25, 0.25, 0.25) and no Smoothin function


    def evaluate(self,
            actual,
            desired,
            max_order=4,
            smooth=False,
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

        #get the n_gram for each sentence
        desired_ngram = self.count_ngram(desired_list, max_order+1)
        actual_ngram = self.count_ngram(actual_list, max_order+1)

        #get overlaps betwen the two
        overlaps = desired_ngram & actual_ngram
        
        return self.get_nltk_bleu_score([desired_list], actual_list, overlaps)
    
