__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import nltk
import itertools
from nltk.collocations import BigramCollocationFinder
import re
from editdistance import eval as edit_distance
from collections import Counter


class PyNgramSpell():


    def __init__(self,
                 ngram_range=(1,2),
                 tokenizer=None,
                 vocabulary={},
                 string_preprocessor_func=str.lower,
                 token_pattern=r"(?u)\b\w\w+\b",
                 lambda_interpolation=0.3):

        self.ngram_range = ngram_range
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.string_preprocessor_func = string_preprocessor_func
        self.token_pattern = token_pattern
        self.lambda_interpolation = lambda_interpolation

    def build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens.
        tokenizer: callable
              A function to split a string into a sequence of tokens.
        """
        if self.tokenizer is not None:
            return self.tokenizer
        token_pattern = re.compile(self.token_pattern)

        if token_pattern.groups > 1:
            raise ValueError(
                "More than 1 capturing group in token pattern. Only a single "
                "group should be captured."
            )

        return token_pattern.findall

    def fit(self, X):
        """
        X iterable of strings (corpus)        
        """
        self.tokenize_func = self.build_tokenizer()
        X_tokenized = [self.tokenize_func(self.string_preprocessor_func(x)) for x in X]
        self.unigram_freq_dict = dict(Counter(itertools.chain(*X_tokenized)))

        bigram_finder = BigramCollocationFinder.from_documents(X_tokenized)
        self.bigram_freq_dict = dict(bigram_finder.ngram_fd.items())
        self.vocabulary = set(list(itertools.chain(*self.bigram_freq_dict.keys())))

    
    def get_candidates(self, token, max_dist):
        distance_token_to_words = {word:edit_distance(word,token.lower()) for word in self.vocabulary}
        minimum_distance = min(distance_token_to_words.values())
        if minimum_distance < max_dist:
            return sorted([word for word, distance in distance_token_to_words.items() if distance == minimum_distance])
        return [token]

    def filter_vocabulary(self, min_freq):
        self.vocabulary = set(dict(filter(lambda x:x[1]>min_freq, self.unigram_freq_dict.items())).keys())


    def correct_with_bigrams(self, tokenized_sentence):

        def prob_word(word):
            return self.unigram_freq_dict.get(word,0) / len(self.unigram_freq_dict)

        def bigrams_starting_by(word): 
            return [t for t in list(self.bigram_freq_dict.keys()) if t[0] == word]

        def count_bigrams(list_bigrams): 
            return sum([ self.bigram_freq_dict.get(bigram, 0) for bigram in list_bigrams])

        def probability_bigram(word1, word2, bigram_freq_dict):
            bigram = ((word1,word2))
            if self.bigram_freq_dict.get(bigram, 0)  == 0:
                return 0
            else:
                return self.bigram_freq_dict.get(bigram, 0)/count_bigrams(bigrams_starting_by(word1))
        
        def interpolation_probability(word1, word2): 
            n_vocabulary = len(self.vocabulary)
            return (1-self.lambda_interpolation) * probability_bigram(word1, word2, self.bigram_freq_dict) +\
                    self.lambda_interpolation * prob_word(word2)

        for index,word in enumerate(tokenized_sentence):
            if word not in self.vocabulary:
                if index == 0: 
                    previous_word = '.'
                else:
                    previous_word = tokenized_sentence[index-1]
                candidates = {candidate:interpolation_probability(previous_word, candidate) for candidate in self.get_candidates(word, max_dist=2)}
                
                tokenized_sentence[index] = max(candidates, key=candidates.get)
        
        return tokenized_sentence


    def transform(self, x, tokenize=True):
        """Corrects the misspelled words on each element from X.
        X iterable of strings        
        """
        x = self.string_preprocessor_func(x)

        if tokenize:
            tokenized_sentence = self.tokenize_func(x)
        else:
            tokenized_sentence = x.copy()

        tokenized_sentence = self.correct_with_bigrams(tokenized_sentence)

        return tokenized_sentence


    def save(self, path):
        """Save the instance an PyNgramSpell to the provided path.
        """
        pass
