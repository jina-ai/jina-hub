__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import itertools
import re
from typing import Callable, List, Tuple, Optional, Iterable
from collections import Counter


class BKTree:
    """
    :class:`BKTree` allows fast search of words up to n distances away

    :param distfn : Function to compute distances between words, like edit distance function.
    :param words : iterable of words to build the tree.
    :param sort_candidates : If true results will be sorted by distance.
    """

    def __init__(self, distfn: Callable, words: List[str], sort_candidates: bool = False):
        self.distfn = distfn
        self.sort_candidates = sort_candidates

        it = iter(words)
        root = next(it)
        self.tree = (root, {})

        for i in it:
            self._add_word(self.tree, i)

    def _add_word(self, parent, word):
        pword, children = parent
        d = self.distfn(word, pword)
        if d in children:
            self._add_word(children[d], word)
        else:
            children[d] = (word, {})

    def _search_descendants(self, parent, n, distance, query_word):
        """
        :param parent : parent in the tree
        :param n : maximum edit distance allowed.
        :param distance : function used to compute the edit distance.
        :param query_word : query word
        """
        node_word, children_dict = parent
        dist_to_node = distance(query_word, node_word)
        self.visited_nodes.append(node_word)
        results = []
        if dist_to_node <= n:
            results.append((dist_to_node, node_word))

        for i in range(dist_to_node - n, dist_to_node + n + 1):
            child = children_dict.get(i)
            if child is not None:
                results.extend(self._search_descendants(child, n, distance, query_word))

        return results

    def query(self, query_word: str, n: int, return_distances: bool = False):
        """
        :param query_word : anchor word used to find candidates.
        :param n : maximum edit distance allowed
        :param return_distances : whether to return distances (True means return distance)
        """
        self.visited_nodes = []

        distance_candidate_list = self._search_descendants(self.tree, n, self.distfn, query_word)
        if self.sort_candidates:
            distance_candidate_list = sorted(distance_candidate_list)

        if return_distances:
            return distance_candidate_list
        else:
            return [x[1] for x in distance_candidate_list]


class PyNgramSpell:
    """
    :class:`PyNgramSpell` Corrects misspelled words from strings.

    :param ngram_range : ngram_range (NOW THIS IS FIXED future versions will improve).
    :param tokenizer : tokenizer function to tokenize an input string.
    :param string_preprocessor_func : function used to pre-process the string, `lower` by default.
    :param token_pattern : Regular expression used to tokenize an input string.
    :param lambda_interpolation : Regularization assigned to the interpolation.
    :param min_freq : Minimum frequency for a word to be  in the vocabulary.
    :param max_dist : Maximum edit distance allowed to generate candidate tokens.
    :param sort_candidates : Provide candidates sorted by edit distance.
    :param bktree : True if bktree is used to search candidates. False means exhaustive search.
    """

    def __init__(self,
                 ngram_range: Tuple[int] = (1, 2),
                 token_pattern: str = r"(?u)\b\w\w+\b",
                 lambda_interpolation: float = 0.3,
                 min_freq: int = 5,
                 max_dist: int = 1,
                 sort_candidates: bool = False,
                 use_bktree: bool = True,
                 tokenizer: Optional[Callable] = None,
                 string_preprocessor_func: Callable = str.lower
                 ):

        self.ngram_range = ngram_range
        self.string_preprocessor_func = string_preprocessor_func
        self.token_pattern = token_pattern
        self.lambda_interpolation = lambda_interpolation
        self.min_freq = min_freq
        self.max_dist = max_dist
        self.sort_candidates = sort_candidates
        self.use_bktree = use_bktree
        self.tokenizer = tokenizer
        self.vocabulary = None
        self.bktree = None

    def _build_tokenizer(self):
        """Return a function that splits a string into a sequence of tokens.
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

    def _get_candidates_bktree(self, token: str, max_dist: int):
        """Return a list of candidate words from the vocabulary at most `max_dist` away from the input token.
        :param token : token to be corrected.
        :param max_dist : maximum allowed edit distance.
        """
        candidate_tokens = self.bktree.query(token, max_dist)

        if len(candidate_tokens) > 0:
            return candidate_tokens
        else:
            return [token]

    def _get_candidates_exhaustive(self, token: str, max_dist: int):
        """Return a list of candidate words from the vocabulary at most `max_dist` away from the input token.
        This version of the function is private and kept for benchmarking pourposes. This function computes the
        edit distance between the input token and all words in the vocabulary. Then it filters candidates by
        the edit distance.
        :param token : word to be corrected.
        :param max_dist : maximum distance allowed for candidates.
        """
        from editdistance import eval as edit_distance

        token = token.lower()
        distance_token_to_words = {word: edit_distance(word, token) for word in self.vocabulary}
        min_dist = min(distance_token_to_words.values())
        if min_dist <= max_dist:

            if self.sort_candidates:
                result = sorted(
                    [(distance, word) for word, distance in distance_token_to_words.items() if distance <= max_dist])
            else:
                result = [word for word, distance in distance_token_to_words.items() if distance <= max_dist]

            return result
        else:
            return [token]

    def _get_candidates(self, token: str, max_dist: int):
        """Get the candidate words with at most `max_dist`
        :param token : token to be corrected.
        :param max_dist : maximum allowed edit distance.
        """
        if self.bktree is not None:
            return self._get_candidates_bktree(token, max_dist)
        else:
            return self._get_candidates_exhaustive(token, max_dist)

    def _filter_vocabulary(self, min_freq: int) -> None:
        """Remove words from the vocabulary that do not have more than `min_freq`
        :param min_freq : minimum frequency required to be in the vocabulary
        """
        self.vocabulary = set(dict(filter(lambda x: x[1] > min_freq, self.unigram_freq_dict.items())).keys())

    def _correct_with_bigrams(self, tokenized_sentence: List[int]) -> List[int]:
        """Correct the words in the tokenized_sentence that are not part of the vocabulary
        :param tokenized_sentence : sentence tokenized
        """

        def _prob_word(word):
            return self.unigram_freq_dict.get(word, 0) / len(self.unigram_freq_dict)

        def _probability_bigram(word1, word2):
            bigram = ((word1, word2))
            if self.bigram_freq_dict.get(bigram, 0) == 0:
                return 0
            else:
                return self.bigram_freq_dict.get(bigram, 0) / self.unigram_freq_dict[word1]

        def _interpolation_probability(word1, word2):
            return (1 - self.lambda_interpolation) * _probability_bigram(word1, word2) + \
                   self.lambda_interpolation * _prob_word(word2)

        previous_word = '.'
        for index, word in enumerate(tokenized_sentence):
            if word not in self.vocabulary:
                candidates = {candidate: _interpolation_probability(previous_word, candidate) for candidate in
                              self._get_candidates(word, max_dist=self.max_dist)}

                tokenized_sentence[index] = max(candidates, key=candidates.get)
            previous_word = tokenized_sentence[index]

        return tokenized_sentence

    def fit(self, X: Iterable[str]):
        """Fit the ngram model and the vocabulary from the training data.
        :param X : Iterable over strings containing the corpus used to train the spellcheker.
        """
        from nltk.collocations import BigramCollocationFinder
        from editdistance import eval as edit_distance

        self.tokenize_func = self._build_tokenizer()
        X_tokenized = [self.tokenize_func(self.string_preprocessor_func(x)) for x in X]
        self.unigram_freq_dict = dict(Counter(itertools.chain(*X_tokenized)))
        bigram_finder = BigramCollocationFinder.from_documents(X_tokenized)
        self.bigram_freq_dict = dict(bigram_finder.ngram_fd.items())
        self.vocabulary = set(list(itertools.chain(*self.bigram_freq_dict.keys())))

        if self.min_freq > 0:
            self._filter_vocabulary(min_freq=self.min_freq)

        if self.use_bktree:
            self.bktree = BKTree(edit_distance, self.vocabulary, sort_candidates=self.sort_candidates)

    def transform(self, x: str, tokenize: bool = True) -> str:
        """Corrects the misspelled words on each element from X.
        :param x : string to be corrected
        :param tokenize:

        """
        x = self.string_preprocessor_func(x)

        if tokenize:
            tokenized_sentence = self.tokenize_func(x)
        else:
            tokenized_sentence = x.copy()

        tokenized_sentence = self._correct_with_bigrams(tokenized_sentence)

        return tokenized_sentence

    def save(self, file_path: str):
        """Save the instance an PyNgramSpell to the provided path.
        :param file_path : path where to store the spell checker object
        """
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
