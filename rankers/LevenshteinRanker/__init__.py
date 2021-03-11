__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Dict, Iterable

from jina.executors.rankers import Match2DocRanker


class LevenshteinRanker(Match2DocRanker):
    """
    :class:`LevenshteinRanker` Computes the negative Levenshtein distance
        between a query and its matches. The distance is negative, in order to
        achieve a bigger=better result, sort in the respective driver.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(query_required_keys=['text'], match_required_keys=['text'], *args, *kwargs)

    def score(
            self,
            old_match_scores: Iterable[float],
            query_meta: Dict,
            match_meta: Iterable[Dict],
    ) -> Iterable[float]:
        """
        Calculate the negative Levenshtein distance

        :param old_match_scores: Contains old scores in a list
        :param query_meta: Dictionary containing all the query meta information requested by the `query_required_keys` class_variable.
        :param match_meta: List containing all the matches meta information requested by the `match_required_keys` class_variable. Sorted in the same way as `old_match_scores`
        :return: An iterable of the .

        """
        from Levenshtein import distance
        return [-distance(query_meta['text'], m['text']) for m in match_meta]
