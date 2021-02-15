from typing import Dict

import numpy as np
from jina.executors.rankers import Match2DocRanker


class LevenshteinRanker(Match2DocRanker):
    """
    :class:`LevenshteinRanker` Computes the negative Levenshtein distance
        between a query and its matches. The distance is negative, in order to
        achieve a bigger=better result, sort in the respective driver.
    """

    required_keys = {"text"}

    def score(
        self, query_meta: Dict, old_match_scores: Dict, match_meta: Dict
    ) -> "np.ndarray":
        """
        Calculate the negative Levenshtein distance

        :param query_meta: A Dict of queries to score
        :param old_match_scores: Previously scored matches
        :param match_meta: A Dict of matches of given query
        :return: an `ndarray` of the size ``M x 2``.
            `M`` is the number of new scores.
            The columns correspond to the ``COL_MATCH_ID`` and ``COL_SCORE``.

        """
        from Levenshtein import distance

        new_scores = [
            (
                match_id,
                -distance(query_meta['text'], match_meta[match_id]['text']),
            )
            for match_id, old_score in old_match_scores.items()
        ]
        return np.array(
            new_scores,
            dtype=[(self.COL_MATCH_ID, np.object), (self.COL_SCORE, np.float64)],
        )
