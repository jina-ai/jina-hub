__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Iterable, Dict

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
        self, old_match_scores: Iterable[float], query_meta: Dict, match_meta: Iterable[Dict]
    ) -> "np.ndarray":
        """
        Calculate the negative Levenshtein distance

        :param old_match_scores: Previously scored matches
        :param query_meta: A Dict of queries to score
        :param match_meta: A Dict of matches of given query
        :return: an `ndarray` of the size ``M x 2``.
            `M`` is the number of new scores.
            The columns correspond to the ``COL_MATCH_ID`` and ``COL_SCORE``.

        """
        from Levenshtein import distance

        return [-distance(query_meta['text'], m['text']) for m in match_meta]
