import numpy as np
from typing import Dict

from jina.executors.rankers import Match2DocRanker


class ReverseRanker(Match2DocRanker):
    """
    :class:`ReverseRanker` This ranker returns the inverse of all scores in
        order to be able to reverse the match ranking.
    """

    required_keys = {}

    def score(
        self, query_meta: Dict, old_match_scores: Dict, match_meta: Dict
    ) -> "np.ndarray":
        new_scores = [
            (
                match_id,
                -old_score,
            )
            for match_id, old_score in old_match_scores.items()
        ]
        return np.array(new_scores, dtype=np.float64)
