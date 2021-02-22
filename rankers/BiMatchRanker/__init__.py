__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Dict, Union, Optional

import numpy as np

from jina.executors.rankers import Chunk2DocRanker


class BiMatchRanker(Chunk2DocRanker):
    """The :class:`BiMatchRanker` counts the best chunk-hit from both query and doc perspective.

    `BiMatchRanker` tries to assign a score for a `match` given a `query` based on the matches found for the `query`
    chunks on the `match` chunks.

    The situation can be seen with an example:

    `Documents in the Index`:
        - Granularity 0 - Doc 1 - `BiMatchRanker is a Jina Hub Executor. It is used for Ranking`
                        - Doc 11 - `BiMatchRanker is a Jina Hub Executor`
                        - Doc 12 - `It is used for Ranking`
        - Granularity 0 - Doc 2 - `A Ranker gives relevance Scores. `
                        - Granularity 1 - Doc 21 - `A Ranker gives relevance Scores`

    `Query`
        - Granularity 0 - 'A Hub Executor. Ranking on Relevance scores'
                        - Granularity 1 - `A Hub Executor`
                                        - `Ranking on Relevance scores`

    `Semantic similarity matches (TOP_K = 2)`
        Query chunk 1 -> `A Hub Executor` -> Doc11, Doc12 (All are children of `Doc1`)
        Query chunk 2 -> `Ranking on Relevance scores` -> Doc21, Doc12 (Children of `Doc2` and `Doc1`)

    In order to compute this relevance score, it tries to take into account two perspectives.
        - How often a specific match (chunk of the `match`) is found in a set of queries (the chunks of the `query`) (query perspective)
        - How many queries is document a match of?
    many queries is document a match of?

    .. warning:: Here we suppose that the smaller chunk score means the more similar.
    """
    query_required_keys = ('length', )
    match_required_keys = ('length', )

    def __init__(self, d_miss: Optional[Union[int, float]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_miss = d_miss or 2000

    def _get_score(self, match_idx: 'np.ndarray', query_chunk_meta: Dict, match_chunk_meta: Dict, *args, **kwargs):
        # match_idx [parent_match_id, match_id, query_id, score]
        # all matches have the same parent_id (the parent_id of the matches)
        s1 = self._directional_score(match_idx, match_chunk_meta, col=self.COL_DOC_CHUNK_ID)
        s2 = self._directional_score(match_idx, query_chunk_meta, col=self.COL_QUERY_CHUNK_ID)
        return self.get_doc_id(match_idx), (s1 + s2) / 2.

    def _directional_score(self, g: Dict, chunk_meta: Dict, col: str):
        # g [parent_match_id, match_id, query_id, score]
        # col = self.COL_MATCH_ID, from matched_chunk aspect
        # col = self.COL_DOC_CHUNK_ID, from query chunk aspect
        # group by "match_id" or "query_chunk_id". So here groups have in common 1st and `n` column (match_parent_id)
        _groups = self._group_by(g, col)
        # take the best match from each group
        _groups_best = np.stack([np.sort(gg, order=col)[0] for gg in _groups])
        # doc total length
        _c = chunk_meta[_groups_best[0][col]]['length']
        # hit chunks
        _h = _groups_best.shape[0]
        # hit distance
        sum_d_hit = np.sum(_groups_best[self.COL_SCORE])
        # all hit => 0, all_miss => 1
        return 1 - (sum_d_hit + self.d_miss * (_c - _h)) / (self.d_miss * _c)
