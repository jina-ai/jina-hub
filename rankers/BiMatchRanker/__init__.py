__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Dict, Union, Optional

import numpy as np
from jina.executors.rankers import Chunk2DocRanker


class BiMatchRanker(Chunk2DocRanker):
    """The :class:`BiMatchRanker` counts the best chunk-hit from both query and doc perspective.

    `BiMatchRanker` tries to assign a score for a `match` given a `query` based on the matches found for the `query`
    chunks on the `match` chunks.

    The situation can be seen with an hypothetical example:

    `Documents in the Index`:
        - Granularity 0 - Doc 1 - `BiMatchRanker is a Hub Executor. computes relevance scores. Found in Jina, the neural search framework`
                        - Granularity 1 - Doc 11 - `BiMatchRanker is a Ranking Jina Hub Executor`
                                        - Doc 12 - `It is a powerfull tool.`
                                        - Doc 13 - `Found in Jina`
                                        - Doc 14 - `the neural search framework`
        - Granularity 0 - Doc 2 - `A Ranker executor gives relevance Scores. Rankers are found in the Hub, and built automatically`
                        - Granularity 1 - Doc 21 - `A Ranker executor gives relevance Scores`
                        - Granularity 1 - Doc 22 - `Rankers are found in the Hub`
                        - Granularity 1 - Doc 23 - `and built automatically`

    `Query`
        - Granularity 0 - 'A Hub Executor. Ranking on Relevance scores. Giving automatically relevant documents from the Hub'
                        - Granularity 1 - `A Hub Executor`
                                        - `Ranking on Relevance scores`
                                        - `Giving automatically relevant documents from the Hub`

    `Semantic similarity matches` (TOP_K 3)
        Query chunk 1 -> `A Hub Executor` -> Doc11, Doc22, Doc21 (Children of `Doc1` and `Doc2`)
        Query chunk 2 -> `Ranking on Relevance scores` -> Doc21, Doc11, Doc22 (Children of `Doc2` and `Doc1`)
        Query chunk 3 -> `Scoring relevant documents`  -> Doc21, Doc22, Doc23 (Children of `Doc2`)

    In order to compute this relevance score, it tries to take into account two perspectives.

        - Match perspective: How many chunks of a match are part of the set of matches (hit) of a query (and its chunks).
            In the example:
                - `Doc1` has 4 chunks, but only 1 (hit) (Doc11) is found in the matches of the query chunks.
                - `Doc2` has 3 chunks and 2 of them (hit) are found in the matches (Doc21, Doc22).

        - Query perspective: Of all the chunks in the query, how many of them are matches by any of the chunks of each match document.
            In the example:
                - `Query` for `Doc1`: Query has 3 chunks, but only 2 of them are matched by chunks of match `Doc1` (Doc11)
                - `Query` for `Doc2`: Query has 3 chunks, and the 3 of them are matched by chunks of match `Doc2`

    So for every `Document`, a score is computed adding some penalty to the `missing` in both perspectives.

    :param d_miss: Cost associated to a miss chunk
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments

    .. warning:: Here we suppose that the smaller chunk score means the more similar.
    """

    def __init__(self, d_miss: Optional[Union[int, float]] = None, *args, **kwargs):
        super().__init__(query_required_keys=('siblings', ), match_required_keys=('siblings', ), *args, **kwargs)
        self.d_miss = d_miss or 2000

    def score(self, match_idx: 'np.ndarray', query_chunk_meta: Dict, match_chunk_meta: Dict, *args, **kwargs):
        # match_idx [parent_match_id, match_id, query_id, score]
        # all matches have the same parent_id (the parent_id of the matches)
        s1 = self._directional_score(match_idx, match_chunk_meta, col=self.COL_DOC_CHUNK_ID)
        s2 = self._directional_score(match_idx, query_chunk_meta, col=self.COL_QUERY_CHUNK_ID)
        return (s1 + s2) / 2.

    def _group_by(self, match_idx, col_name):
        # sort by ``col
        _sorted_m = np.sort(match_idx, order=col_name)
        _, _doc_counts = np.unique(_sorted_m[col_name], return_counts=True)
        # group by ``col``
        return np.split(_sorted_m, np.cumsum(_doc_counts))[:-1]

    def _directional_score(self, g: Dict, chunk_meta: Dict, col: str):
        # g [parent_match_id, match_id, query_id, score]
        # col = self.COL_DOC_CHUNK_ID, from matched_chunk aspect
        # col = self.COL_QUERY_CHUNK_ID, from query chunk aspect
        # group by "match_id" or "query_chunk_id". So here groups have in common 1st (match_parent_id) and `n` column
        _groups = self._group_by(g, col)
        # take the best match from each group
        _groups_best = np.stack([np.sort(gg, order='score')[0] for gg in _groups])
        # doc total siblings, which is the the number of chunks
        # how many chunks in the document (match or query)
        _c = chunk_meta[_groups_best[0][col]]['siblings']
        # how many chunks hit
        _h = _groups_best.shape[0]
        # hit distance
        sum_d_hit = np.sum(_groups_best[self.COL_SCORE])
        # all hit => 0, all_miss => 1
        return 1 - (sum_d_hit + self.d_miss * (_c - _h)) / (self.d_miss * _c)
