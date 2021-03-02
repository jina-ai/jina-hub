__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import copy
import json
import numpy as np

from jina.executors.rankers import Match2DocRanker

from .. import LevenshteinRanker


def test_levenshteinranker():
    query_meta = {"text": "cool stuff"}
    query_meta_json = json.dumps(query_meta, sort_keys=True)
    old_match_scores = {'1': 5, '2': 4}
    old_match_scores_json = json.dumps(old_match_scores, sort_keys=True)
    match_meta = {'1': {"text": "cool stuff"}, '2': {"text": "kewl stuff"}}
    match_meta_json = json.dumps(match_meta, sort_keys=True)

    ranker = LevenshteinRanker()
    new_scores = ranker.score(
        copy.deepcopy(query_meta),
        copy.deepcopy(old_match_scores),
        copy.deepcopy(match_meta)
    )

    np.testing.assert_array_equal(
        new_scores,
        np.array(
            [('1', 0), ('2', -3)],
            dtype=[(Match2DocRanker.COL_MATCH_ID, np.object), (Match2DocRanker.COL_SCORE, np.float64)],
        )
    )
    # Guarantee no side-effects happen
    assert query_meta_json == json.dumps(query_meta, sort_keys=True)
    assert old_match_scores_json == json.dumps(old_match_scores, sort_keys=True)
    assert match_meta_json == json.dumps(match_meta, sort_keys=True)
