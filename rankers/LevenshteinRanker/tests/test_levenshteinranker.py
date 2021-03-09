__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import copy
import json

from jina.executors.rankers import Match2DocRanker

from .. import LevenshteinRanker


def test_levenshteinranker():
    query_meta = {'text': 'cool stuff'}
    query_meta_json = json.dumps(query_meta, sort_keys=True)
    old_match_scores = [5, 4]
    old_match_scores_json = json.dumps(old_match_scores, sort_keys=True)
    match_meta = [{'text': 'cool stuff'}, {'text': 'kewl stuff'}]
    match_meta_json = json.dumps(match_meta, sort_keys=True)

    ranker = LevenshteinRanker()
    new_scores = ranker.score(
        copy.deepcopy(old_match_scores),
        copy.deepcopy(query_meta),
        copy.deepcopy(match_meta)
    )

    assert new_scores == [0, -3]

    # Guarantee no side-effects happen
    assert query_meta_json == json.dumps(query_meta, sort_keys=True)
    assert old_match_scores_json == json.dumps(old_match_scores, sort_keys=True)
    assert match_meta_json == json.dumps(match_meta, sort_keys=True)
