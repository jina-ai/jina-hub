__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from .. import LevenshteinRanker


def test_levenshteinranker():
    query_meta = {"text": "cool stuff"}
    old_match_scores = [5, 4]
    match_meta = [{"text": "cool stuff"}, {"text": "kewl stuff"}]

    ranker = LevenshteinRanker()
    new_scores = ranker.score(
        old_match_scores,
        query_meta,
        match_meta
    )

    assert len(new_scores) == 2
    assert new_scores[0] == 0
    assert new_scores[1] == -3
