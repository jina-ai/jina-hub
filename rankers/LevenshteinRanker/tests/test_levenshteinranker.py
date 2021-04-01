__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina.executors.rankers import Match2DocRanker

from .. import LevenshteinRanker


def test_levenshteinranker():
    queries_metas = [{'text': 'cool stuff'}, {'text': 'cool stuff'}]
    old_matches_scores = [[5, 4], [5, 4]]
    matches_metas = [
        [{'text': 'cool stuff'}, {'text': 'kewl stuff'}],
        [{'text': 'cool stuff'}, {'text': 'kewl stuff'}],
    ]

    ranker = LevenshteinRanker()
    new_scores = ranker.score(
        old_matches_scores,
        queries_metas,
        matches_metas,
    )

    assert len(new_scores) == 2
    assert new_scores[0] == [0, -3]
    assert new_scores[1] == [0, -3]
