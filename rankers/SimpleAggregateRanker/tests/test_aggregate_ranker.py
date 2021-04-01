import numpy as np

from jina.executors.rankers import Chunk2DocRanker
import pytest
from .. import SimpleAggregateRanker

COL_STR_TYPE = 'U64'


def group_by(match_idx, col_name):
    """
    fake the group function in Driver
    """
    # sort by ``col
    _sorted_m = np.sort(match_idx, order=col_name)
    _, _doc_counts = np.unique(_sorted_m[col_name], return_counts=True)
    # group by ``col``
    return np.split(_sorted_m, np.cumsum(_doc_counts))[:-1]


def sort_doc_by_score(r):
    """
    fake the sort function in Driver
    """
    r = np.array(
        r,
        dtype=[
            (Chunk2DocRanker.COL_PARENT_ID, COL_STR_TYPE),
            (Chunk2DocRanker.COL_SCORE, np.float64),
        ],
    )
    return np.sort(r, order=Chunk2DocRanker.COL_SCORE)[::-1]


def fake_group_and_score(ranker, match_idx,query_chunk_meta,match_chunk_meta):
    _groups = group_by(match_idx, Chunk2DocRanker.COL_PARENT_ID)
    r = []
    for _g in _groups:
        match_id = _g[0][Chunk2DocRanker.COL_PARENT_ID]
        score = ranker.score(_g, query_chunk_meta, match_chunk_meta)
        r.append((match_id, score))
    return sort_doc_by_score(r)


def chunk_scores(factor=1):
    query_chunk2match_chunk = {
        100: [
            {'parent_id': 1, 'id': 10, 'score': 0.4, 'length': 200},
        ],
        101: [
            {'parent_id': 2, 'id': 20, 'score': 0.25, 'length': 200},
            {'parent_id': 2, 'id': 21, 'score': 0.25, 'length': 200},
            {'parent_id': 2, 'id': 22, 'score': 0.15, 'length': 200},
            {'parent_id': 2, 'id': 23, 'score': 0.15, 'length': 200},
        ],
        110: [
            {'parent_id': 1, 'id': 10, 'score': 0.5, 'length': 200},
            {'parent_id': 1, 'id': 11, 'score': 0.2, 'length': 200},
            {'parent_id': 3, 'id': 20, 'score': 0.1, 'length': 300},
        ]
    }
    query_chunk_meta = {}
    match_chunk_meta = {}
    match_idx = []
    num_query_chunks = len(query_chunk2match_chunk)
    for query_chunk_id, matches in query_chunk2match_chunk.items():
        query_chunk_meta[query_chunk_id] = {'length': num_query_chunks}
        for c in matches:
            match_chunk_meta[c['id']] = {'length': c['length']}
            match_idx.append((
                c['parent_id'],
                c['id'],
                query_chunk_id,
                c['score'] * factor,
            ))

    match_idx_numpy = np.array(
        match_idx,
        dtype=[
            (Chunk2DocRanker.COL_PARENT_ID, COL_STR_TYPE),
            (Chunk2DocRanker.COL_DOC_CHUNK_ID, COL_STR_TYPE),
            (Chunk2DocRanker.COL_QUERY_CHUNK_ID, COL_STR_TYPE),
            (Chunk2DocRanker.COL_SCORE, np.float64)
        ]
    )
    return match_idx_numpy, query_chunk_meta, match_chunk_meta


def assert_document_order(doc_idx):
    for i in range(0, len(doc_idx) - 1):
        assert doc_idx[i][1] > doc_idx[i + 1][1]


@pytest.mark.parametrize('chunk_scores, aggregate_function, inverse_score, doc_ids,doc_scores', [
    (
            chunk_scores(),
            'max',
            False,
            [1, 2, 3],
            [0.5, 0.25, 0.1]
    ),
    (
            chunk_scores(),
            'max',
            True,
            [3, 2, 1],
            [1 / (1 + 0.1), 1 / (1 + 0.25), 1 / (1 + 0.5)]
    ),
    (
            chunk_scores(),
            'min',
            False,
            [1, 2, 3],
            [0.2, 0.15, 0.1]
    ),
    (
            chunk_scores(),
            'min', True,
            [3, 2, 1],
            [1 / (1 + 0.1), 1 / (1 + 0.15), 1 / (1 + 0.2)]
    ),
    (
            chunk_scores(),
            'mean',
            False,
            [1, 2, 3],
            [1.1 / 3, 0.2, 0.1]
    ),
    (
            chunk_scores(),
            'mean',
            True,
            [3, 2, 1],
            [1 / (1 + 0.1), 1 / (1 + 0.2), 1 / (1 + 1.1 / 3)]
    ),
    (
            chunk_scores(),
            'median',
            False,
            [1, 2, 3],
            [0.4, 0.2, 0.1]
    ),
    (
            chunk_scores(),
            'median',
            True,
            [3, 2, 1],
            [1 / (1 + 0.1), 1 / (1 + 0.2), 1 / (1 + 0.4)]
    ),
    (
            chunk_scores(),
            'sum',
            False,
            [1, 2, 3],
            [1.1, 0.8, 0.1]
    ),
    (
            chunk_scores(),
            'sum',
            True,
            [3, 2, 1],
            [1 / (1 + 0.1), 1 / (1 + 0.8), 1 / (1 + 1.1)]
    ),
    (
            chunk_scores(),
            'prod',
            False,
            [3, 1, 2],
            [0.1, 0.2 * 0.4 * 0.5, 0.25 * 0.25 * 0.15 * 0.15]
    ),
    (
            chunk_scores(),
            'prod',
            True,
            [2, 1, 3],
            [1 / (1 + 0.25 * 0.25 * 0.15 * 0.15), 1 / (1 + 0.2 * 0.4 * 0.5), 1 / (1 + 0.1)]
    ),
])
def test_aggregate_functions(chunk_scores, aggregate_function, inverse_score, doc_ids, doc_scores):
    ranker = SimpleAggregateRanker(aggregate_function=aggregate_function, inverse_score=inverse_score)
    doc_idx = fake_group_and_score(ranker,*chunk_scores)
    assert_document_order(doc_idx)
    for i, (doc_id, score) in enumerate(zip(doc_ids, doc_scores)):
        assert int(doc_idx[i][0]) == doc_id
        assert doc_idx[i][1] == score
    assert len(doc_idx) == len(doc_ids) == len(doc_scores)

    ranker_deprecated = SimpleAggregateRanker(aggregate_function=aggregate_function, is_reversed_score=inverse_score)
    doc_idx = fake_group_and_score(ranker_deprecated, *chunk_scores)
    assert_document_order(doc_idx)
    for i, (doc_id, score) in enumerate(zip(doc_ids, doc_scores)):
        assert int(doc_idx[i][0]) == doc_id
        assert doc_idx[i][1] == score
    assert len(doc_idx) == len(doc_ids) == len(doc_scores)


def test_invalid_aggregate_function():
    with pytest.raises(ValueError):
        SimpleAggregateRanker(aggregate_function='invalid_name', inverse_score=True)


def test_doc_score_of_minus_one_invalid():
    ranker = SimpleAggregateRanker(aggregate_function='min', inverse_score=True)
    with pytest.raises(ValueError):
        ranker.score(*chunk_scores(factor=-10))


@pytest.mark.parametrize("factor", [1, -1])
def test_negative_values_allowed(factor):
    ranker = SimpleAggregateRanker(aggregate_function='min', inverse_score=False)
    ranker.score(*chunk_scores(factor=factor))
