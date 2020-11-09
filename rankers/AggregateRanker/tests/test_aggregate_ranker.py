import numpy as np

from jina.executors.rankers import Chunk2DocRanker
import pytest
from .. import AggregateRanker


def create_data():
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
                c['score'],
            ))

    match_idx_numpy = np.array(
        match_idx,
        dtype=[
            (Chunk2DocRanker.COL_MATCH_PARENT_HASH, np.int64),
            (Chunk2DocRanker.COL_MATCH_HASH, np.int64),
            (Chunk2DocRanker.COL_DOC_CHUNK_HASH, np.int64),
            (Chunk2DocRanker.COL_SCORE, np.float64)
        ]
    )
    return match_idx_numpy, query_chunk_meta, match_chunk_meta


def assert_document_order(doc_idx):
    for i in range(0, len(doc_idx) - 1):
        assert doc_idx[i][1] > doc_idx[i + 1][1]


def test_max_ranker():
    ranker = AggregateRanker(aggregate_function='max', is_reversed_score=False)
    match_idx, query_chunk_meta, match_chunk_meta = create_data()
    doc_idx = ranker.score(match_idx, query_chunk_meta, match_chunk_meta)

    assert_document_order(doc_idx)
    assert doc_idx[0][0] == 1
    assert doc_idx[0][1] == 0.5

    assert doc_idx[1][0] == 2
    assert doc_idx[1][1] == 0.25

    assert doc_idx[2][0] == 3
    assert doc_idx[2][1] == 0.1

    assert len(doc_idx) == 3


def test_max_ranker_reversed_score():
    ranker = AggregateRanker(aggregate_function='max', is_reversed_score=True)
    match_idx, query_chunk_meta, match_chunk_meta = create_data()
    doc_idx = ranker.score(match_idx, query_chunk_meta, match_chunk_meta)

    assert doc_idx[0][0] == 3
    assert doc_idx[0][1] == 1 / (1 + 0.1)

    assert doc_idx[1][0] == 2
    assert doc_idx[1][1] == 1 / (1 + 0.25)

    assert doc_idx[2][0] == 1
    assert doc_idx[2][1] == 1 / (1 + 0.5)

    assert len(doc_idx) == 3


def test_min_ranker():
    ranker = AggregateRanker(aggregate_function='min', is_reversed_score=False)
    match_idx, query_chunk_meta, match_chunk_meta = create_data()
    doc_idx = ranker.score(match_idx, query_chunk_meta, match_chunk_meta)

    assert_document_order(doc_idx)
    assert doc_idx[0][0] == 1
    assert doc_idx[0][1] == 0.2

    assert doc_idx[1][0] == 2
    assert doc_idx[1][1] == 0.15

    assert doc_idx[2][0] == 3
    assert doc_idx[2][1] == 0.1

    assert len(doc_idx) == 3


def test_min_ranker_reversed_score():
    ranker = AggregateRanker(aggregate_function='min', is_reversed_score=True)
    match_idx, query_chunk_meta, match_chunk_meta = create_data()
    doc_idx = ranker.score(match_idx, query_chunk_meta, match_chunk_meta)

    assert_document_order(doc_idx)
    assert doc_idx[0][0] == 3
    assert doc_idx[0][1] == 1 / (1 + 0.1)

    assert doc_idx[1][0] == 2
    assert doc_idx[1][1] == 1 / (1 + 0.15)

    assert doc_idx[2][0] == 1
    assert doc_idx[2][1] == 1 / (1 + 0.2)

    assert len(doc_idx) == 3


def test_mean_ranker():
    ranker = AggregateRanker(aggregate_function='mean', is_reversed_score=False)
    match_idx, query_chunk_meta, match_chunk_meta = create_data()
    doc_idx = ranker.score(match_idx, query_chunk_meta, match_chunk_meta)

    assert_document_order(doc_idx)
    assert doc_idx[0][0] == 1
    assert doc_idx[0][1] == 1.1 / 3

    assert doc_idx[1][0] == 2
    assert doc_idx[1][1] == 0.2

    assert doc_idx[2][0] == 3
    assert doc_idx[2][1] == 0.1

    assert len(doc_idx) == 3


def test_mean_ranker_reversed_score():
    ranker = AggregateRanker(aggregate_function='mean', is_reversed_score=True)
    match_idx, query_chunk_meta, match_chunk_meta = create_data()
    doc_idx = ranker.score(match_idx, query_chunk_meta, match_chunk_meta)

    assert_document_order(doc_idx)
    assert doc_idx[0][0] == 3
    assert doc_idx[0][1] == 1 / (1 + 0.1)

    assert doc_idx[1][0] == 2
    assert doc_idx[1][1] == 1 / (1 + 0.2)

    assert doc_idx[2][0] == 1
    assert doc_idx[2][1] == 1 / (1 + 1.1 / 3)

    assert len(doc_idx) == 3


def test_median_ranker():
    ranker = AggregateRanker(aggregate_function='median', is_reversed_score=False)
    match_idx, query_chunk_meta, match_chunk_meta = create_data()
    doc_idx = ranker.score(match_idx, query_chunk_meta, match_chunk_meta)

    assert_document_order(doc_idx)
    assert doc_idx[0][0] == 1
    assert doc_idx[0][1] == 0.4

    assert doc_idx[1][0] == 2
    assert doc_idx[1][1] == 0.2

    assert doc_idx[2][0] == 3
    assert doc_idx[2][1] == 0.1

    assert len(doc_idx) == 3


def test_median_ranker_reversed_score():
    ranker = AggregateRanker(aggregate_function='median', is_reversed_score=True)
    match_idx, query_chunk_meta, match_chunk_meta = create_data()
    doc_idx = ranker.score(match_idx, query_chunk_meta, match_chunk_meta)

    assert_document_order(doc_idx)
    assert doc_idx[0][0] == 3
    assert doc_idx[0][1] == 1 / (1 + 0.1)

    assert doc_idx[1][0] == 2
    assert doc_idx[1][1] == 1 / (1 + 0.2)

    assert doc_idx[2][0] == 1
    assert doc_idx[2][1] == 1 / (1 + 0.4)

    assert len(doc_idx) == 3


def test_sum_ranker():
    ranker = AggregateRanker(aggregate_function='sum', is_reversed_score=False)
    match_idx, query_chunk_meta, match_chunk_meta = create_data()
    doc_idx = ranker.score(match_idx, query_chunk_meta, match_chunk_meta)

    assert_document_order(doc_idx)
    assert doc_idx[0][0] == 1
    assert doc_idx[0][1] == 1.1

    assert doc_idx[1][0] == 2
    assert doc_idx[1][1] == 0.8

    assert doc_idx[2][0] == 3
    assert doc_idx[2][1] == 0.1

    assert len(doc_idx) == 3


def test_sum_ranker_reversed_score():
    ranker = AggregateRanker(aggregate_function='sum', is_reversed_score=True)
    match_idx, query_chunk_meta, match_chunk_meta = create_data()
    doc_idx = ranker.score(match_idx, query_chunk_meta, match_chunk_meta)

    assert_document_order(doc_idx)
    assert doc_idx[0][0] == 3
    assert doc_idx[0][1] == 1 / (1 + 0.1)

    assert doc_idx[1][0] == 2
    assert doc_idx[1][1] == 1 / (1 + 0.8)

    assert doc_idx[2][0] == 1
    assert doc_idx[2][1] == 1 / (1 + 1.1)

    assert len(doc_idx) == 3


def test_prod_ranker():
    ranker = AggregateRanker(aggregate_function='prod', is_reversed_score=False)
    match_idx, query_chunk_meta, match_chunk_meta = create_data()
    doc_idx = ranker.score(match_idx, query_chunk_meta, match_chunk_meta)

    assert_document_order(doc_idx)
    assert doc_idx[0][0] == 3
    assert doc_idx[0][1] == 0.1

    assert doc_idx[1][0] == 1
    assert doc_idx[1][1] == 0.2 * 0.4 * 0.5

    assert doc_idx[2][0] == 2
    assert doc_idx[2][1] == 0.25 * 0.25 * 0.15 * 0.15

    assert len(doc_idx) == 3


def test_prod_ranker_reversed_score():
    ranker = AggregateRanker(aggregate_function='prod', is_reversed_score=True)
    match_idx, query_chunk_meta, match_chunk_meta = create_data()
    doc_idx = ranker.score(match_idx, query_chunk_meta, match_chunk_meta)

    assert_document_order(doc_idx)
    assert doc_idx[0][0] == 2
    assert doc_idx[0][1] == 1 / (1 + 0.25 * 0.25 * 0.15 * 0.15)

    assert doc_idx[1][0] == 1
    assert doc_idx[1][1] == 1 / (1 + 0.2 * 0.4 * 0.5)

    assert doc_idx[2][0] == 3
    assert doc_idx[2][1] == 1 / (1 + 0.1)

    assert len(doc_idx) == 3


def test_invalid_ranker():
    with pytest.raises(ValueError):
        AggregateRanker(aggregate_function='invalid_name', is_reversed_score=True)
