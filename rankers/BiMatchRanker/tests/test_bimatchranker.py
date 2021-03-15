import numpy as np
import pytest

from .. import BiMatchRanker


def test_bimatchranker():
    """
    Before grouping:
    query_chunk2match_chunk = {
        100: [
            {'parent_id': 1, 'id': 10, 'score': 0.4, 'length': 200},
        ],
        110: [
            {'parent_id': 1, 'id': 10, 'score': 0.3, 'length': 200},
            {'parent_id': 1, 'id': 11, 'score': 0.2, 'length': 200},
            {'parent_id': 4294967294, 'id': 20, 'score': 0.1, 'length': 300},
        ]
    }
    """
    ranker = BiMatchRanker()
    query_chunk_meta = {'100': {'length': 2}, '110': {'length': 2}}
    match_chunk_meta = {'10': {'length': 200}, '11': {'length': 200}, '20': {'length': 300}}

    # test score() with fake grouped data
    match_idx_1 = np.array([('1', '10', '100', 0.40000001), ('1', '10', '110', 0.30000001),
                            ('1', '11', '110', 0.2)],
                           dtype=[('match_parent_id', '<U64'), ('match_doc_chunk_id', '<U64'),
                                  ('match_query_chunk_id', '<U64'), ('score', '<f8')])
    match_score_1 = ranker.score(match_idx_1, query_chunk_meta, match_chunk_meta)
    assert match_score_1 == pytest.approx(0.5048, 0.001)

    match_idx_2 = np.array([('4294967294', '20', '110', 0.1)],
                           dtype=[('match_parent_id', '<U64'), ('match_doc_chunk_id', '<U64'),
                                  ('match_query_chunk_id', '<U64'), ('score', '<f8')])
    match_score_2 = ranker.score(match_idx_2, query_chunk_meta, match_chunk_meta)
    assert match_score_2 == pytest.approx(0.2516, 0.001)


def test_bimatchranker_readme():
    """
    query_chunk2match_chunk = {
    1: [
        {'parent_id': 1, 'id': 11, 'score': 0.1, 'length': 4},
        {'parent_id': 2, 'id': 22, 'score': 0.5, 'length': 3},
        {'parent_id': 2, 'id': 21, 'score': 0.7, 'length': 3},
    ],
    2: [
        {'parent_id': 2, 'id': 21, 'score': 0.1, 'length': 3},
        {'parent_id': 1, 'id': 11, 'score': 0.5, 'length': 4},
        {'parent_id': 2, 'id': 22, 'score': 0.7, 'length': 3},
    ],
    3: [
        {'parent_id': 2, 'id': 21, 'score': 0.1, 'length': 3},
        {'parent_id': 2, 'id': 22, 'score': 0.5, 'length': 3},
        {'parent_id': 2, 'id': 23, 'score': 0.7, 'length': 3},
    ]
    }
    """
    ranker = BiMatchRanker(d_miss=1)
    query_chunk_meta = {'1': {'length': 3}, '2': {'length': 3}, '3': {'length': 3}}
    match_chunk_meta = {'11': {'length': 4}, '22': {'length': 3}, '21': {'length': 3}, '23': {'length': 3}}
    # test score() with fake grouped data

    match_idx_1 = np.array([('1', '11', '1', 0.1), ('1', '11', '2', 0.5)],
                           dtype=[('match_parent_id', '<U64'), ('match_doc_chunk_id', '<U64'),
                                  ('match_query_chunk_id', '<U64'), ('score', '<f8')])
    match_score_1 = ranker.score(match_idx_1, query_chunk_meta, match_chunk_meta)
    assert match_score_1 == pytest.approx(0.3458, 0.001)

    match_idx_2 = np.array([('2', '21', '1', 0.69999999), ('2', '21', '2', 0.1),
                            ('2', '21', '3', 0.1), ('2', '22', '1', 0.5),
                            ('2', '22', '2', 0.69999999), ('2', '22', '3', 0.5),
                            ('2', '23', '3', 0.69999999)],
                           dtype=[('match_parent_id', '<U64'), ('match_doc_chunk_id', '<U64'),
                                  ('match_query_chunk_id', '<U64'), ('score', '<f8')])
    match_score_2 = ranker.score(match_idx_2, query_chunk_meta, match_chunk_meta)
    assert match_score_2 == pytest.approx(0.5333, 0.001)
