import pytest
import numpy as np

from jina.executors.rankers import Chunk2DocRanker

from .. import BiMatchRanker


def create_data(query_chunk2match_chunk):
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
            (Chunk2DocRanker.COL_PARENT_ID, np.int64),
            (Chunk2DocRanker.COL_DOC_CHUNK_ID, np.int64),
            (Chunk2DocRanker.COL_QUERY_CHUNK_ID, np.int64),
            (Chunk2DocRanker.COL_SCORE, np.float64)
        ]
    )
    return match_idx_numpy, query_chunk_meta, match_chunk_meta


def test_bimatchranker():
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
    ranker = BiMatchRanker()
    match_idx, query_chunk_meta, match_chunk_meta = create_data(query_chunk2match_chunk)
    doc_idx = ranker.score(match_idx, query_chunk_meta, match_chunk_meta)
    # check the matched docs are in descending order of the scores
    assert doc_idx[0][1] > doc_idx[1][1]
    assert doc_idx[1][0] == '4294967294'
    assert doc_idx[0][0] == '1'
    # check the number of matched docs
    assert len(doc_idx) == 2


def test_bimatchranker_readme():
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
    ranker = BiMatchRanker(d_miss=1)
    match_idx, query_chunk_meta, match_chunk_meta = create_data(query_chunk2match_chunk)
    doc_idx = ranker.score(match_idx, query_chunk_meta, match_chunk_meta)

    # check the matched docs are in descending order of the scores
    assert doc_idx[0][1] > doc_idx[1][1]
    assert doc_idx[0][0] == '2'
    assert doc_idx[0][1] == pytest.approx(0.5333, 0.001)
    assert doc_idx[1][0] == '1'
    assert doc_idx[1][1] == pytest.approx(0.3458, 0.001)
    # check the number of matched docs
    assert len(doc_idx) == 2
