import numpy as np

from .. import BiMatchRanker


def create_data():
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
    query_chunk_meta = {}
    match_chunk_meta = {}
    match_idx = []
    num_query_chunks = len(query_chunk2match_chunk)
    for query_chunk_id, matches in query_chunk2match_chunk.items():
        query_chunk_meta[query_chunk_id] = {'length': num_query_chunks}
        for c in matches:
            match_chunk_meta[c['id']] = {'length': c['length']}
            match_idx.append([
                c['parent_id'],
                c['id'],
                query_chunk_id,
                c['score'],
            ])
    return np.array(match_idx), query_chunk_meta, match_chunk_meta


def test_bimatchranker():
    ranker = BiMatchRanker()
    match_idx, query_chunk_meta, match_chunk_meta = create_data()
    doc_idx = ranker.score(np.array(match_idx), query_chunk_meta, match_chunk_meta)
    # check the matched docs are in descending order of the scores
    assert doc_idx[0][1] > doc_idx[1][1]
    assert doc_idx[1][0] == 4294967294
    assert doc_idx[0][0] == 1
    # check the number of matched docs
    assert len(doc_idx) == 2
