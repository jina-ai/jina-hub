import pytest
from jina import Document
from jina.drivers.rank.aggregate import Chunk2DocRankDriver
from jina.types.score import NamedScore
from jina.types.sets import DocumentSet

from .. import BiMatchRanker

DISCOUNT_VAL = 0.5


class SimpleChunk2DocRankDriver(Chunk2DocRankDriver):
    def __init__(self, docs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._docs = docs

    @property
    def exec_fn(self):
        return self._exec_fn

    @property
    def docs(self):
        return self._docs


def create_document_to_score():
    # doc: 1
    # |- chunk: 2
    # |  |- matches: (id: 4, parent_id: 40, score.value: 4),
    # |  |- matches: (id: 5, parent_id: 50, score.value: 5),
    # |
    # |- chunk: 3
    #    |- matches: (id: 6, parent_id: 60, score.value: 6),
    #    |- matches: (id: 7, parent_id: 70, score.value: 7)

    # The smaller score will have higher value after Bimatchranker, so the order will be 40, 50, 60, 70.
    doc = Document()
    doc.id = '1'
    for c in range(2):
        chunk = Document()
        chunk_id = str(c + 2)
        chunk.id = chunk_id
        chunk.length = (c + 2) * 2
        for m in range(2):
            match = Document()
            match_id = 2 * int(chunk_id) + m
            match.id = str(match_id)
            parent_id = 10 * int(match_id)
            match.parent_id = str(parent_id)
            match.length = int(match_id)
            # to be used by MaxRanker and MinRanker
            match.score = NamedScore(value=int(match_id), ref_id=chunk.id)
            match.tags['price'] = match.score.value
            match.tags['discount'] = DISCOUNT_VAL
            chunk.matches.append(match)
        doc.chunks.append(chunk)
    return doc


@pytest.mark.parametrize('keep_source_matches_as_chunks', [False, True])
def test_chunk2doc_ranker_driver_bimatch_ranker(keep_source_matches_as_chunks):
    doc = create_document_to_score()
    driver = SimpleChunk2DocRankDriver(
        docs=DocumentSet([doc]),
        keep_source_matches_as_chunks=keep_source_matches_as_chunks,
    )
    executor = BiMatchRanker()
    driver.attach(executor=executor, runtime=None)
    driver()
    assert len(doc.matches) == 4
    assert doc.matches[0].id == '40'
    assert doc.matches[1].id == '50'
    assert doc.matches[2].id == '60'
    assert doc.matches[3].id == '70'

    for i in range(0, 3):
        assert doc.matches[i].score.value > doc.matches[i + 1].score.value

    for match in doc.matches:
        # match score is computed w.r.t to doc.id
        assert match.score.ref_id == doc.id
        expected_chunk_matches_length = 1 if keep_source_matches_as_chunks else 0
        assert len(match.chunks) == expected_chunk_matches_length


def create_data(query_chunk2match_chunk):
    doc = Document()
    doc.id = '1'
    doc.granularity = 0
    num_query_chunks = len(query_chunk2match_chunk)
    for query_chunk_id, matches in query_chunk2match_chunk.items():
        chunk = Document()
        chunk.id = str(query_chunk_id)
        chunk.granularity = doc.granularity + 1
        chunk.length = num_query_chunks
        for match_data in matches:
            match = Document()
            match.granularity = chunk.granularity
            match.parent_id = match_data['parent_id']
            match.score = NamedScore(value=match_data['score'], ref_id=chunk.id)
            match.id = match_data['id']
            match.length = match_data['length']
            chunk.matches.append(match)
        doc.chunks.append(chunk)
    return doc


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
    doc = create_data(query_chunk2match_chunk)
    driver = SimpleChunk2DocRankDriver(
        docs=DocumentSet([doc]),
    )
    executor = BiMatchRanker(d_miss=1)
    driver.attach(executor=executor, runtime=None)
    driver()

    # check the matched docs are in descending order of the scores
    match_score_1 = doc.matches[0].score.value
    match_score_2 = doc.matches[1].score.value

    assert match_score_1 > match_score_2
    assert doc.matches[0].id == '2'
    assert match_score_1 == pytest.approx(0.5333, 0.001)
    assert doc.matches[1].id == '1'
    assert match_score_2 == pytest.approx(0.3458, 0.001)
    # check the number of matched docs
    assert len(doc.matches) == 2


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

    doc = create_data(query_chunk2match_chunk)
    driver = SimpleChunk2DocRankDriver(
        docs=DocumentSet([doc]),
    )
    executor = BiMatchRanker()
    driver.attach(executor=executor, runtime=None)
    driver()

    match_score_1 = doc.matches[0].score.value
    match_score_2 = doc.matches[1].score.value

    assert match_score_1 > match_score_2
    assert doc.matches[0].id == '1'
    assert match_score_1 == pytest.approx(0.5048, 0.001)
    assert doc.matches[1].id == '4294967294'
    assert match_score_2 == pytest.approx(0.2516, 0.001)
    # check the number of matched docs
    assert len(doc.matches) == 2
