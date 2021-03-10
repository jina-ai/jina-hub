import pytest
from jina import Document
from jina.drivers.rank.aggregate import Chunk2DocRankDriver
from jina.types.score import NamedScore
from jina.types.sets import DocumentSet

from .. import BiMatchRanker


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



