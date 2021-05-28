import os

from jina import Flow

from .helpers import *
from .. import DocCache


def duplicate_docs():
    for _ in range(3):
        d = Document()
        d.text = 'abc'
        d.embedding = np.array([1, 3, 4])
        yield d


def test_doc_cache(tmpdir):
    os.environ['WORKSPACE'] = str(tmpdir)
    f = Flow.load_config('flow.yml')
    with f:
        f.post(
            on='/index',
            inputs=duplicate_docs(),
        )

        results = f.post(
            on='/search',
            inputs=duplicate_docs(),
            parameters={'top_k': 3}
        )
        assert len(results[0].docs[0].matches) == 1
