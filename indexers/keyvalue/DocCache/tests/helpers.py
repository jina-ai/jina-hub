from typing import Tuple, Dict

import numpy as np
from jina import Executor, DocumentArray, requests, Document
from jina.logging import JinaLogger


def _safeget(dct, *keys):
    for key in keys:
        try:
            dct = dct[key]
        except KeyError:
            return None
    return dct


class MyIndexer(Executor):
    def __init__(
            self,
            default_traversal_paths: Tuple[str] = ('r',),
            **kwargs,
    ):
        self.logger = JinaLogger('MyIndexer')
        self.default_traversal_paths = default_traversal_paths
        super().__init__(**kwargs)
        self._docs = DocumentArray()

    @requests(on='/index')
    def index(self, docs: 'DocumentArray', parameters: Dict, **kwargs):
        self._docs.extend(docs)

    @requests(on=['/search', '/eval'])
    def search(self, docs: 'DocumentArray', parameters: Dict, **kwargs):
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        docs_to_search_with = docs.traverse_flat(traversal_paths)
        self.logger.info(f'Searching with {len(docs_to_search_with)} documents...')

        a = np.stack(docs_to_search_with.get_attributes('embedding'))
        b = np.stack(self._docs.get_attributes('embedding'))
        q_emb = _ext_A(_norm(a))
        d_emb = _ext_B(_norm(b))
        dists = _cosine(q_emb, d_emb)
        idx, dist = self._get_sorted_top_k(dists, int(parameters['top_k']))
        for _q, _ids, _dists in zip(docs_to_search_with, idx, dist):
            for _id, _dist in zip(_ids, _dists):
                d = Document(self._docs[int(_id)], copy=True)
                d.score.value = 1 - _dist
                _q.matches.append(d)

    @staticmethod
    def _get_sorted_top_k(
            dist: 'np.array', top_k: int
    ) -> Tuple['np.ndarray', 'np.ndarray']:
        if top_k >= dist.shape[1]:
            idx = dist.argsort(axis=1)[:, :top_k]
            dist = np.take_along_axis(dist, idx, axis=1)
        else:
            idx_ps = dist.argpartition(kth=top_k, axis=1)[:, :top_k]
            dist = np.take_along_axis(dist, idx_ps, axis=1)
            idx_fs = dist.argsort(axis=1)
            idx = np.take_along_axis(idx_ps, idx_fs, axis=1)
            dist = np.take_along_axis(dist, idx_fs, axis=1)

        return idx, dist


def _get_ones(x, y):
    return np.ones((x, y))


def _ext_A(A):
    nA, dim = A.shape
    A_ext = _get_ones(nA, dim * 3)
    A_ext[:, dim: 2 * dim] = A
    A_ext[:, 2 * dim:] = A ** 2
    return A_ext


def _ext_B(B):
    nB, dim = B.shape
    B_ext = _get_ones(dim * 3, nB)
    B_ext[:dim] = (B ** 2).T
    B_ext[dim: 2 * dim] = -2.0 * B.T
    del B
    return B_ext


def _euclidean(A_ext, B_ext):
    sqdist = A_ext.dot(B_ext).clip(min=0)
    return np.sqrt(sqdist)


def _norm(A):
    return A / np.linalg.norm(A, ord=2, axis=1, keepdims=True)


def _cosine(A_norm_ext, B_norm_ext):
    return A_norm_ext.dot(B_norm_ext).clip(min=0) / 2

