__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Tuple, Dict

import numpy as np

from jina import Executor, requests, DocumentArray, Document
from jina.hub.indexers.dump import import_vectors


"""
potential TODO:
- _validate_key_vector_shapes on query
- sample
- query_by_id
- support euclidena/cosine etc.
"""


class NumpyIndexer(Executor):
    def __init__(self, source_path: str, **kwargs):
        super().__init__(**kwargs)
        ids, vecs = import_vectors(source_path, str(self.metas.pea_id))
        self._ids = np.array(list(ids))
        self._vecs = np.array(list(vecs))
        self._ids_to_idx = {}

    @requests(on='/search')
    def search(self, docs: 'DocumentArray', parameters: Dict = None, **kwargs):
        if parameters is None:
            parameters = {'top_k': 5}
        doc_embeddings = np.stack(docs.get_attributes('embedding'))
        q_emb = _ext_A(_norm(doc_embeddings))
        d_emb = _ext_B(_norm(self._vecs))
        dists = _cosine(q_emb, d_emb)
        positions, dist = self._get_sorted_top_k(dists, int(parameters['top_k']))
        for _q, _positions, _dists in zip(docs, positions, dist):
            for position, _dist in zip(_positions, _dists):
                d = Document(id=self._ids[position], embedding=self._vecs[position])
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
    A_ext[:, dim : 2 * dim] = A
    A_ext[:, 2 * dim :] = A ** 2
    return A_ext


def _ext_B(B):
    nB, dim = B.shape
    B_ext = _get_ones(dim * 3, nB)
    B_ext[:dim] = (B ** 2).T
    B_ext[dim : 2 * dim] = -2.0 * B.T
    del B
    return B_ext


def _euclidean(A_ext, B_ext):
    sqdist = A_ext.dot(B_ext).clip(min=0)
    return np.sqrt(sqdist)


def _norm(A):
    return A / np.linalg.norm(A, ord=2, axis=1, keepdims=True)


def _cosine(A_norm_ext, B_norm_ext):
    return A_norm_ext.dot(B_norm_ext).clip(min=0) / 2
