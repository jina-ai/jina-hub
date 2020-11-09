import numpy as np

from jina.executors.evaluators.embedding import BaseEmbeddingEvaluator, expand_vector


class L3NormEvaluator(BaseEmbeddingEvaluator):
    """A :class:`L3NormEvaluator` evaluates the distance between actual and desired embeddings computing
    the Minkowski distance ( p=3 ) between them.

        D(x, y) = ( sum_i |x_i - y_i|^p )^(1/p)
    """

    @property
    def metric(self):
        return 'L3NormEvaluator'

    def evaluate(self, actual: 'np.array', desired: 'np.array', *args, **kwargs) -> float:
        """"
        :param actual: the embedding of the document (resulting from an Encoder)
        :param desired: the expected embedding of the document
        :return the evaluation metric value for the request document
        """
        actual = expand_vector(actual)
        desired = expand_vector(desired)
        return _l3norm(_ext_A(actual), _ext_B(desired))


def _get_ones(x, y):
    return np.ones((x, y))


def _ext_A(A):
    nA, dim = A.shape
    A_ext = _get_ones(nA, dim * 4)
    A_ext[:, dim:2 * dim] = A
    A_ext[:, 2 * dim:3 * dim] = A ** 2
    A_ext[:, 3 * dim:] = A ** 3
    return A_ext


def _ext_B(B):
    nB, dim = B.shape
    B_ext = _get_ones(dim * 4, nB)
    B_ext[:dim] = -(B ** 3).T
    B_ext[dim:2 * dim] = 3.0 * (B ** 2).T
    B_ext[2 * dim:3 * dim] = -3.0 * B.T
    del B
    return B_ext


def _l3norm(A_ext, B_ext):
    cubdist = abs(A_ext.dot(B_ext))
    return cubdist ** (1/3)