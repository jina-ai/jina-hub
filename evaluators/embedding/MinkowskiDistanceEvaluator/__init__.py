import numpy as np

from jina.executors.evaluators.embedding import BaseEmbeddingEvaluator, expand_vector


class MinkowskiDistanceEvaluator(BaseEmbeddingEvaluator):
    """A :class:`MinkowskiDistanceEvaluator` evaluates the distance between actual and desired 
    embeddings computing the Minkowski distance ( p>0 ) between them, which can be considered 
    as a generalization of both the Euclidean distance and the Manhattan distance.

        D(x, y) = ( sum_i |x_i - y_i|^p )^(1/p)
    """
    def __init__(self, order=1, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.order = order

    def evaluate(self, actual: 'np.array', desired: 'np.array', *args, **kwargs) -> float:
        """"
        :param actual: the embedding of the document (resulting from an Encoder)
        :param desired: the expected embedding of the document
        :return the evaluation metric value for the request document
        """
        actual = expand_vector(actual)
        desired = expand_vector(desired)
        return _minkowski_distance(actual, desired, self.order)


def _minkowski_distance(actual, desired, order):
    if order <= 0:
        raise ValueError('Order must be positive.')
    minkowski_dist_exp = np.sum(np.power(np.abs(actual - desired), order))
    return minkowski_dist_exp ** (1/order)