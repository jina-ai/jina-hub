from math import log
from typing import Sequence, Any

from jina.executors.evaluators.rank import BaseRankingEvaluator


def _compute_dcg(gains, power_relevance):
    """Compute discounted cumulative gain."""
    ret = 0.0
    if not power_relevance:
        for score, position in zip(gains[1:], range(2, len(gains) + 1)):
            ret += score / log(position, 2)
        return gains[0] + ret
    for score, position in zip(gains, range(1, len(gains) + 1)):
        ret += (pow(2, score) - 1) / log(position + 1, 2)
    return ret


def _compute_idcg(gains, power_relevance):
    """Compute ideal discounted cumulative gain."""
    sorted_gains = sorted(gains, reverse=True)
    return _compute_dcg(sorted_gains, power_relevance)

class NDCGEvaluator(BaseRankingEvaluator):
    """
    :class:`NDCGEvaluator` evaluates normalized discounted cumulative gain for information retrieval.
    """

    @property
    def metric(self):
        return f'nDCG@{self.eval_at}'

    def evaluate(
            self,
            actual: Sequence[Any],
            desired: Sequence[Any],
            power_relevance=True,
            *args, **kwargs
    ) -> float:
        """"
        :param actual: the scores predicted by the search system.
        :param desired: the expected score given by user as groundtruth.
        :param use_traditional_formula: if use traditional formula.
            The new formula places stronger emphasis on retrieving relevant documents.
            For detailed information, please check https://en.wikipedia.org/wiki/Discounted_cumulative_gain
        :return the evaluation metric value for the request document.
        """
        # Information gain must be greater or equal to 0.
        if any(item < 0 for item in actual) or any(item < 0 for item in desired):
            raise ValueError('One or multiple score is less than 0.')
        # Desired must in desc order.
        if desired != sorted(desired, reverse=True):
            raise ValueError('Information gain of desired must in a desc order.')
        actual_at_k = actual[:self.eval_at]
        desired_at_k = desired[:self.eval_at]
        if len(actual_at_k) < 2:
            raise ValueError(f'Expecting gains at k with minimal length of 2, {len(actual_at_k)} received.')
        if not desired_at_k:
            raise ValueError(f'Expecting desired at k with minimal length of 2, {len(desired_at_k)} received.')
        dcg  = _compute_dcg(gains=actual_at_k, power_relevance=power_relevance)
        idcg = _compute_idcg(gains=desired_at_k, power_relevance=power_relevance)
        return 0.0 if idcg == 0.0 else dcg/idcg

