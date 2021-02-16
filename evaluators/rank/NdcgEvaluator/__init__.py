from typing import Sequence, Union, Optional, Tuple, Any
from math import log

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
    From a list of sorted retrieved sorted scores and expected scores, evaluates normalized discounted cumulative gain for information retrieval.
    :param eval_at: The number of documents in each of the lists to consider in the NDCG computation. If None. the complete lists are considered
        for the evaluation computation
    :param power_relevance: The power relevance places stronger emphasis on retrieving relevant documents.
        For detailed information, please check https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    """

    def __init__(self, eval_at: Optional[int] = None, power_relevance: bool = True):
        super().__init__()
        self.eval_at = eval_at
        self._power_relevance = power_relevance

    def evaluate(
            self,
            actual: Sequence[Tuple[Any, Union[int, float]]],
            desired: Sequence[Tuple[Any, Union[int, float]]],
            *args, **kwargs
    ) -> float:
        """"
        :param actual: the tuple of Ids and Scores predicted by the search system. actual is sorted in descending order
        :param desired: the expected id, relevance tuples given by user as matching groundtruth.
        :return the evaluation metric value for the request document.
        """

        # Information gain must be greater or equal to 0.
        actual_at_k = actual[:self.eval_at] if self.eval_at else actual
        desired_at_k = desired[:self.eval_at] if self.eval_at else desired
        if not actual_at_k:
            raise ValueError(f'Expecting gains at k with minimal length of 1, {len(actual_at_k)} received.')
        if not desired_at_k:
            raise ValueError(f'Expecting desired at k with minimal length of 1, {len(desired_at_k)} received.')
        if any(item < 0 for item in actual_at_k) or any(item < 0 for item in desired_at_k):
            raise ValueError('One or multiple score is less than 0.')
        dcg = _compute_dcg(gains=actual_at_k, power_relevance=self._power_relevance)
        idcg = _compute_idcg(gains=desired_at_k, power_relevance=self._power_relevance)
        return 0.0 if idcg == 0.0 else dcg / idcg
