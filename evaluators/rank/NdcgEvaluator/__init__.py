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
    From a sorted list of retrieved, scores and scores,
    evaluates normalized discounted cumulative gain for information retrieval.

    :param eval_at: The number of documents in each of the lists to consider
        in the NDCG computation. If ``None``is given, the complete lists are
        considered for the evaluation.
    :param power_relevance: The power relevance places stronger emphasis on
        retrieving relevant documents. For detailed information, please check
        https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    :param is_relevance_score: Boolean indicating if the actual scores are
        to be considered relevance. Highest value is better.
        If True, the information coming from the actual system results will
        be sorted in descending order, otherwise in ascending order.
        Since the input of the evaluate method is sorted according to the
        `scores` of both actual and desired input, this parameter is
        useful for instance when the ``matches` come directly from a ``VectorIndexer``
        where score is `distance` and therefore the smaller the better.

    .. note:
        All the IDs that are not found in the ground truth will be considered to have
        relevance 0.
    """

    def __init__(self,
                 eval_at: Optional[int] = None,
                 power_relevance: bool = True,
                 is_relevance_score: bool = True):
        super().__init__()
        self._eval_at = eval_at
        self._power_relevance = power_relevance
        self._is_relevance_score = is_relevance_score

    def evaluate(
            self,
            actual: Sequence[Tuple[Any, Union[int, float]]],
            desired: Sequence[Tuple[Any, Union[int, float]]],
            *args, **kwargs
    ) -> float:
        """"
        Evaluate normalized discounted cumulative gain for information retrieval.

        :param actual: The tuple of Ids and Scores predicted by the search system.
            They will be sorted in descending order.
        :param desired: The expected id and relevance tuples given by user as
            matching round truth.
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: The evaluation metric value for the request document.
        """
        relevances = dict(desired)
        actual_relevances = list(map(lambda x: relevances[x[0]] if x[0] in relevances else 0.,
                                     sorted(actual, key=lambda x: x[1], reverse=self._is_relevance_score)))
        desired_relevances = list(map(lambda x: x[1], sorted(desired, key=lambda x: x[1], reverse=True)))

        # Information gain must be greater or equal to 0.
        actual_at_k = actual_relevances[:self._eval_at] if self._eval_at else actual
        desired_at_k = desired_relevances[:self._eval_at] if self._eval_at else desired
        if not actual_at_k:
            raise ValueError(f'Expecting gains at k with minimal length of 1, {len(actual_at_k)} received.')
        if not desired_at_k:
            raise ValueError(f'Expecting desired at k with minimal length of 1, {len(desired_at_k)} received.')
        if any(item < 0 for item in actual_at_k) or any(item < 0 for item in desired_at_k):
            raise ValueError('One or multiple score is less than 0.')
        dcg = _compute_dcg(gains=actual_at_k, power_relevance=self._power_relevance)
        idcg = _compute_idcg(gains=desired_at_k, power_relevance=self._power_relevance)
        return 0.0 if idcg == 0.0 else dcg / idcg
