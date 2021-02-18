from typing import Sequence, Any, Optional

from jina.executors.evaluators.rank import BaseRankingEvaluator


class fScoreEvaluator(BaseRankingEvaluator):
    """
    :class:`fScoreEvaluator` Gives the f score of a search system result. (https://en.wikipedia.org/wiki/F-score)
    :param eval_at: the point at which precision and recall are computed, if None give, will consider all the input to evaluate
    :param beta: Parameter to weight differently precision and recall. When beta is 1, the fScore corresponds to the harmonic mean
        of precision and recall
    """

    def __init__(self, eval_at: Optional[int] = None, beta: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert beta != 0, 'fScore is not defined for beta 0'
        self.eval_at = eval_at
        self.weight = beta**2

    def evaluate(self, actual: Sequence[Any], desired: Sequence[Any], *args, **kwargs) -> float:
        """"
        :param actual: the matched document identifiers from the request as matched by jina indexers and rankers
        :param desired: the expected documents matches
        :return the evaluation metric value for the request document
        """
        if not desired or self.eval_at == 0:
            """TODO: Agree on a behavior"""
            return 0.0

        actual_at_k = actual[:self.eval_at] if self.eval_at else actual
        common_count = len(set(actual_at_k).intersection(set(desired)))
        recall = common_count / len(desired)

        divisor = min(self.eval_at or len(desired), len(desired))

        if divisor != 0.0:
            precision = common_count / divisor
        else:
            precision = 0

        if precision + recall == 0:
            return 0

        return ((1 + self.weight) * precision * recall) / ((self.weight * precision) + recall)
