from typing import Sequence, Any, Optional

from jina.executors.evaluators.rank import BaseRankingEvaluator


class f1ScoreEvaluator(BaseRankingEvaluator):
    """
    :class:`f1ScoreEvaluator` Gives the f1 score of a result given the groundtruth..
    """

    def __init__(self, eval_at: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_at = eval_at

    def evaluate(self, actual: Sequence[Any], desired: Sequence[Any], *args, **kwargs) -> float:
        """"
        :param actual: the matched document identifiers from the request as matched by jina indexers and rankers
        :param desired: the expected documents matches ids sorted as they are expected
        :return the evaluation metric value for the request document
        """
        if not desired or self.eval_at == 0:
            """TODO: Agree on a behavior"""
            return 0.0

        actual_at_k = actual[:self.eval_at] if self.eval_at else actual
        ret = len(set(actual_at_k).intersection(set(desired)))

        recall = ret / len(desired)

        divisor = min(self.eval_at or len(desired), len(desired))

        if divisor != 0.0:
            precision = ret / divisor
        else:
            precision = 0

        if precision + recall == 0:
            return 0

        return 2 * (precision * recall) / (precision + recall)
