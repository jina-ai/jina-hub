from typing import Sequence, Union, Optional

from jina.executors.evaluators.rank import BaseRankingEvaluator


class ReciprocalRankEvaluator(BaseRankingEvaluator):
    """
    :class:`ReciprocalRankEvaluator` Gives score as per reciprocal rank metric.
    """

    def __init__(self, eval_at: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_at = eval_at

    def evaluate(self, actual: Sequence[Union[str, int]], desired: Sequence[Union[str, int]], *args, **kwargs) -> float:
        """
        :param actual: should be a sequence of document IDs
        :param desired: should be a sequence of document IDs
        :return gives reciprocal rank score
        """
        if len(actual) == 0 or len(desired) == 0:
            return 0.0
        try:
            actual_at_k = actual[:self.eval_at]
            return 1.0 / (actual_at_k.index(desired[0]) + 1)
        except:
            return 0.0
