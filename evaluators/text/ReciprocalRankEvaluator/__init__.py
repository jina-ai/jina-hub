from typing import Sequence

from jina.executors.evaluators.rank import BaseRankingEvaluator

class ReciprocalRankEvaluator(BaseRankingEvaluator):
    """
    :class:`ReciprocalRankEvaluator` Gives score as per reciprocal rank metric.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def metric(self):
        return f'ReciprocalRank@{self.eval_at}'

    def evaluate(self, actual: Sequence[int], desired: Sequence[int], *args, **kwargs) -> float:
        """
        Args:
            actual (Sequence[int]): should be a sequence of document IDs
            desired (Sequence[int]): should be a sequence of document IDs

        Returns:
            float: Returns reciprocal rank score
        """
        if len(actual)==0 or len(desired)==0:
            return 0.0
        try:
            return 1.0/(actual[:self.eval_at].index(desired[0])+1)
        except:
            return 0.0