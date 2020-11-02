from typing import Sequence, Union

from jina.executors.evaluators.rank import BaseRankingEvaluator

class ReciprocalRankEvaluator(BaseRankingEvaluator):
    """
    :class:`ReciprocalRankEvaluator` Gives score as per reciprocal rank metric.
    """

    def __init__(self, eval_at=3, *args, **kwargs):
        super().__init__(eval_at, *args, **kwargs)

    @property
    def metric(self):
        return f'ReciprocalRank@{self.eval_at}'

    def evaluate(self, actual: Sequence[Union[str,int]], desired: Sequence[Union[str,int]], *args, **kwargs) -> float:
        """
        :param actual: should be a sequence of document IDs
        :param desired: should be a sequence of document IDs
        :return gives reciprocal rank score
        """
        if len(actual)==0 or len(desired)==0:
            return 0.0
        try:
            return 1.0/(actual[:self.eval_at].index(desired[0])+1)
        except:
            return 0.0