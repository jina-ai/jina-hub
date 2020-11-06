from typing import Sequence, Union

import numpy as np

from jina.executors.evaluators.rank import BaseRankingEvaluator


class MeanReciprocalRankEvaluator(BaseRankingEvaluator):
    """
    :class:`MeanReciprocalRankEvaluator` Gives score as per mean reciprocal rank metric.
    """

    def __init__(self, eval_at=3, *args, **kwargs):
        super().__init__(eval_at, *args, **kwargs)

    @property
    def metric(self):
        return f'MeanReciprocalRank@{self.eval_at}'

    def _evaluate(self, actual, desired):
        if len(actual)==0 or len(desired)==0:
            return 0.0
        try:
            return 1.0/(actual[:self.eval_at].index(desired[0])+1)
        except:
            return 0.0

    def evaluate(self, actual: Sequence[Sequence[Union[str,int]]], desired: Sequence[Sequence[Union[str,int]]], *args, **kwargs) -> float:
        """
        :param actual: should be a sequence of sequence of document IDs
        :param desired: should be a sequence of sequence of document IDs
        :return gives mean reciprocal rank score
        """
        if len(actual)==0 or len(desired)==0:
            return 0.0
        return float(np.mean([self._evaluate(actual_i, desired_i) for actual_i, desired_i in zip(actual, desired)]))