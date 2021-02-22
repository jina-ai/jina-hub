from typing import Sequence, Union, Optional

from jina.executors.evaluators.rank import BaseRankingEvaluator


class ReciprocalRankEvaluator(BaseRankingEvaluator):
    """
    :class:`ReciprocalRankEvaluator` Gives score as per reciprocal rank metric.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, actual: Sequence[Union[str, int]], desired: Sequence[Union[str, int]], *args, **kwargs) -> float:
        """
        :param actual: should be a sequence of sorted document IDs.
        :param desired: It is the sequence of sorted relevant document IDs (the first is the most relevant) and the one to be considered
            by the algorithm.
        :return gives reciprocal rank score
        """
        if len(actual) == 0 or len(desired) == 0:
            return 0.0
        try:
            return 1.0 / (actual.index(desired[0]) + 1)
        except:
            return 0.0
