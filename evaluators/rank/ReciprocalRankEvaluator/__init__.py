from typing import Sequence, Union, Optional

from jina.executors.evaluators.rank import BaseRankingEvaluator


class ReciprocalRankEvaluator(BaseRankingEvaluator):
    """
    Gives score as per reciprocal rank metric.

    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, actual: Sequence[Union[str, int]], desired: Sequence[Union[str, int]], *args, **kwargs) -> float:
        """
        Evaluate score as per reciprocal rank metric.

        :param actual: Sequence of sorted document IDs.
        :param desired: Sequence of sorted relevant document IDs
            (the first is the most relevant) and the one to be considered.
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: Reciprocal rank score
        """
        if len(actual) == 0 or len(desired) == 0:
            return 0.0
        try:
            return 1.0 / (actual.index(desired[0]) + 1)
        except:
            return 0.0
