from typing import Sequence, Any

from jina.executors.evaluators import BaseRankingEvaluator


class NDCGEvaluator(BaseRankingEvaluator):
    """
    :class:`NDCGEvaluator` evaluates normalized discounted cumulative gain for information retrieval.
    """

    @property
    def metric(self):
        return f'nDCG@{self.eval_at}'

    def evaluate(self, actual: Sequence[Any], desired: Sequence[Any], *args, **kwargs) -> float:
        """"
        :param actual: the matched document identifiers from the request as matched by jina indexers and rankers
        :param desired: the expected documents matches ids sorted as they are expected
        :return the evaluation metric value for the request document
        """
        ret = 0.0
        pass
