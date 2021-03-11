from typing import Sequence, Any
import numpy as np
from jina.executors.evaluators.rank import BaseRankingEvaluator


class AveragePrecisionEvaluator(BaseRankingEvaluator):
    """
    Evaluates the Average Precision of the search.
    https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision

    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, actual: Sequence[Any], desired: Sequence[Any], *args, **kwargs) -> float:
        """"
        Evaluate the Average Precision of the search.

        :param actual: the matched document identifiers from the request
            as matched by Indexers and Rankers
        :param desired: A list of all the relevant IDs. All documents
            identified in this list are considered to be relevant
        :return: the evaluation metric value for the request document
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        """

        if len(desired) == 0 or len(actual) == 0:
            return 0.0

        desired_set = set(desired)

        def _precision(eval_at: int):
            if actual[eval_at - 1] not in desired_set:
                return 0.0
            actual_at_k = actual[:eval_at]
            ret = len(set(actual_at_k).intersection(desired_set))
            sub = len(actual_at_k)
            return ret / sub if sub != 0 else 0.

        precisions = list(map(lambda eval_at: _precision(eval_at), range(1, len(actual) + 1)))
        return sum(precisions) / len(desired)
