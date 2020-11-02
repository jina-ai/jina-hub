from jina.executors.evaluators.rank import BaseRankingEvaluator

class f1ScoreEvaluator(BaseRankingEvaluator):
    """
    :class:`f1ScoreEvaluator` Gives the f1 score of a result given the groundtruth..
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def metric(self):
        return 'f1ScoreDistance'

    def evaluate(self, actual: Sequence[Any], desired: Sequence[Any], *args, **kwargs) -> float:
        """"
        :param actual: the matched document identifiers from the request as matched by jina indexers and rankers
        :param desired: the expected documents matches ids sorted as they are expected
        :return the evaluation metric value for the request document
        """
        if not desired:
            """TODO: Agree on a behavior"""
            return 0.0

        ret = 0.0
        for doc_id in actual[:self.eval_at]:
            if doc_id in desired:
                ret += 1.0

        recall = ret/len(desired)
        divisor = min(self.eval_at, len(desired))
        
        if divisor != 0.0:
            precision = ret / divisor 
        else:
            precision = 0

        return 2*(precision*recall)/(precision+recall)
        