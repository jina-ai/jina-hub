from jina.executors.evaluators.text import BaseTextEvaluator

class JaccardSimilarityEvaluator(BaseTextEvaluator):
    """A:class:`JaccardSimilarityEvaluator` Gives the Jaccard similarity between result and groundtruth string..
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def metric(self):
        return 'JaccardSimilarity'

    def evaluate(self, actual: str, desired: str):
        a = set(actual.lower().split())
        b = set(desired.lower().split())
        c = a.intersection(b)
        total = len(a) + len(b) - len(c)
        return float(len(c)) / total if total != 0.0 else 0.0
    