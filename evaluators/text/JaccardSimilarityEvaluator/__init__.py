from jina.executors.evaluators.text import BaseTextEvaluator


class JaccardSimilarityEvaluator(BaseTextEvaluator):
    """A:class:`JaccardSimilarityEvaluator` Gives the Jaccard similarity between result and groundtruth string..
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, actual: str, desired: str):
        acutal_words = set(actual.lower().split())
        desired_words = set(desired.lower().split())
        intersection = acutal_words.intersection(desired_words)
        union_length = len(acutal_words) + len(desired_words) - len(intersection)
        return len(intersection) / union_length if union_length != 0.0 else 0.0
