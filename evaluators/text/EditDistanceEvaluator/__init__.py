from jina.executors.evaluators.text import BaseTextEvaluator


class EditDistanceEvaluator(BaseTextEvaluator):
    """
    :class:`EditDistanceEvaluator` Gives the edit distance between result and groundtruth string..
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, actual: str, desired: str):
        from Levenshtein import distance
        return distance(actual, desired)
