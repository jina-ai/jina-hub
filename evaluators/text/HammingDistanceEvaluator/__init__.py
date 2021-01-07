from jina.executors.evaluators.text import BaseTextEvaluator


class HammingDistanceEvaluator(BaseTextEvaluator):
    """A:class:`HammingDistanceEvaluator` Gives the Hamming distance between result and groundtruth string..
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, actual: str, desired: str):
        if len(actual) != len(desired):
            raise ValueError('Undefined for sequences of unequal length.')
        dist_counter = 0
        for n in range(len(actual)):
            if actual[n] != desired[n]:
                dist_counter += 1
        return dist_counter
