from typing import Sequence

from rouge import Rouge

from jina.executors.evaluators.text import BaseTextEvaluator

class RougeEvaluator(BaseTextEvaluator):
    """
    :class:`RougeEvaluator` Evaluate Rouge score between acutal and ground truth.
    """

    def __init__(self, metrics: str='rouge-1', stats: str='r', *args, **kwargs):
        """metrics: can be rouge-1, rouge-2 or rouge-l
        stats: can be r, p or f
        """
        super().__init__(*args, **kwargs)
        self.metrics = metrics
        self.stats = stats
        self.rouge = Rouge(metrics=[metrics], stats=[stats])

    def _get_score(self, actual_, desired_):
        if (not len(actual_)) or (not len(desired_)):
            return 0.0
        return self.rouge.get_scores(actual_, desired_)[0][self.metrics][self.stats]

    def evaluate(self, actual: Sequence[str], desired: Sequence[str]) -> float:
        return sum(self._get_score(actual_, desired_) for actual_, desired_ in zip(actual, desired))