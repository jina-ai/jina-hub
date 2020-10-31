from typing import Sequence

from rouge import Rouge

from jina.executors.evaluators.text import BaseTextEvaluator

class RougeEvaluator(BaseTextEvaluator):
    """
    :class:`RougeEvaluator` Evaluate Rouge score between acutal and ground truth.
    """

    def __init__(self, metric: str='rouge-1', stat: str='r', *args, **kwargs):
        """metric: can be rouge-1, rouge-2 or rouge-l
        stat: can be r, p or f
        """
        super().__init__(*args, **kwargs)
        self.metric_ = metric
        self.stat = stat
        self.rouge = Rouge(metrics=[metric], stats=[stat])

    @property
    def metric(self):
        return f'{self.metric_.upper()}'

    def _get_score(self, actual_, desired_):
        if (not len(actual_)) or (not len(desired_)):
            return 0.0
        return self.rouge.get_scores(actual_, desired_)[0][self.metric_][self.stat]

    def evaluate(self, actual: Sequence[str], desired: Sequence[str]) -> float:
        return sum(self._get_score(actual_, desired_) for actual_, desired_ in zip(actual, desired))