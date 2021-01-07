from jina.executors.evaluators.text import BaseTextEvaluator


class RougeEvaluator(BaseTextEvaluator):
    """
    :class:`RougeEvaluator` Evaluate Rouge score between acutal and ground truth.
    """

    def __init__(self, metric: str = 'rouge-1', stat: str = 'r', *args, **kwargs):
        """metric: can be rouge-1, rouge-2 or rouge-l
        stat: can be r for recall, p for precision and f for f1
        """
        super().__init__(*args, **kwargs)
        self._metric = metric.lower()
        self.stat = stat.lower()

    def post_init(self):
        super().post_init()
        from rouge import Rouge
        self.rouge = Rouge(metrics=[self._metric], stats=[self.stat])

    def evaluate(self, actual: str, desired: str) -> float:
        if (not len(actual)) or (not len(desired)):
            return 0.0
        return float(self.rouge.get_scores(actual, desired)[0][self._metric][self.stat])
