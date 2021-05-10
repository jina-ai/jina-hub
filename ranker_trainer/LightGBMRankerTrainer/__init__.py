
from jina.executors.rankers.trainer import RankerTrainer

class LightGBMRankerTrainer(RankerTrainer):
    """Ranker trainer to train the `LightGBMRanker` to enable offline/online-learning."""

    def __init__(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        """Train ranker based on user feedback, updating ranker weights based on
        the `loss` function.
        :param args: Additional arguments.
        :param kwargs: Additional key value arguments.
        """
        pass

    def save(self):
        """Save the of the ranker model."""
        pass