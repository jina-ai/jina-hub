import os
from typing import Tuple, List, Dict

import lightgbm as lgb

from jina.executors.rankers.trainer import RankerTrainer


class LightGBMRankerTrainer(RankerTrainer):
    """Ranker trainer to train the `LightGBMRanker` to enable offline/online-learning."""

    def __init__(
        self,
        model_path: str,
        param: Dict,
        query_feature_names: Tuple[str],
        match_feature_names: Tuple[str],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = None
        self.param = param
        self.model_path = model_path
        self._is_trained = False

    def post_init(self):
        """Load the model."""
        if os.path.exists(self.model_path):
            self.model = lgb.Booster(model_file=self.model_path)
        else:
            raise FileNotFoundError(f'The model path {self.model_path} not found.')

    def train(self, *args, **kwargs):
        """Train ranker based on user feedback, updating ranker weights based on
        the `loss` function.

        :param args: Additional arguments.
        :param kwargs: Additional key value arguments.
        """
        # self.model = lgb.train(self.param, train_data, 2, valid_sets=[validation_data])
        self._is_trained = True

    def save(self):
        """Save the of the ranker model."""
        if not self.is_trained:
            msg = 'The model has not been trained.'
            msg += 'Will skip the save since the model is the same as the original one.'
            raise ValueError(msg)
        self.model.save_model(self.model_path)

    @property
    def is_trained(self):
        return self._is_trained
