import os
from typing import Dict

import lightgbm as lgb

from jina.executors.rankers.trainer import RankerTrainer


class LightGBMRankerTrainer(RankerTrainer):
    """Ranker trainer to train the `LightGBMRanker` to enable offline/online-learning.

    :param model_path: Path to the pretrained model previously trained using LightGBM.
    :param param: Parameters for training.
    :param train_set: Data to be trained on.
    :param num_boost_round: Number of boosting iterations.
    :param valid_sets: List of data to be evaluated on during training.
    :param valid_names: Names of valid_sets.
    """

    def __init__(
        self,
        model_path: str,
        train_set: lgb.Dataset,
        param: Dict,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = None
        self.param = param
        self.train_set = train_set
        self.model_path = model_path
        self._is_trained = False

    def post_init(self):
        """Load the model."""
        super().post_init()
        if os.path.exists(self.model_path):
            self.model = lgb.Booster(model_file=self.model_path)
        else:
            msg = f'The model path {self.model_path} not found.'
            msg += 'Will train from scratch once you call `train` method!'
            self.logger.info(msg)

    def train(self, *args, **kwargs):
        """Train ranker based on user feedback, updating ranker in an incremental fashion.

        This function make use of lightgbm `train` to update the tree structure through continued
        training.

        :param args: Additional arguments.
        :param kwargs: Additional key value arguments.
        :return: Whether the update was successfully finished, 0 succeed, 1 failed.
        """
        self.model = lgb.train(
            train_set=self.train_set,
            init_model=self.model,
            keep_training_booster=True,
            params=self.param,
        )
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
