import os
from typing import List, Dict, Union

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
        params: Dict,
        train_set: lgb.Dataset,
        num_boost_round: int = 100,
        valid_sets: Union[None, List[lgb.Dataset]] = None,
        valid_names: List[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = None
        self.params = params
        self.train_set = train_set
        self.num_boost_round = num_boost_round
        self.valid_sets = valid_sets
        self.valid_names = valid_names
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
        self.model = lgb.train(
            params=self.params,
            train_set=self.train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=self.valid_sets,
            valid_names=self.valid_names,
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
