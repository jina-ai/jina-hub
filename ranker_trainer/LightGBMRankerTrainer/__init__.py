import os
from typing import Dict

import lightgbm as lgb

from jina.executors.rankers.trainer import RankerTrainer


class LightGBMRankerTrainer(RankerTrainer):
    """Ranker trainer to train the `LightGBMRanker` to enable offline/online-learning.

    :param model_path: Path to the pretrained model previously trained using LightGBM.
    :param param: Parameters for training.
    :param train_set: Data to be trained on.
    """

    def __init__(
        self,
        model_path: str,
        train_set: lgb.Dataset,
        valid_set: lgb.Dataset,
        param: Dict,
        verbose_eval: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = None
        self.param = param
        self.train_set = train_set
        self.valid_set = valid_set
        self.model_path = model_path
        self.verbose_eval = verbose_eval

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
        """
        eval_res = {}
        self.model = lgb.train(
            train_set=self.train_set,
            valid_sets=[self.valid_set],
            init_model=self.model,
            evals_result=eval_res,
            verbose_eval=self.verbose_eval,
            params=self.param,
            keep_training_booster=True,
        )
        return eval_res

    def save(self):
        """Save the trained lightgbm ranker model."""
        self.model.save_model(self.model_path)
