import os
from typing import Dict, Optional, Tuple

import numpy as np

from jina.executors.rankers import Match2DocRanker
from jina.excepts import PretrainedModelFileDoesNotExist


class LambdaMartRanker(Match2DocRanker):
    """
    :class:`LightGBMRanker` Computes a relevance score for each match using a pretrained Ltr model trained with LightGBM
    """

    def __init__(self,
                 model_path: Optional[str],
                 feature_names: Tuple[str],
                 *args,
                 **kwargs):
        """
        :param model_path: the path where the model is stored.
        :param feature_names: The feature names that will be stored from the match
        """
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.feature_names = feature_names

    def post_init(self):
        super().post_init()
        if self.model_path and os.path.exists(self.model_path):
            import lightgbm as lgb
            self.booster = lgb.Booster(model_file=self.model_path)
        else:
            raise PretrainedModelFileDoesNotExist(f'model {self.model_path} does not exist')

    @property
    def required_keys(self):
        return self.feature_names

    def _get_features_dataset(self, match_meta: Dict) -> 'np.array':
        return np.array([[match_meta[match_id][feat] for feat in self.feature_names] for match_id in match_meta])

    def score(
            self, query_meta: Dict, old_match_scores: Dict, match_meta: Dict
    ) -> 'np.ndarray':

        dataset = self._get_features_dataset(match_meta=match_meta)
        scores = self.booster.predict(dataset)
        new_scores = [
            (
                match_id,
                scores[id]
            )
            for id, match_id in enumerate(match_meta)
        ]

        return np.array(
            new_scores,
            dtype=[(self.COL_MATCH_ID, np.int64), (self.COL_SCORE, np.float64)],
        )
