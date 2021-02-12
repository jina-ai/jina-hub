import os
from typing import Dict, Optional, Tuple

import numpy as np

from jina.executors.rankers import Match2DocRanker
from jina.excepts import PretrainedModelFileDoesNotExist


class LightGBMRanker(Match2DocRanker):
    """
    Computes a relevance score for each match using a pretrained Ltr model trained with LightGBM (https://lightgbm.readthedocs.io/en/latest/index.html)

    .. note::
        For now all the features are extracted from `match` object by the `class:Match2DocRankerDriver`.

    :param model_path: path to the pretrained model previously trained using LightGBM
    :param feature_names: name of the features to extract from Documents and used to compute relevance scores by the model loaded
    from model_path
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(self,
                 model_path: Optional[str] = 'tmp/model.txt',
                 feature_names: Tuple[str] = ('feature-1', 'feature-2', 'feature-3', 'feature-4', 'feature-5'),
                 *args,
                 **kwargs):
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
            dtype=[(self.COL_MATCH_ID, np.object), (self.COL_SCORE, np.float64)],
        )
