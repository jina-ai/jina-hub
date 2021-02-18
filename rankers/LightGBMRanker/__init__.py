import os
from typing import Dict, Optional, Tuple, List

import numpy as np

from jina.executors.rankers import Match2DocRanker
from jina.excepts import PretrainedModelFileDoesNotExist

if False:
    import lightgbm


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
                 query_feature_names: Tuple[str] = (
                         'tags__feature-1', 'tags__feature-2', 'tags__feature-3', 'tags__feature-4', 'tags__feature-5'),
                 match_feature_names: Tuple[str] = (
                         'tags__feature-1', 'tags__feature-2', 'tags__feature-3', 'tags__feature-4', 'tags__feature-5'),
                 query_categorical_features: Optional[List[str]] = None,
                 match_categorical_features: Optional[List[str]] = None,
                 query_features_before: bool = True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.query_feature_names = query_feature_names
        self.match_feature_names = match_feature_names
        self.query_categorical_features = query_categorical_features
        self.match_categorical_features = match_categorical_features
        self.query_features_before = query_features_before

    def post_init(self):
        super().post_init()
        if self.model_path and os.path.exists(self.model_path):
            import lightgbm
            self.booster = lightgbm.Booster(model_file=self.model_path)
            model_num_features = self.booster.num_feature()
            expected_num_features = len(self.query_feature_names + self.match_feature_names)
            if model_num_features != expected_num_features:
                raise ValueError(f'The number of features expected by the LightGBM model {model_num_features} is different'
                                 f'than the ones provided in input {expected_num_features}')

        else:
            raise PretrainedModelFileDoesNotExist(f'model {self.model_path} does not exist')

    @property
    def required_keys(self):
        return self.feature_names

    def _get_features_dataset(self, query_meta: Dict, match_meta: Dict) -> 'lightgbm.Dataset':
        import lightgbm

        query_features = np.array(
            [[query_meta[query_id][feat] for feat in self.query_feature_names] for query_id in query_meta])
        query_dataset = lightgbm.Dataset(data=query_features, feature_names=self.query_feature_names,
                                         categorical_features=self.query_categorical_features)
        match_features = np.array(
            [[match_meta[match_id][feat] for feat in self.match_feature_names] for match_id in match_meta])
        match_dataset = lightgbm.Dataset(data=match_features, feature_names=self.match_feature_names,
                                         categorical_features=self.match_categorical_features)
        if self.query_features_before:
            return query_dataset.add_features_from(match_dataset)
        else:
            return match_dataset.add_features_from(query_dataset)

    def score(
            self, query_meta: Dict, old_match_scores: Dict, match_meta: Dict
    ) -> 'np.ndarray':

        dataset = self._get_features_dataset(query_meta=query_meta, match_meta=match_meta)
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
