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
                 query_feature_names: Tuple[str] = ['tags__query_length', 'tags__query_language'],
                 match_feature_names: Tuple[str] = ['tags__document_length', 'tags__document_language', 'tags__document_pagerank'],
                 query_categorical_features: Optional[List[str]] = None,
                 match_categorical_features: Optional[List[str]] = None,
                 query_features_before: bool = True,
                 *args,
                 **kwargs):
        super().__init__(query_required_keys=query_feature_names, match_required_keys=match_feature_names, *args, **kwargs)
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

    def _get_features_dataset(self, query_meta: Dict, match_meta: Dict) -> 'lightgbm.Dataset':
        import lightgbm
        query_features = np.array(
            [[query_meta[feat] for feat in self.query_feature_names] for _ in range(0, len(match_meta))])
        query_dataset = lightgbm.Dataset(data=query_features, feature_name=self.query_feature_names,
                                         categorical_feature=self.query_categorical_features, free_raw_data=False)

        match_features = np.array(
            [[meta[feat] for feat in self.match_feature_names] for meta in match_meta])
        match_dataset = lightgbm.Dataset(data=match_features, feature_name=self.match_feature_names,
                                         categorical_feature=self.match_categorical_features, free_raw_data=False)
        if self.query_features_before:
            return query_dataset.construct().add_features_from(match_dataset.construct())
        else:
            return match_dataset.construct().add_features_from(query_dataset.construct())

    def score(
            self, query_meta: Dict, old_match_scores: Dict, match_meta: Dict
    ) -> 'np.ndarray':

        dataset = self._get_features_dataset(query_meta=query_meta, match_meta=match_meta)
        return self.booster.predict(dataset.get_data())
