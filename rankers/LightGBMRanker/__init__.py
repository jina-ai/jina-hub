import os
from typing import Dict, Optional, Tuple, List, Union

import numpy as np

from jina.executors.decorators import batching
from jina.executors.rankers import Match2DocRanker
from jina.excepts import PretrainedModelFileDoesNotExist

if False:
    import lightgbm


class LightGBMRanker(Match2DocRanker):
    """
    Computes a relevance score for each match using a pretrained Ltr model trained with LightGBM (https://lightgbm.readthedocs.io/en/latest/index.html)

    :param model_path: path to the pretrained model previously trained using LightGBM
    :param query_feature_names: name of the features to extract from query Documents and used to compute relevance scores by the model loaded
    from model_path
    :param match_feature_names: name of the features to extract from match Documents and used to compute relevance scores by the model loaded
    from model_path
    :param query_categorical_features: name of features contained in `query_feature_names` corresponding to categorical features.
    :param match_categorical_features: name of features contained in `match_feature_names` corresponding to categorical features.
    :param query_features_before: True if `query_feature_names` must be placed before the `match` ones in the `dataset` used for prediction.
    :param args: Additional positional arguments
    :param kwargs: Additional keyword arguments

    .. note::
        The name of the features are used to extract the features from incoming `documents`. Check how these features are accessed in
        :class:`Document` at https://docs.jina.ai/api/jina.types.document/

    """

    def __init__(self,
                 model_path: Optional[str] = 'tmp/model.txt',
                 query_feature_names: Tuple[str] = ['tags__query_length', 'tags__query_language'],
                 match_feature_names: Tuple[str] = ['tags__document_length', 'tags__document_language',
                                                    'tags__document_pagerank'],
                 query_categorical_features: Optional[List[str]] = None,
                 match_categorical_features: Optional[List[str]] = None,
                 query_features_before: bool = True,
                 *args,
                 **kwargs):
        super().__init__(query_required_keys=query_feature_names, match_required_keys=match_feature_names, *args,
                         **kwargs)
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
                raise ValueError(
                    f'The number of features expected by the LightGBM model {model_num_features} is different'
                    f'than the ones provided in input {expected_num_features}')
        else:
            raise PretrainedModelFileDoesNotExist(f'model {self.model_path} does not exist')

    def _get_features_dataset(self, query_meta: List[Dict], match_meta: List[List[Dict]]) -> 'lightgbm.Dataset':
        import lightgbm

        def _get_features_per_query(q_meta, m_meta):
            query_features = np.array(
                [[q_meta[feat] for feat in self.query_feature_names] for _ in range(0, len(m_meta))])
            match_features = np.array([[meta[feat] for feat in self.match_feature_names] for meta in m_meta])
            return query_features, match_features

        q_features, m_features = [], []
        for q_meta, m_meta in zip(query_meta, match_meta):
            q_f, m_f = _get_features_per_query(q_meta, m_meta)
            q_features.append(q_f)
            m_features.append(m_f)

        query_features = np.vstack(q_features)
        query_dataset = lightgbm.Dataset(data=query_features, feature_name=self.query_feature_names,
                                         categorical_feature=self.query_categorical_features, free_raw_data=False)

        match_features = np.vstack(m_features)
        match_dataset = lightgbm.Dataset(data=match_features, feature_name=self.match_feature_names,
                                         categorical_feature=self.match_categorical_features, free_raw_data=False)
        if self.query_features_before:
            return query_dataset.construct().add_features_from(match_dataset.construct())
        else:
            return match_dataset.construct().add_features_from(query_dataset.construct())

    @batching(slice_nargs=3)
    def score(
            self, old_match_scores: List[List[float]], query_meta: List[Dict], match_meta: List[List[Dict]]
    ) -> 'np.ndarray':
        """
            Computes a relevance score for each match using a pretrained Ltr model trained with LightGBM (https://lightgbm.readthedocs.io/en/latest/index.html)

        :param query_meta: List of features extracted from query `Document`. Extracted according to `query_feature_names`.
            The length of the list is the number of features for which the scores will be computed
        :param old_match_scores: Previous scores assigned to query match pairs, unused in this case.
        :param match_meta: List of list of features extracted from match `Document`. Extracted according to `match_feature_names`.
            The list's length is equal to the number of queries, and each element is a list of the number of matches.
        :return:
        """

        dataset = self._get_features_dataset(query_meta=query_meta, match_meta=match_meta)
        return self.booster.predict(dataset.get_data())
