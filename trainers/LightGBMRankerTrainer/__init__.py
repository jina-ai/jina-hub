import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import lightgbm as lgb

from jina.executors.rankers.trainer import RankerTrainer


class LightGBMRankerTrainer(RankerTrainer):
    """Ranker trainer to train the `LightGBMRanker` to enable offline/online-learning.

    :param model_path: path to the pretrained model previously trained using LightGBM。
    :param params: parameter to retrain the lightgbm ranker.
    :param query_feature_names: name of the features to extract from query Documents and used to compute relevance scores by the model loaded
    from model_path
    :param match_feature_names: name of the features to extract from match Documents and used to compute relevance scores by the model loaded
    from model_path
    :param label_feature_name: The column name for labels/groundtruth for ranker training.
    :param query_categorical_features: name of features contained in `query_feature_names` corresponding to categorical features.
    :param match_categorical_features: name of features contained in `match_feature_names` corresponding to categorical features.
    :param query_features_before: True if `query_feature_names` must be placed before the `match` ones in the `dataset` used for prediction.
    :param args: Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        model_path: Optional[str] = 'tmp/model.txt',
        query_feature_names: Tuple[str] = ['tags__query_length', 'tags__query_language'],
        match_feature_names: Tuple[str] = ['tags__document_length', 'tags__document_language',
                                           'tags__document_pagerank'],
        params: Dict = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'lambdarank',},
        label_feature_name: str = 'tags__relevance',
        query_categorical_features: Optional[List[str]] = None,
        match_categorical_features: Optional[List[str]] = None,
        query_features_before: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = None
        self.params = params
        self.model_path = model_path
        self.query_feature_names = query_feature_names
        self.match_feature_names = match_feature_names
        self.query_categorical_features = query_categorical_features
        self.match_categorical_features = match_categorical_features
        self.query_features_before = query_features_before
        self.label_feature_name = label_feature_name

    def post_init(self):
        """Load the model."""
        super().post_init()
        if self.model_path and os.path.exists(self.model_path):
            self.model = lgb.Booster(model_file=self.model_path)
            model_num_features = self.model.num_feature()
            expected_num_features = len(
                self.query_feature_names + self.match_feature_names
            )
            if model_num_features != expected_num_features:
                raise ValueError(
                    f'The number of features expected by the LightGBM model {model_num_features} is different'
                    f'than the ones provided in input {expected_num_features}'
                )

    def _get_features_dataset(self, query_metas, matches_metas):
        def _get_features_per_query(q_meta, m_meta):
            query_features = [
                [q_meta[feat] for feat in self.query_feature_names]
                for _ in range(0, len(m_meta))
            ]

            match_features = [
                [meta[feat] for feat in self.match_feature_names] for meta in m_meta
            ]
            return query_features, match_features

        q_features, m_features, groups = [], [], []
        for q_meta, m_meta in zip(query_metas, matches_metas):
            q_f, m_f = _get_features_per_query(q_meta, m_meta)
            groups.append(len(m_meta))
            q_features.append(q_f)
            m_features.append(m_f)

        query_features = np.vstack(q_features)
        labels = self._get_labels(matches_metas)
        query_dataset = lgb.Dataset(
            data=query_features,
            group=groups,
            feature_name=self.query_feature_names,
            categorical_feature=self.query_categorical_features,
            free_raw_data=False,
        )
        match_features = np.vstack(m_features)
        match_dataset = lgb.Dataset(
            data=match_features,
            group=groups,
            label=labels,
            feature_name=self.match_feature_names,
            categorical_feature=self.match_categorical_features,
            free_raw_data=False,
        )
        if self.query_features_before:
            return query_dataset.construct().add_features_from(
                match_dataset.construct()
            )
        else:
            return match_dataset.construct().add_features_from(
                query_dataset.construct()
            )

    def _get_labels(self, matches_metas):
        rv = []
        for m_meta in matches_metas:
            labels = [item[self.label_feature_name] for item in m_meta]
            rv.extend(labels)
        return rv

    def train(
        self,
        query_metas: List[Dict],
        matches_metas: List[List[Dict]],
        *args,
        **kwargs,
    ):
        """Train ranker based on user feedback, updating ranker in an incremental fashion.

        This function make use of lightgbm `train` to update the tree structure through continued
        training.

        :param query_metas: List of features extracted from query `Document`. Extracted according to `query_feature_names`.
            The length of the list is the number of features for which the scores will be computed
        :param matches_metas: List of list of features extracted from match `Document`. Extracted according to `match_feature_names`.
            The list's length is equal to the number of queries, and each element is a list of the number of matches.
        :param args: Additional arguments.
        :param kwargs: Additional key value arguments.
        """
        train_set = self._get_features_dataset(query_metas, matches_metas)
        self.model = lgb.train(
            train_set=train_set,
            init_model=self.model,
            params=self.params,
            keep_training_booster=True,
        )

    def save(self):
        """Save the trained lightgbm ranker model."""
        if self.model:
            self.model.save_model(self.model_path)
