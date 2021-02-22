import os

import pytest
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))

query_features = ['tags__query_length', 'tags__query_language']
match_features = ['tags__document_length', 'tags__document_language', 'tags__document_pagerank']


def _pretrained_model(model_path):
    from sklearn.datasets import load_svmlight_file
    import lightgbm as lgb
    train_group = np.repeat(5, 5)  # 100 matches per query (5 groups of 100 entries)

    X_train, y_train = load_svmlight_file(os.path.join(cur_dir, 'training_dataset.train'))
    q_train = np.loadtxt(os.path.join(cur_dir, 'training_dataset.train.query'))
    lgb_train = lgb.Dataset(X_train, y_train, group=q_train, feature_name=query_features + match_features)

    X_validation, y_validation = load_svmlight_file(os.path.join(cur_dir, 'training_dataset.train'))
    q_validation = np.loadtxt(os.path.join(cur_dir, 'training_dataset.train.query'))
    lgb_validation = lgb.Dataset(X_validation, y_validation, group=q_validation,
                                 feature_name=query_features + match_features)

    param = {'num_leaves': 31, 'objective': 'lambdarank', 'metric': 'ndcg'}
    booster = lgb.train(param, lgb_train, 2, valid_sets=[lgb_validation])
    booster.save_model(model_path)

    return model_path


@pytest.fixture
def pretrained_model(tmpdir):
    """
    Pretrained model to be stored in tmpdir.
    It will be trained on fake data forcing that the first feature correlates positively with relevance, while the rest are
    are random
    :param tmpdir:
    :return:
    """
    model_path = os.path.join(tmpdir, 'model.txt')
    return _pretrained_model(model_path)


def test_lightgbmranker(pretrained_model):
    from .. import LightGBMRanker

    ranker = LightGBMRanker(model_path=pretrained_model, query_feature_names=query_features, match_feature_names=match_features)

    query_meta = {
        'tags__query_length': 1.0,
        'tags__query_language': 0.0,
    }
    match_meta = [
        {
            'tags__document_length': 0.0,
            'tags__document_language': 0.0,
            'tags__document_pagerank': 0.0,
        },
        {
            'tags__document_length': 0.0,
            'tags__document_language': 0.0,
            'tags__document_pagerank': 2.0,
        },
        {
            'tags__document_length': 0.0,
            'tags__document_language': 0.0,
            'tags__document_pagerank': 5.0,
        }
    ]
    scores = ranker.score(query_meta=query_meta, old_match_scores=None, match_meta=match_meta)
    # it does not come sorted
    assert scores.shape == (3,)
    assert scores[0] == 0.0
    assert scores[1] == 0.0
    assert scores[2] == 0.0
