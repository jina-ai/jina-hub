import os

import pytest
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))

query_features = ['tags__query_length', 'tags__query_language']
match_features = ['tags__document_length', 'tags__document_language', 'tags__document_pagerank']


def _pretrained_model(model_path):
    from sklearn.datasets import load_svmlight_file
    import lightgbm as lgb

    X_train, y_train = load_svmlight_file(os.path.join(cur_dir, 'training_dataset.train'))
    X_train = X_train.todense()
    # randomize pagerank feature so that there is something to be trained
    X_train[:, -1] = np.random.randint(2, size=(25, 1))

    q_train = np.loadtxt(os.path.join(cur_dir, 'training_dataset.train.query'))
    lgb_train = lgb.Dataset(np.asarray(X_train), y_train, group=q_train, free_raw_data=False,
                            feature_name=query_features + match_features,
                            params={'min_data_in_bin': 1, 'verbose': 1, 'max_bin': 2, 'min_data_in_leaf': 1})

    param = {'num_leaves': 2, 'objective': 'lambdarank', 'metric': 'ndcg', 'min_data_in_bin': 1, 'max_bin': 2,
             'num_trees': 1,
             'learning_rate': 0.05, 'min_data_in_leaf': 1}
    booster = lgb.train(param, lgb_train, 2, valid_sets=[lgb_train])
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

    ranker = LightGBMRanker(model_path=pretrained_model, query_feature_names=query_features,
                            match_feature_names=match_features)

    queries_metas = [
        {
            'tags__query_length': 1.0,
            'tags__query_language': 0.0,
        },
        {
            'tags__query_length': 1.0,
            'tags__query_language': 0.0,
        }
    ]
    matches_metas = [
        [
            {
                'tags__document_length': 0.0,
                'tags__document_language': 0.0,
                'tags__document_pagerank': 0.0,
            },
            {
                'tags__document_length': 0.0,
                'tags__document_language': 0.0,
                'tags__document_pagerank': 1.0,
            },
            {
                'tags__document_length': 0.0,
                'tags__document_language': 0.0,
                'tags__document_pagerank': 0.0,
            }
        ],
        [
            {
                'tags__document_length': 0.0,
                'tags__document_language': 0.0,
                'tags__document_pagerank': 1.0,
            },
            {
                'tags__document_length': 0.0,
                'tags__document_language': 0.0,
                'tags__document_pagerank': 0.0,
            },
            {
                'tags__document_length': 0.0,
                'tags__document_language': 0.0,
                'tags__document_pagerank': 1.0,
            }
        ]
    ]
    scores = ranker.score([None, None], queries_metas, matches_metas)
    # it does not come sorted, we know that the ones with the same `tags_document_pagerank` have the same score
    # because the model did only split based on that feature
    assert scores.shape == (6,)
    assert scores[0] == scores[2]
    assert scores[0] == scores[4]
    assert scores[1] == scores[3]
    assert scores[1] == scores[5]
    assert scores[0] != scores[1]
