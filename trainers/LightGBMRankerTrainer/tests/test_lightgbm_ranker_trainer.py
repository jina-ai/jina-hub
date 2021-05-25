import os

import pytest
import numpy as np
import lightgbm as lgb

from .. import LightGBMRankerTrainer

cur_dir = os.path.dirname(os.path.abspath(__file__))


def _pretrained_model(model_path):
    from sklearn.datasets import load_svmlight_file

    X_train, y_train = load_svmlight_file(
        os.path.join(cur_dir, 'training_dataset.train')
    )
    X_train = X_train.todense()
    # randomize pagerank feature so that there is something to be trained
    X_train[:, -1] = np.random.randint(2, size=(25, 1))

    q_train = np.loadtxt(os.path.join(cur_dir, 'training_dataset.train.query'))
    lgb_train = lgb.Dataset(
        np.asarray(X_train),
        y_train,
        group=q_train,
        free_raw_data=False,
        feature_name=['tags__query_length', 'tags__query_language']
        + [
            'tags__document_length',
            'tags__document_language',
            'tags__document_pagerank',
        ],
        params={
            'min_data_in_bin': 1,
            'verbose': 1,
            'max_bin': 2,
            'min_data_in_leaf': 1,
        },
    )

    param = {
        'num_leaves': 2,
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'min_data_in_bin': 1,
        'max_bin': 2,
        'num_trees': 1,
        'learning_rate': 0.05,
        'min_data_in_leaf': 1,
    }
    booster = lgb.train(param, lgb_train, 2, valid_sets=[lgb_train])
    booster.save_model(model_path)
    return model_path


@pytest.fixture
def model(tmpdir):
    """
    Pretrained model to be stored in tmpdir.
    It will be trained on fake data forcing that the first feature correlates positively with relevance, while the rest are
    are random
    :param tmpdir:
    :return:
    """
    model_path = os.path.join(tmpdir, 'model.txt')
    return _pretrained_model(model_path)


@pytest.fixture
def param():
    return {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10],
        'metric_freq': 1,
        'is_training_metric': True,
        'max_bin': 255,
        'num_trees': 100,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'tree_learner': 'serial',
        'feature_fraction': 1.0,
        'min_data_in_leaf': 50,
        'min_sum_hessian_in_leaf': 5.0,
        'is_enable_sparse': True,
    }


@pytest.fixture
def query_metas():
    return [
        {
            'tags__query_length': 1.0,
            'tags__query_language': 0.0,
        },
        {
            'tags__query_length': 1.0,
            'tags__query_language': 0.0,
        },
    ]


@pytest.fixture
def matches_metas():
    return [
        [
            {
                'tags__document_length': 25.0,
                'tags__document_language': 0.0,
                'tags__document_pagerank': 1.0,
                'tags__document_relevance': 0.8,
            },
            {
                'tags__document_length': 4.0,
                'tags__document_language': 0.0,
                'tags__document_pagerank': 3.0,
                'tags__document_relevance': 0.4,
            },
            {
                'tags__document_length': 10.0,
                'tags__document_language': 0.0,
                'tags__document_pagerank': 4.0,
                'tags__document_relevance': 0.6,
            },
        ],
        [
            {
                'tags__document_length': 12.0,
                'tags__document_language': 0.0,
                'tags__document_pagerank': 2.0,
                'tags__document_relevance': 0.7,
            },
            {
                'tags__document_length': 2.0,
                'tags__document_language': 0.0,
                'tags__document_pagerank': 8.0,
                'tags__document_relevance': 0.1,
            },
            {
                'tags__document_length': 6.0,
                'tags__document_language': 0.0,
                'tags__document_pagerank': 5.0,
                'tags__document_relevance': 0.3,
            },
        ],
    ]


def test_ranker_trainer(
    param,
    model,
    query_metas,
    matches_metas,
):
    ranker_trainer = LightGBMRankerTrainer(
        model_path=model,
        params=param,
        query_feature_names=['tags__query_length', 'tags__query_language'],
        match_feature_names=[
            'tags__document_length',
            'tags__document_language',
            'tags__document_pagerank',
        ],
        label_feature_name='tags__document_relevance',
    )
    ranker_trainer.train(query_metas, matches_metas)
    ranker_trainer.save()
    assert model
