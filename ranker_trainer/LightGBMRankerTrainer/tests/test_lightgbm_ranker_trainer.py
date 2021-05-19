import os

import pytest
import numpy as np
import numpy.testing as npt
import lightgbm as lgb

from .. import LightGBMRankerTrainer

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def train_data():
    data = lgb.Dataset(os.path.join(cur_dir, 'rank.train'))
    group_data = [
        l.strip("\n") for l in open(os.path.join(cur_dir, 'rank.train.query'))
    ]
    data.set_group(group_data)
    return data


@pytest.fixture
def valid_data():
    data = lgb.Dataset(os.path.join(cur_dir, 'rank.test'))
    group_data = [l.strip("\n") for l in open(os.path.join(cur_dir, 'rank.test.query'))]
    data.set_group(group_data)
    return data


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


def get_ndcg_eval_res_at_k(k, eval_res, valid_set):
    return eval_res[valid_set]['ndcg@' + str(k)]


def test_ranker_trainer(
    train_data,
    valid_data,
    param,
):
    ranker_trainer = LightGBMRankerTrainer(
        model_path='test.txt',
        train_set=train_data,
        valid_set=valid_data,
        param=param,
    )
    eval_res = ranker_trainer.train()
    # mearesure ndcg 100 times @ different levels
    # assert ndcg@k increases after training
    valid_zero_at_5 = get_ndcg_eval_res_at_k(5, eval_res, 'valid_0')
    assert valid_zero_at_5[0] <= valid_zero_at_5[50] <= valid_zero_at_5[99]
    valid_zero_at_10 = get_ndcg_eval_res_at_k(10, eval_res, 'valid_0')
    assert valid_zero_at_10[0] <= valid_zero_at_10[50] <= valid_zero_at_10[99]
