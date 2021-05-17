import os
import pytest
import numpy as np
import lightgbm as lgb

from .. import LightGBMRankerTrainer


@pytest.fixture
def train_data():
    data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
    label = np.random.randint(2, size=500)  # binary target
    return lgb.Dataset(data, label=label)


@pytest.fixture
def param():
    return {'num_leaves': 31, 'objective': 'binary'}


@pytest.fixture
def train_data_online():
    data = np.random.rand(100, 10)  # 100 entities, each contains 10 features
    label = np.random.randint(2, size=100)  # binary target
    return lgb.Dataset(data, label=label)


def test_ranker_trainer(train_data, param, train_data_online, tmpdir):
    num_round = 10
    bst = lgb.train(param, train_data, num_round, keep_training_booster=True)
    model_save_path = os.path.join(tmpdir, 'test.txt')
    bst.save_model(model_save_path)
    ranker_trainer = LightGBMRankerTrainer(
        model_path=model_save_path, train_set=train_data_online, param=param
    )
    is_success = ranker_trainer.train()
    ranker_trainer.save()
    assert ranker_trainer.is_trained
