import os

import pytest
import numpy as np
import numpy.testing as npt
import lightgbm as lgb

from .. import LightGBMRankerTrainer


@pytest.fixture
def train_data():
    np.random.seed(0)
    data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
    label = np.random.randint(2, size=500)  # binary target
    return lgb.Dataset(data, label=label)


@pytest.fixture
def param():
    return {'num_leaves': 31, 'objective': 'binary'}


@pytest.fixture
def train_data_online():
    np.random.seed(1)
    data = np.random.rand(100, 10)  # 100 entities, each contains 10 features
    label = np.random.randint(2, size=100)  # binary target
    return lgb.Dataset(data, label=label)


@pytest.fixture
def data_to_predict():
    np.random.seed(2)
    return np.random.rand(2, 10)  # 2 entities, each contains 10 features


@pytest.fixture
def expected_result_before_training():
    return np.array([0.3280431, 0.3768629])


@pytest.fixture
def expected_result_after_training():
    return np.array([0.28399384, 0.54611499])


def test_ranker_trainer(
    train_data,
    param,
    train_data_online,
    data_to_predict,
    expected_result_before_training,
    expected_result_after_training,
    tmpdir,
):
    num_round = 10
    bst = lgb.train(param, train_data, num_round, keep_training_booster=True)
    npt.assert_almost_equal(
        bst.predict(data_to_predict), expected_result_before_training
    )
    model_save_path = os.path.join(tmpdir, 'test.txt')
    bst.save_model(model_save_path)
    ranker_trainer = LightGBMRankerTrainer(
        model_path=model_save_path, train_set=train_data_online, param=param
    )
    ranker_trainer.train()
    ranker_trainer.save()
    assert ranker_trainer.is_trained
    # load model again and assert result
    bst2 = lgb.Booster(model_file=model_save_path)
    npt.assert_almost_equal(
        bst2.predict(data_to_predict), expected_result_after_training
    )
