import os

import pytest
import numpy as np

from .. import LambdaMartRanker

NUM_FEATURES = 5
NUM_LABELS = 5
NUM_QUERIES_TRAINING = 5
SIZE_TRAINING_DATA = 500
SIZE_VALIDATION_DATA = 50
NUM_QUERIES_VALIDATION = 5


@pytest.fixture
def pretrained_model(tmpdir):
    import lightgbm as lgb
    model_path = os.path.join(tmpdir, 'model.txt')

    train_group = np.repeat(NUM_QUERIES_TRAINING, SIZE_TRAINING_DATA/NUM_QUERIES_TRAINING)
    train_features = np.random.rand(SIZE_TRAINING_DATA, NUM_FEATURES)  # 500 entities, each contains 5 features
    train_labels = np.random.randint(NUM_LABELS, size=500)  # 5 relevance labels
    train_data = lgb.Dataset(train_features, group=train_group, label=train_labels)

    validation_group = np.repeat(NUM_QUERIES_VALIDATION, SIZE_VALIDATION_DATA/NUM_QUERIES_VALIDATION)
    validation_features = np.random.rand(SIZE_VALIDATION_DATA, 5)  # 500 entities, each contains 5 features
    validation_labels = np.random.randint(5, size=50)  # 5 relevance labels
    validation_data = lgb.Dataset(validation_features, group=validation_group, label=validation_labels)

    param = {'num_leaves': 31, 'objective': 'lambdarank', 'metric': 'ndcg'}
    booster = lgb.train(param, train_data, 2, valid_sets=[validation_data])
    booster.save_model(model_path)

    return model_path


def test_lambdamart_ranker(pretrained_model):
    feature_names = [f'feature-{i + 1}' for i in range(0, NUM_FEATURES)]
    ranker = LambdaMartRanker(model_path=pretrained_model, feature_names=feature_names)

    match_meta = {
        1: {
            'feature-1': 1.0,
            'feature-2': 2.0,
            'feature-3': 3.0,
            'feature-4': 4.0,
            'feature-5': 5.0,
            },
        2: {
            'feature-1': 5.0,
            'feature-2': 4.0,
            'feature-3': 3.0,
            'feature-4': 2.0,
            'feature-5': 1.0,
        }
    }

    scores = ranker.score(None, None, match_meta=match_meta)
    print(f' scores {scores}')