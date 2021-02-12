import os

import pytest
import numpy as np

NUM_FEATURES = 5
NUM_LABELS = 5
NUM_QUERIES_TRAINING = 5
SIZE_TRAINING_DATA = 500
SIZE_VALIDATION_DATA = 50
NUM_QUERIES_VALIDATION = 5


def _pretrained_model(model_path):
    import lightgbm as lgb
    train_group = np.repeat(NUM_QUERIES_TRAINING, SIZE_TRAINING_DATA / NUM_QUERIES_TRAINING)
    train_features = np.zeros((SIZE_TRAINING_DATA, NUM_FEATURES))  # 500 entities, each contains 5 features
    train_labels = np.random.randint(NUM_LABELS, size=SIZE_TRAINING_DATA)  # 5 relevance labels
    train_data = lgb.Dataset(train_features, group=train_group, label=train_labels)

    # force first feature to strongly correlate with relevance label
    train_features[:, 0] = train_labels

    validation_group = np.repeat(NUM_QUERIES_VALIDATION, SIZE_VALIDATION_DATA / NUM_QUERIES_VALIDATION)
    validation_features = np.zeros((SIZE_VALIDATION_DATA, NUM_FEATURES))  # 500 entities, each contains 5 features
    validation_labels = np.random.randint(5, size=SIZE_VALIDATION_DATA)  # 5 relevance labels
    validation_data = lgb.Dataset(validation_features, group=validation_group, label=validation_labels)

    # force first feature to strongly correlate with relevance label
    validation_features[:, 0] = validation_labels

    param = {'num_leaves': 31, 'objective': 'lambdarank', 'metric': 'ndcg'}
    booster = lgb.train(param, train_data, 2, valid_sets=[validation_data])
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


def test_lambdamart_ranker(pretrained_model):
    from .. import LambdaMartRanker

    feature_names = [f'feature-{i + 1}' for i in range(0, NUM_FEATURES)]
    ranker = LambdaMartRanker(model_path=pretrained_model, feature_names=feature_names)

    match_meta = {
        '1': {
            'feature-1': 1.0,
            'feature-2': 0.0,
            'feature-3': 0.0,
            'feature-4': 0.0,
            'feature-5': 0.0,
        },
        '2': {
            'feature-1': 5.0,
            'feature-2': 0.0,
            'feature-3': 0.0,
            'feature-4': 0.0,
            'feature-5': 0.0,
        },
        '3': {
            'feature-1': 1.0,
            'feature-2': 0.0,
            'feature-3': 0.0,
            'feature-4': 0.0,
            'feature-5': 0.0,
        }
    }

    scores = ranker.score(None, None, match_meta=match_meta).tolist()
    # it does not come sorted
    assert scores[0][0] == '1'
    assert scores[1][0] == '2'
    assert scores[2][0] == '3'

    scores.sort(key=lambda x: x[1], reverse=True)
    assert len(scores) == 3

    # the one with best score is the id 2, because feature-1 is 5
    assert scores[0][0] == '2'
    # id 1 and 3 have the same relevance because they have the same features
    assert scores[1][0] in {'1', '3'}
    assert scores[2][0] in {'1', '3'}
    assert scores[1][0] != scores[2][0]

    assert scores[0][1] > scores[1][1]
    assert scores[1][1] == scores[2][1]
