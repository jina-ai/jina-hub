# LightGBMRanker

This ranker allows the usage of any learning-to-rank model trained using [LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html) to rank a list of search results.

In order to use this executor, these are the 2 needed parameters:
 
 - A local path from where to load a model trained using LightGBM (usually using `lambdarank` or any other ranking loss function).
 - The features to be extracted from matches which are to be used when predicting relevance scores. It is important to provide them in the order
 in which the features are expected to be fed to the `LightGBMBooster` (https://lightgbm.readthedocs.io/en/latest/Python-API.html)
 
### Usage example.

A simple example on how one would use it in Python.

```python
import numpy as np
import lightgbm as lgb
model_path = 'lightgbm-model.txt'

train_features = np.random.rand(500, 3)  # 500 entities, each contains 3 features
train_labels = np.random.randint(5, size=500)  # 5 relevance labels
train_data = lgb.Dataset(train_features,label=train_labels)
param = {'num_leaves': 31, 'objective': 'lambdarank', 'metric': 'ndcg'}
booster = lgb.train(param, train_data, 2, valid_sets=[validation_data])
booster.save_model(model_path)

ranker = LightGBMRanker(model_path=model_path, feature_names=['tags__price', 'tags__size', 'tags__brand'])
```

This is how it would look a `yaml` configuration to be loaded inside a Jina Flow.

```yaml
!LightGBMRanker
with:
  model_path: './lightgbm-model.txt'
  feature_names: ['tags__price', 'tags__size', 'tags__brand']
metas:
  py_modules:
    - __init__.py
```

## Image Usage

Users can use Pod images in several ways:

1. Run with Docker (`docker run`)
   ```bash
    docker run --network host jinahub/pod.ranker.LightGBMRanker:0.0.1-1.0.2 --port-in 55555 --port-out 55556
    ```
    
2. Run with Flow API
   ```python
    from jina.flow import Flow

    f = (Flow()
        .add(name='my_ranker', uses='docker://jinahub/pod.ranker.LightGBMRanker:0.0.1-1.0.2'))
    ```
    
3. Run with Jina CLI
   ```bash
    jina pod --uses docker:jinahub/pod.ranker.LightGBMRanker:0.0.1-1.0.2
    ```
    
4. Conventional local usage with `uses` argument
    ```bash
    jina pod --uses lightgbmranker.yml --port-in 55555 --port-out 55556
    ```
