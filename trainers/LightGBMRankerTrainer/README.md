# LightGBMRankerTrainer

This ranker trainer allows the incremental learning of any learning-to-rank model trained using [LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html) to rank a list of search results.
The RankerTrainer is corresponded to the [LightGBMRanker](https://github.com/jina-ai/jina-hub/tree/master/rankers/LightGBMRanker) in Jina Hub.


### Usage example.

This is how it would look a `yaml` configuration to be loaded inside a Jina Flow.

```yaml
jtype: LightGBMRankerTrainer
with:
  model_path: './lightgbm-model.txt'
  query_feature_names: ['tags__price', 'tags__size', 'tags__brand']
  match_feature_names: ['tags__price', 'tags__size', 'tags__brand']
  label_feature_name: ['tags__relevance']
metas:
  py_modules:
    - __init__.py
```

## Image Usage

Users can use Pod images in several ways:

1. Run with Docker (`docker run`)
```bash
docker run --network host jinahub/pod.trainer.LightGBMRankerTrainer:{VERSION} --port-in 55555 --port-out 55556
```
    
2. Run with Flow API
```python
from jina.flow import Flow

f = (Flow()
    .add(name='my_trainer', uses='docker://jinahub/pod.trainer.LightGBMRankerTrainer:{VERSION}'))
```
    
3. Run with Jina CLI
```bash
jina pod --uses docker:jinahub/pod.trainer.LightGBMRankerTrainer:{VERSION}
```
    
4. Conventional local usage with `uses` argument
```bash
jina pod --uses config.yml --port-in 55555 --port-out 55556
```