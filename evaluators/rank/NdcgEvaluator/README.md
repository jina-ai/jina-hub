# NDCGEvaluator

`NDCGEvaluator` computes the [NDCG (Normalized Discounted Cumulative Gain)](https://en.wikipedia.org/wiki/Discounted_cumulative_gain). 
It measures the performance of a retrieval system based on the graded relevance of the predicted scores and desired scores. 
It ranges from `0.0` to `1.0`, with `1.0` representing the ideal ranking of the retrieved result.

`NDCGEvaluator` takes 3 parameters:
- eval_at: The number of documents in each of the lists to consider in the NDCG computation. If None. the complete lists are considered
        for the evaluation computation
- power_relevance: The power relevance places stronger emphasis on retrieving relevant documents.
        For detailed information, please check https://en.wikipedia.org/wiki/Discounted_cumulative_gain
- is_relevance_value: Indicating if the scores coming from the `Search System` results are to be considered relevance, meaning highest value is better.
     Since the `input` of the `evaluate` method is sorted according to the `scores` of both actual and desired input, this parameter is
        useful for instance when the `matches` come directly from a `VectorIndexer` where score is `distance` and therefore the `smaller` the `better`.

## Usage example.

A simple example on how one would use it in Python.

```python
expected = [('1', 5.0), ('2', 3.0), ('3', 4.0)]
results = [('2', 1.0), ('1', 0.8), ('3', 0.7)]
evaluator = NDCGEvaluator(eval_at=5, power_relevance=False)
evaluation = evaluator.evaluate(actual=results, desired=expected)
```

Example of a YAML configuration:

```yaml
!NDCGEvaluator
with:
  eval_at: 10
  power_relevance: False
metas:
  py_modules:
    - __init__.py
requests:
    on:
      SearchRequest:
        - !RankEvaluateDriver
          with:
            fields: ['tags__id', 'score__value']
            traversal_paths: ['r']
```

And this is how Documents and GroundTruths can be provided to the Flow for being evaluated at ranking time.

```python
from jina.flow import Flow
index_docs = [Document({'text': 'some text to encode', 'tags': {'id': 1}}), 
            Document({'text': 'some different text to encode', 'tags': {'id': 2}})]

# index time
f = Flow().add(name='encoder', uses='encoder.yml').\
        add(name='indexer', uses='indexer.yml')
with f:
    f.index(index_docs)

evaluate_doc = Document({'text': 'query text'})

evaluate_groundtruth = Document()
expected_match_1 = Document({'tags': {'id': 1}, {'score': {'value': 5.0}}})
expected_match_2 = Document({'tags': {'id': 2}, {'score': {'value': 4.0}}})
evaluate_groundtruth.matches.append(expected_match_1)
evaluate_groundtruth.matches.append(expected_match_2)

def print_evaluation_score(resp):
    for doc in resp.search.docs:
       print(f' Evaluation {doc.evaluations[0].op_name}: {doc.evaluations[0].value}')

#evaluate time
f = Flow().add(name='encoder', uses='encoder.yml').\
        add(name='indexer', uses='indexer.yml').\
        add(name='ndcgevaluator', uses='NDCGEvaluator.yml')
with f:
    f.search(input_fn=[(evaluate_doc, evaluate_groundtruth)], on_done=print_evaluation_score)
```
