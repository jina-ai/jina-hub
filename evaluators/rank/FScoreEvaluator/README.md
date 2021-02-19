# FScoreEvaluator

`FScoreEvaluator` computes the [F-Score](https://en.wikipedia.org/wiki/F-score) for the ranking result of a search system.

`FScoreEvaluator` gives an evaluation measure which considers both `Precision` and `Recall` evaluation metrics.

`FScoreEvaluator` takes 2 parameters:
- eval_at: The number of documents in each of the lists to consider in the `Precision` and `Recall` computation
- beta: Parameter to weight differently precision and recall. When beta is 1, the fScore corresponds to the harmonic mean
        of precision and recall
 
 ## Usage example.

A simple example on how one would use it in Python.

```python
results = ['0', '1', '2', '3', '4']
expected = ['1', '0', '20', '30', '40']
evaluator = FScoreEvaluator(eval_at=None, beta=1)
evaluation = evaluator.evaluate(actual=results, desired=expected)
assert evaluation == 0.4
```

Example of a YAML configuration:

```yaml
!FScoreEvaluator
with:
  {}
metas:
  py_modules:
    - __init__.py
requests:
    on:
      SearchRequest:
        - !RankEvaluateDriver
          with:
            fields: ['tags__id']
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
expected_match_1 = Document({'tags': {'id': 3}})
expected_match_2 = Document({'tags': {'id': 2}})
evaluate_groundtruth.matches.append(expected_match_1)
evaluate_groundtruth.matches.append(expected_match_2)

def print_evaluation_score(resp):
    for doc in resp.search.docs:
       print(f' Evaluation {doc.evaluations[0].op_name}: {doc.evaluations[0].value}')

#evaluate time
f = Flow().add(name='encoder', uses='encoder.yml').\
        add(name='indexer', uses='indexer.yml').\
        add(name='evaluator', uses='!FScoreEvaluator')
with f:
    f.search(input_fn=[(evaluate_doc, evaluate_groundtruth)], on_done=print_evaluation_score)
```
