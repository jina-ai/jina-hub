# AveragePrecisionEvaluator

`AveragePrecisionEvaluator` computes the [Average Precision](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision) for the ranking result of a search system.

`AveragePrecisionEvaluator` computes the area under the `Precision-Recall` curve. The finite sum is equivalent to
the sum of `Precision` values at all possible `k` for the `k` where a `Relevant` document is found, divided by the amount of `Relevant` documents.

`AveragePrecisionEvaluator` receives two lists of `Document` identifiers:
- The first one is considered to be the actual result of a search system to evaluate
- The second one is the groundtruth, and is considered to be a list containing all the `Relevant` Documents. Thus, the relevance of
the documents in the first list is considered by its presence or not in the `groundtruth` set. 
 
 ## Usage example.

A simple example on how one would use it in Python.

```python
expected = [1, 3, 6, 9, 10]
results = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evaluator = AveragePrecisionEvaluator()
evaluation = evaluator.evaluate(actual=results, desired=expected)
```

Example of a YAML configuration:

```yaml
!AveragePrecisionEvaluator
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
        add(name='averagprecisionevaluator', uses='!AveragePrecisionEvaluator')
with f:
    f.search(input_fn=[(evaluate_doc, evaluate_groundtruth)], on_done=print_evaluation_score)
```