# BleuEvaluator

Bilingual Evaluation Understudy Score. Evaluates the generated sentence against a desired sentence. 
A perfect match will score 1.0 and a complete mismatch will score 0.0

BLEU works well with n-grams of at least 4, if less than that, it's necessary to use a smoothing function and reset the weights.
We use accumulative-error to check the n-grams and set the weights accordingly.

Read more about BleuEvaluators in the links below:
  - https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
  - https://towardsdatascience.com/bleu-bilingual-evaluation-understudy-2b4eab9bcfd1
  - https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213

## Usage:

Initialise the Executor and use `evaluate` method specifying arguments i.e.:

| `arg_name`  | `arg_remarks` |
| ------------- | ------------- |
| `actual`  | The text predicted by the search system  |
| `desired`  | The expected text given by user as groundtruth |

The `score` returned is the evaluation metric value for the request document.

### Snippets:

Users can use Pod images in several ways:

- Run with Docker (`docker run`)
  - ```bash
    docker run jinahub/pod.evaluator.bleuevaluator:0.0.4-1.0.7 --port-in 55555 --port-out 55556
    ```

- Flow API
  - ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my_evaluator', uses='jinahub/pod.evaluator.bleuevaluator:0.0.4-1.0.7', port_in=55555, port_out=55556)
    ```

- Jina CLI
  - ```bash
    jina pod --uses docker://jinahub/pod.evaluator.bleuevaluator:0.0.4-1.0.7 --port-in 55555 --port-out 55556
    ```

- Conventional local usage with `uses` argument
  - ```bash
    jina pod --uses hub/example/config.yml --port-in 55555 --port-out 55556
    ```

- Docker command

  - Specify the image name along with the version tag. The snippet below uses Jina version `1.0.7`

  - ```bash
    docker pull jinahub/pod.evaluator.bleuevaluator:0.0.4-1.0.7
    ```