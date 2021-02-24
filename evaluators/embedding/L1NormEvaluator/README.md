# L1NormEvaluator

`L1NormEvaluator` evaluates the distance between actual and desired embeddings computing the L1 Norm between them.

## Usage:

Initialise this Executor specifying parameters i.e.:

| `param_name`  | `param_remarks` |
| ------------- | ------------- |
| `actual`  | the embedding of the document  |
| `desired`  | the expected embedding of the document |

The model is pre-trained on [MobileNetV2] data
The pretrained default path is the result of downloading the models in `download.sh`

### Snippets:

Initialise L1NormEvaluator:

`L1NormEvaluator(model_path='pretrained', channel_axis=1, metas=metas, model_name=MobileNetV2)`

Users can use Pod images in several ways:

- Run with Docker (`docker run`)
  - ```bash
    docker run jinahub/pod.evaluator.l1normevaluator:0.0.8-0.9.28 --port-in 55555 --port-out 55556
    ```

- Flow API
  - ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-encoder', image='jinahub/pod.evaluator.l1normevaluator:0.0.8-0.9.28', port_in=55555, port_out=55556)
    ```

- Jina CLI
  - ```bash
    jina pod --uses jinahub/pod.evaluator.l1normevaluator:0.0.8-0.9.28 --port-in 55555 --port-out 55556
    ```

- Conventional local usage with `uses` argument
  - ```bash
    jina pod --uses hub/example/config.yml --port-in 55555 --port-out 55556
    ```

- Docker command

  - Specify the image name along with the version tag. The snippet below uses Jina version `0.9.20`

  - ```bash
    docker pull jinahub/pod.evaluator.l1normevaluator:0.0.8-0.9.28
    ```