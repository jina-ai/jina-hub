# LevenshteinRanker

Computes the Levenshtein distance between a query and its matches.
The distance is negative, in order to achieve a better result, sorted in the respective driver.

## Usage:

Initialise the Executor and use `score` method specifying arguments i.e.:

| `arg_name`  | `arg_remarks` |
| ------------- | ------------- |
| `query_meta`  | A Dict of queries to score  |
| `old_match_scores`  | Previously scored matches |
| `match_meta` | A Dict of matches of given query |

The `distance` is returned as `col_score` by the `score` method

### Snippets:

Initialise LevenshteinRanker:

`LevenshteinRanker(model_path='pretrained', channel_axis=1, metas=metas, model_name=MobileNetV2)`

Users can use Pod images in several ways:

- Run with Docker (`docker run`)
  - ```bash
    docker run jinahub/pod.ranker.levenshteinranker:0.0.8-1.0.7 --port-in 55555 --port-out 55556
    ```

- Flow API
  - ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my_ranker', uses='jinahub/pod.ranker.levenshteinranker:0.0.8-1.0.7', port_in=55555, port_out=55556)
    ```

- Jina CLI
  - ```bash
    jina pod --uses docker://jinahub/pod.ranker.levenshteinranker:0.0.8-1.0.7 --port-in 55555 --port-out 55556
    ```

- Conventional local usage with `uses` argument
  - ```bash
    jina pod --uses hub/example/config.yml --port-in 55555 --port-out 55556
    ```

- Docker command

  - Specify the image name along with the version tag. The snippet below uses Jina version `1.0.7`

  - ```bash
    docker pull jinahub/pod.ranker.levenshteinranker:0.0.8-1.0.7
    ```