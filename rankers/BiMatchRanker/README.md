# BiMatchRanker

`BiMatchRanker` tries to assign a score for a `match` given a `query` based on the matches found for the `query`
    chunks on the `match` chunks.

The situation can be seen with an hypothetical example:

    `Documents in the Index`:
        - Granularity 0 - Doc 1 - `BiMatchRanker is a Hub Executor. computes relevance scores. Found in Jina, the neural search framework`
                        - Granularity 1 - Doc 11 - `BiMatchRanker is a Ranking Jina Hub Executor`
                                        - Doc 12 - `It is a powerfull tool.`
                                        - Doc 13 - `Found in Jina`
                                        - Doc 14 - `the neural search framework`
        - Granularity 0 - Doc 2 - `A Ranker executor gives relevance Scores. Rankers are found in the Hub, and built automatically`
                        - Granularity 1 - Doc 21 - `A Ranker executor gives relevance Scores`
                        - Granularity 1 - Doc 22 - `Rankers are found in the Hub`
                        - Granularity 1 - Doc 23 - `and built automatically`

    `Query`
        - Granularity 0 - 'A Hub Executor. Ranking on Relevance scores. Giving automatically relevant documents from the Hub'
                        - Granularity 1 - `A Hub Executor`
                                        - `Ranking on Relevance scores`
                                        - `Giving automatically relevant documents from the Hub`

    `Semantic similarity matches` (TOP_K 3)
        Query chunk 1 -> `A Hub Executor` -> Doc11, Doc22, Doc21 (Children of `Doc1` and `Doc2`)
        Query chunk 2 -> `Ranking on Relevance scores` -> Doc21, Doc11, Doc22 (Children of `Doc2` and `Doc1`)
        Query chunk 3 -> `Scoring relevant documents`  -> Doc21, Doc22, Doc23 (Children of `Doc2`)

In order to compute this relevance score, it tries to take into account two perspectives.

- Match perspective: How many chunks of a match are part of the set of matches (hit) of a query (and its chunks).
  In the example:
  - `Doc1` has 4 chunks, but only 1 (hit) (Doc11) is found in the matches of the query chunks.
  - `Doc2` has 3 chunks and 2 of them (hit) are found in the matches (Doc21, Doc22).

- Query perspective: Of all the chunks in the query, how many of them are matches by any of the chunks of each match document.
  In the example:
  - `Query` for `Doc1`: Query has 3 chunks, but only 2 of them are matched by chunks of match `Doc1` (Doc11)
  - `Query` for `Doc2`: Query has 3 chunks, and the 3 of them are matched by chunks of match `Doc2`

So for every `Document`, a score is computed adding some penalty to the `missing` in both perspectives.

The **BiMatchRanker** executor needs only one parameter:

| `param_name`  | `param_remarks` |
| ------------- | ------------- |
| `d_miss`  |Controls how much penalty to give to non-hit chunks|

## Usage

Users can use Pod images in several ways:

1. Run with Docker (`docker run`)
   ```bash
    docker run jinahub/pod.ranker.bimatchranker:0.0.10-1.0.1 --port-in 55555 --port-out 55556
    ```
    
2. Run with Flow API
   ```python
    from jina.flow import Flow

    f = (Flow()
        .add(name='my_encoder', uses='docker://jinahub/pod.ranker.bimatchranker:0.0.10-1.0.1'))
    ```
    
3. Run with Jina CLI
   ```bash
    jina pod --uses docker://jinahub/pod.ranker.bimatchranker:0.0.10-1.0.1
    ```
    
4. Conventional local usage with `uses` argument
    ```bash
    jina pod --uses hub/ramler/bimatchranker.yml
    ```
