# SptagIndexer

SPTAG (Space Partition Tree And Graph) is a library for large scale vector approximate nearest neighbor search scenario released by Microsoft Research (MSR) and Microsoft Bing.
SPTAG  assumes that the samples are represented as vectors and that the vectors can be compared by L2 distances or cosine distances. Vectors returned for a query vector are the vectors that have smallest L2 distance or cosine distances with the query vector.
SPTAG provides two methods: kd-tree and relative neighborhood graph (SPTAG-KDT) and balanced k-means tree and relative neighborhood graph (SPTAG-BKT). SPTAG-KDT is advantageous in index building cost, and SPTAG-BKT is advantageous in search accuracy in very high-dimensional data.
SptagIndexer is an SPTAG powered vector indexer.
For more information, refer [SPTAG's GitHub repo](https://github.com/microsoft/SPTAG).

## Snippets:

Initialise SptagIndexer:

`SptagIndexer(method, samples, tpt_number, tpt_leaf_size, neighborhood_size, graph_neighborhood_size, cef, max_check_for_refined_graph, num_threads, max_ckeck, dist_calc_method, bkt_number, bkt_meansk, kdt_number)`
| File                            | Descriptions                                                                   |
|---------------------------------|--------------------------------------------------------------------------------|
| `method`                        | The index method to use, index Algorithm type (e.g. BKT, KDT), required        |                                      |
| `samples`                       | how many points will be sampled to do tree node split                          |                                          |
| `tpt_number`                    | number of TPT trees to help with graph construction                            | 
| `tpt_leaf_size`                 | TPT tree leaf size                                                             |
| `neighborhood_size`             | number of neighbors each node has in the neighborhood graph                    |                                                |
| `graph_neighborhood_size`       | number of neighborhood size scale in the build stage                           |
| `cef                          ` | number of results used to construct RNG                                        |
| `max_check_for_refined_graph`   | how many nodes each node will visit during graph refine in the build stage     |   
| `num_threads`                   | The number of threads to use                                                   |
| `max_check`                     | how many nodes will be visited for a query in the search stage                 |
| `dist_calc_method`              | the distance type, currently SPTAG only support Cosine and L2 distances        |
| `bkt_number`                    | number of BKT trees (only used if method is BKT)                               |
| `bkt_meansk`                    | how many childs each tree node has (only used if method is BKT)                |
| `kdt_number`                    | number of KDT trees (only used if method is BKT)                               |


**NOTE**: 

- `MODULE_VERSION` is the version of the SptagIndexer, in semver format. E.g. `0.0.13`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.6` 

- Flow API

  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-indexer', uses='docker://jinahub/pod.indexer.scannindexer:MODULE_VERSION-JINA_VERSION')
    ```
- Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: ngt
      uses: indexers/vector/SptagIndexer/config.yml
  ```
  
  and then in `sptag.yml`:
  ```yaml
  !SptagIndexer
  with:
    hostname: yourdomain.com
    port: 6379
    db: 0
  ```
- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.indexer.scannindexer:MODULE_VERSION-JINA_VERSION
  ```
- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses indexers/vector/SptagIndexer/config.yml --port-in 55555 --port-out 55556
  ```
- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.
  ```bash
    docker run --network host docker://jinahub/pod.indexer.scannindexer:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```