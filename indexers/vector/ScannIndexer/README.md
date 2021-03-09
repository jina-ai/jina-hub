# ScannIndexer

ScaNN (Scalable Nearest Neighbors) is a method for efficient vector similarity search at scale. This code release implements [1], which includes search space pruning and quantization for Maximum Inner Product Search and also supports other distance functions such as Euclidean distance. The implementation is designed for x86 processors with AVX2 support. ScaNN achieves state-of-the-art performance on ann-benchmarks.com on the glove-100-angular dataset.
ScannIndexer is hence a Scann powered vector indexer.

## Usage

For Ubuntu 16.04 or later and Python 3.7.

If you are not using linux it is necessary to build Scann from source  [follow instructions from here](https://github.com/google-research/google-research/tree/master/scann). 

## Snippets:

Initialise ScannIndexer:

`ScannIndexer(num_leaves, training_iterations, distance_measure, num_leaves_to_search, training_sample_size, scoring, anisotropic_quantization_threshold, dimensions_per_block, reordering_num_neighbors)`
| File                                 | Descriptions                                                                                        |
|--------------------------------------|-----------------------------------------------------------------------------------------------------|
| `num_leaves`                         | Roughly the square root of the number of datapoints, directly linked to partition quality           |                                      |
| `training_iterations`                | Number of iterations per training                                                                   |
| `distance_measure`                   | Distance measurement used between the query and the points                                          | 
| `num_leaves_to_search`               | Number of leaves to search tuned as per recall target                                               |
| `training_sample_size`               | The size of the training sample                                                                     |
| `scoring`                            | It can be score_ah (asymmetric hashing) or score_bf (brute force)                                   |
| `anisotropic_quantization_threshold` | Refer https://arxiv.org/abs/1908.10396                                                              |
| `dimensions_per_block`               | Recommended for AH is 2                                                                             |
| `reordering_num_neighbors`           | Should be higher than the final number of neighbors. Increases accuracy but impacts speed           |
       


**NOTE**: 

- `MODULE_VERSION` is the version of the ScannIndexer, in semver format. E.g. `0.0.11`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.0` 

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
      uses: indexers/vector/ScannIndexer/config.yml
  ```
  
  and then in `scann.yml`:
  ```yaml
  !ScannIndexer
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
  jina pod --uses indexers/vector/ScannIndexer/config.yml --port-in 55555 --port-out 55556
  ```
- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.
  ```bash
    docker run --network host docker://jinahub/pod.indexer.scannindexer:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
    ```

## Possible bugs

### Typo on config.py

If you are running under MacOs probably you will need to do the following changes.

In `config.py` modify `generate_shared_lib_name` as follows :

```
def generate_shared_lib_name(namespec):
"""Converts the linkflag namespec to the full shared library name."""
# Assume Linux for now
# return namespec[1][3:]
#They have a typo with this, this is a dirty fix,hardcoded the name of the lib
return "libtensorflow_framework.2.dylib"
```
They have a typo when running under MacOs,  [you can find more info here](https://github.com/google-research/google-research/issues/342). 

### Compatibiliy with hash_set

You need to change the following files:

`memory_logging.h`
`partitioner_base.h`
`dataset.cc`
`kmeans_tree_node.cc`

in ech file change the  `#include <hash_set>`
for 

```
#if defined __GNUC__ || defined __APPLE__
#include <ext/hash_set>
#else
#include <hash_set>
#endif
```


