# PysparnnIndexer


[PySparNN](https://github.com/facebookresearch/pysparnn) is a library for fast similarity search of Sparse Scipy vectors. In contains an algorithm that can be used to perform fast approximate search with sparse inputs. Developed by Facebook AI Research.

## How to

### Dependencies

`PysparnnIndexer` is dependent on `scipy` and `PySparNN`:

```
pip install scipy
pip install git+https://github.com/facebookresearch/pysparnn.git
```

### Use as a Python class

Initialize `PysparnnIndexer`:

```python
indexer = PysparnnIndexer(
    k_clusters = 2,
    metric = 'cosine',
    num_indexes = 2,
    prefix_filename = 'pysparnn_index'
)
```

Note, if it is **not** the first time use the indexer in your project,
``PysparnnIndexer`` will load the index in your workspace automatically.

Query, Add, Update, Delete and Save interface:

```python
import numpy as np
from scipy.sparse import csr_matrix

indexer = PysparnnIndexer(
    k_clusters = 2,
    metric = 'cosine',
)
keys = list(range(0, 50))
vectors1 = csr_matrix(np.random.binomial(1, 0.01, size=(50, 100)))
vectors2 = csr_matrix(np.random.binomial(1, 0.01, size=(50, 100)))

indexer.add(keys=keys, vectors=vectors1)
indexer.update(keys=keys, vectors=vectors2)
indexer.query(vectors=vectors2[:5], top_k=1)
indexer.delete(keys=keys)
# Close indexer will trigger the save method.
# and dumpy index to workspace.
indexer.close()
```

### Use In Flow API

- `MODULE_VERSION` is the version of the PysparnnIndexer, in semver format. E.g. `0.0.16`.
- `JINA_VERSION` is the version of the Jina core version with which the Docker image was built. E.g. `1.0.7` 

- Flow API

  ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my-indexer', uses='docker://jinahub/pod.indexer.pysparnnindexer:MODULE_VERSION-JINA_VERSION')
  ```
- Flow YAML file
  This is the only way to provide arguments to its parameters:
  
  ```yaml
  pods:
    - name: faiss
      uses: indexers/vector/PysparnnIndexer/config.yml
  ```
  
  and then in `pysparnn.yml`:
  ```yaml
  !PysparnnIndexer
  ```
- Jina CLI
  
  ```bash
  jina pod --uses docker://jinahub/pod.indexer.pysparnnindexer:MODULE_VERSION-JINA_VERSION
  ```
- Conventional local usage with `uses` argument
  
  ```bash
  jina pod --uses indexers/vector/PysparnnIndexer/config.yml --port-in 55555 --port-out 55556
  ```
- Run with Docker (`docker run`)
 
  Specify the image name along with the version tag. The snippet below uses Jina version as `JINA_VERSION`.
  ```bash
    docker run --network host docker://jinahub/pod.indexer.pysparnnindexer:MODULE_VERSION-JINA_VERSION --port-in 55555 --port-out 55556
  ```
