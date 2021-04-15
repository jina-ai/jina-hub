# PysparnnIndexer


[PySparNN](https://github.com/facebookresearch/pysparnn) is a library for fast similarity search of Sparse Scipy vectors. In contains an algorithm that can be used to perform fast approximate search with sparse inputs. Developed by Facebook AI Research.

### How to

#### Dependencies

`PysparnnIndexer` is dependent on `scipy` and `PySparNN`:

```
pip install scipy
pip install git+https://github.com/facebookresearch/pysparnn.git
```

#### Usage

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
