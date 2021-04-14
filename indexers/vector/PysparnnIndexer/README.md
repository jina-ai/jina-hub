# PysparnnIndexer


[PySparNN](https://github.com/facebookresearch/pysparnn) is a library for fast similarity search of Sparse Scipy vectors. In contains an algorithm that can be used to perform fast approximate search with sparse inputs. Developed by Facebook AI Research.

### Snippets:

Initialize `PysparnnIndexer`:

```
indexer = PysparnnIndexer(k_clusters: int = 2,
                 metric: str = 'cosine',
                 num_indexes: int = 2,
                 prefix_filename: str = 'pysparnn_index')
```
