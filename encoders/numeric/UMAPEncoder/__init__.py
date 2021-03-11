import numpy as np

from jina.executors.decorators import batching
from jina.executors.encoders import BaseNumericEncoder


class UMAPEncoder(BaseNumericEncoder):
    """
    :class:`UMAPEncoder` data using Uniform Manifold Approximation and Projection Embedding.

    Encodes data from an ndarray of size `B x T` into an ndarray of size `B x D`
    Where `B` is the batch's size and `T` and `D` are the dimensions pre (`T`)
    and after (`D`) the encoding.

    Full code and documentation can be found
    `here <https://github.com/lmcinnes/umap>`_.

    :param output_dim: Dimension of the embedded space
    :param random_state: Used to seed the random number generator
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """
    def __init__(self, output_dim: int = 64,
                 random_state: int = 2020,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dim = output_dim
        self.random_state = random_state

    def post_init(self):
        """Load UMAP model"""
        super().post_init()
        from umap import UMAP
        self.model = UMAP(n_components=self.output_dim, random_state=self.random_state)

    @batching
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        Encode data from an ndarray of size `B x T` into an ndarray 
        of size `B x D`

        :param data: a `B x T` numpy ndarray
        :return: a `B x D` numpy ndarray
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        """
        return self.model.fit_transform(data)