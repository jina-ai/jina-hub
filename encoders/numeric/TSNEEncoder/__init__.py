__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np

from jina.executors.decorators import batching
from jina.executors.encoders import BaseNumericEncoder


class TSNEEncoder(BaseNumericEncoder):
    """
    Encode ``Document`` content using t-distributed Stochastic Neighbor Embedding.

    Encodes ``Document`` content from an ndarray in size `B x T` into an ndarray in size `B x D`
    Where `B` is the batch's size and `T` and `D` are the dimensions pre (`T`)
    and after (`D`) the encoding.

    See more details
    `here <https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html>`_

    :param output_dim: Dimension of the embedded space
    :param random_state: Used to seed the cost_function of TSNE
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments

    .. note:
        Unlike other numeric encoders, TSNE does not inherit Transform encoder
        because it can't have a transform without fit.
    """

    def __init__(self, output_dim: int = 64,
                 random_state: int = 2020,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dim = output_dim
        self.random_state = random_state

    def post_init(self):
        """Load TSNE model"""
        super().post_init()
        from sklearn.manifold import TSNE
        self.model = TSNE(n_components=self.output_dim, random_state=self.random_state)

    @batching
    def encode(self, content: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        Encode ``Document`` content from an ndarray in size `B x T` into an ndarray in size `B x D`

        :param content: a `B x T` numpy ``ndarray``, `B` is the size of the batch
        :return: a `B x D` numpy ``ndarray``
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        """
        return self.model.fit_transform(content)
