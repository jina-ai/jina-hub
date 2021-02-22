__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"
from jina.executors.encoders.numeric import TransformEncoder


class RandomGaussianEncoder(TransformEncoder):
    """
    Reduce dimensionality using Gaussian random projection.

    Encodes data from an ndarray in size `B x T` into an ndarray in size `B x D`
    Where `B` is the batch's size and `T` and `D` are the dimensions pre (`T`)
    and after (`D`) the encoding.

    More info can be found
    `here <https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html>`_
    """

    def post_init(self):
        """Load GaussianRandomProjection model"""
        super().post_init()
        if not self.model:
            from sklearn.random_projection import GaussianRandomProjection
            self.model = GaussianRandomProjection(n_components=self.output_dim, random_state=self.random_state)
