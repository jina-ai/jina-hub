__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina.executors.encoders.numeric import TransformEncoder


class TruncatedSVDEncoder(TransformEncoder):
    """
    Encodes data using truncated SVD, and does not center the data before
    computing SVD which makes it efficient when working with sparse matrices.

    Encodes data from a ndarray of size `B x T` into a ndarray of size `B x D`.
    Where `B` is the batch's size and `T` and `D` are the dimensions pre (`T`)
    and after (`D`) the encoding.

    :param output_dim: Dimension of the output embedded space
    :param n_iter: Number of iterations for the encoder to run
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments

    More details can be found
    `here <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html>`_

    .. note::
        :class:`TruncatedSVDEncoder` must be trained before calling ``encode()``.
    """

    def __init__(
        self,
        output_dim: int = None,
        algorithm: str = "randomized",
        max_iter: int = 200,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.output_dim = output_dim
        self.max_iter = max_iter
        self.algorithm = algorithm
        self.is_trained = False

    def post_init(self):
        """Load TruncatedSVD model"""
        super().post_init()

        from sklearn.decomposition import TruncatedSVD

        self.model = TruncatedSVD(
            n_components=self.output_dim, algorithm=self.algorithm, n_iter=self.max_iter
        )
