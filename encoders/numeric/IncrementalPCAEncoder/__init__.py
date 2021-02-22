__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina.executors.encoders.numeric import TransformEncoder

class IncrementalPCAEncoder(TransformEncoder):
    """
    Encodes data from an ndarray in size `B x T` into an ndarray in size `B x D`.

    Where `B` is the batch's size and `T` and `D` are the dimensions pre (`T`)
    and after (`D`) the encoding.

    :param output_dim: the output size.
    :param num_features: the number of input features.
        If ``num_features`` is None, then ``num_features``
        is inferred from the data
    :param whiten: If whiten is false, the data is already considered to be whitened,
        and no whitening is performed.
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments

    More details can be found
    `here <https://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html>`_

    .. note::
        :class:`IncrementalPCAEncoder` must be trained before calling ``encode()``.
        This encoder can be trained in an incremental way.
    """

    def __init__(self, num_features: int = None, whiten: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.whiten = whiten
        self.num_features = num_features
        self.is_trained = False
        self.model = None

    def post_init(self):
        """Load IncrementalPCA model"""
        super().post_init()
        if not self.model:
            from sklearn.decomposition import IncrementalPCA
            self.model = IncrementalPCA(
                n_components=self.output_dim,
                whiten=self.whiten)