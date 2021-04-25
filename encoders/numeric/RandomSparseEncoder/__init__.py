__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina.executors.encoders.numeric import TransformEncoder


class RandomSparseEncoder(TransformEncoder):
    """
    Reduce dimensionality using sparse random projection.

    Encodes ``Document`` content from an ndarray in size `B x T` into an ndarray in size `B x D`
    Where `B` is the batch's size and `T` and `D` are the dimensions pre (`T`)
    and after (`D`) the encoding.

    More info can be found
    `here <https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.SparseRandomProjection.html>`_
    """
