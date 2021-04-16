__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina.executors.encoders.numeric import TransformEncoder


class FastICAEncoder(TransformEncoder):
    """
    Encodes data using a fast algorithm for Independent Component Analysis (FastICA)

    Encodes data from an ndarray in size `B x T` into an ndarray in size `B x D`.
    Where `B` is the batch's size and `T` and `D` are the dimensions pre (`T`)
    and after (`D`) the encoding.
    """