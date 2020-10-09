__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders import BaseNumericEncoder
from jina.executors.encoders.frameworks import BaseTFEncoder


class CompressionVaeEncoder(BaseNumericEncoder, BaseTFEncoder):
    """
    :class:`CompressionVaeEncoder` is a dimensionality reduction tool based on the idea of
    Variational Autoencoders. It encodes data from an ndarray in size `B x T` into an ndarray in size `B x D`.

    Full code and documentation can be found here: https://github.com/maxfrenzel/CompressionVAE..
    """

    def __init__(self, model_path='temp',
                 *args, **kwargs):
        """
        :param model_path: specifies the path to the pretrained model

        """
        super().__init__(*args, **kwargs)
        self.model_path = model_path

    def post_init(self):
        from cvae import cvae
        import tensorflow as tf
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=2)
            self.model = cvae.load(saver, sess, self.model_path)
        self.to_device()

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        :param data: a `B x T` numpy ``ndarray``, `B` is the size of the batch
        :return: a `B x D` numpy ``ndarray``
        """
        return self.model.embed(data)
