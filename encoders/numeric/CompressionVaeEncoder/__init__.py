__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np
import json

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
        import cvae.lib.model_iaf as model
        import tensorflow as tf

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True

        with tf.Graph().as_default():
            # Load parameter file.
            with open(f'{self.model_path}/params.json', 'r') as f:
                param = json.load(f)

            net = model.VAEModel(param,
                                 None,
                                 input_dim=param['dim_feature'],
                                 keep_prob=tf.placeholder_with_default(input=tf.cast(1.0, dtype=tf.float32),
                                                                       shape=(),
                                                                       name="KeepProb"),
                                 initializer='orthogonal')
            # Placeholder for data features
            self.data_feature_placeholder = tf.placeholder_with_default(
                input=tf.zeros([64, param['dim_feature']], dtype=tf.float32),
                shape=[None, param['dim_feature']])

            self.embeddings = net.embed(self.data_feature_placeholder)

            self.sess = tf.Session(config=config)
            init = tf.global_variables_initializer()
            self.sess.run(init)

            # Saver for loading checkpoints of the model.
            saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=2)
            cvae.load(saver, self.sess, self.model_path)

            self.to_device()

    @batching(batch_size=64)
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        :param data: a `B x T` numpy ``ndarray``, `B` is the size of the batch
        :return: a `B x D` numpy ``ndarray``
        """
        import tensorflow as tf

        return self.sess.run([self.embeddings],
                             feed_dict={self.data_feature_placeholder: data})[0]
