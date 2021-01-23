__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Optional

import numpy as np
import json

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders import BaseNumericEncoder
from jina.executors.devices import TFDevice
from jina.excepts import PretrainedModelFileDoesNotExist


class CompressionVaeEncoder(TFDevice, BaseNumericEncoder):
    """
    :class:`CompressionVaeEncoder` is a dimensionality reduction tool based on the idea of
    Variational Autoencoders. It encodes data from an ndarray in size `B x T` into an ndarray in size `B x D`.

    Full code and documentation can be found here: https://github.com/maxfrenzel/CompressionVAE..
    """

    def __init__(self, model_path: Optional[str] = 'model',
                 *args, **kwargs):
        """
        :param model_path: specifies the path to the pretrained model

        """
        super().__init__(*args, **kwargs)
        self.model_path = model_path

    def post_init(self):
        super().post_init()
        from cvae import cvae
        import cvae.lib.model_iaf as model
        import tensorflow as tf

        params_path = os.path.join(self.model_path, 'params.json') if self.model_path and os.path.exists(
            self.model_path) else None

        if params_path and os.path.exists(params_path):

            config = tf.ConfigProto(log_device_placement=False)
            config.gpu_options.allow_growth = True

            with tf.Graph().as_default():

                # Load parameter file.
                with open(params_path, 'r') as f:
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
                saver = tf.train.Saver(var_list=tf.trainable_variables())
                cvae.load(saver, self.sess, self.model_path)

                self.to_device()
        else:
            raise PretrainedModelFileDoesNotExist()

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        :param data: a `B x T` numpy ``ndarray``, `B` is the size of the batch
        :return: a `B x D` numpy ``ndarray``
        """
        return self.sess.run([self.embeddings],
                             feed_dict={self.data_feature_placeholder: data})[0]

    def close(self) -> None:
        super().close()
        self.sess.close()
