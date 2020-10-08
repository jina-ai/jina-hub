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

    def __init__(self, model_path='temp', X=None, train_valid_split=0.99, output_dim=16,
                 iaf_flow_length=10, cells_encoder=[512, 256, 128],
                 initializer='lecun_normal', batch_size=32, batch_size_test=32,
                 feature_normalization=False, tb_logging=False,
                 *args, **kwargs):
        """
        :param model_path: specifies the path to the model, and also acts as the model name.
        :param X : array, shape (n_samples, n_features)
            Training data for the VAE.
            Alternatively, X can be the path to a root-directory containing npy files (potentially nested), each
            representing a single feature vector. This allows for handling of datasets that are too large to fit
            in memory.
            Can be None (default) only if a model with this name has previously been trained. Otherwise None will
            raise an exception.
        :param train_valid_split: controls the random splitting into train and test data. Here 99% of X is used
        for training, and only 1% is reserved for validation.
        :param dim_latent: specifies the dimensionality of the latent space.
        :param iaf_flow_length: controls how many IAF(Inverse Autoregressive Flow) flow layers the model has.
        :param cells_encoder: determines the number, as well as size of the encoders fully connected layers.
        :param initializer: controls how the model weights are initialized, with options being `orthogonal` (default),
        `truncated_normal`, and `lecun_normal`
        :param batch_size: determine the batch sizes used for training.
        :param batch_size_test: determine the batch sizes used for testing.
        :param feature_normalization: tells CVAE whether it should internally apply feature normalization
        (zero mean, unit variance, based on the training data) or not. If True, the normalisation factors are stored
        with the model and get applied to any future data.
        :param tb_logging: determines whether the model writes summaries for TensorBoard during the training process.
        """
        super().__init__(*args, **kwargs)
        self.X = X
        self.train_valid_split = train_valid_split
        self.output_dim = output_dim
        self.iaf_flow_length = iaf_flow_length
        self.cells_encoder = cells_encoder
        self.initializer = initializer
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.model_path = model_path
        self.feature_normalization = feature_normalization
        self.tb_logging = tb_logging

    def post_init(self):
        super().post_init()
        self.to_device()

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        :param data: a `B x T` numpy ``ndarray``, `B` is the size of the batch
        :return: a `B x D` numpy ``ndarray``
        """
        from cvae import cvae
        model = cvae.CompressionVAE(X=self.X,
                                    train_valid_split=self.train_valid_split,
                                    dim_latent=self.output_dim,
                                    iaf_flow_length=self.iaf_flow_length,
                                    cells_encoder=self.cells_encoder,
                                    initializer=self.initializer,
                                    batch_size=self.batch_size,
                                    batch_size_test=self.batch_size_test,
                                    logdir=self.model_path,
                                    feature_normalization=self.feature_normalization,
                                    tb_logging=self.tb_logging)
        return model.embed(data)
