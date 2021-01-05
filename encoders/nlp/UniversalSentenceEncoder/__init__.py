__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional

import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseTFEncoder


class UniversalSentenceEncoder(BaseTFEncoder):
    """
    :class:`UniversalSentenceEncoder` is a encoder based on the Universal Sentence
    Encoder family (https://tfhub.dev/google/collections/universal-sentence-encoder/1).
    It encodes data from an 1d array of string in size `B` into an ndarray in size `B x D`.
    """

    def __init__(
            self,
            model_url: str = 'https://tfhub.dev/google/universal-sentence-encoder/4',
            preprocessor_url: Optional[str] = None,
            * args,
            **kwargs):
        """
        :param model_url: the url of the model (TensorFlow Hub). For supported models see
                          family overview: https://tfhub.dev/google/collections/universal-sentence-encoder/1)
        :param preprocessor_url: the url of preprocessors (TensorFlow Hub).
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.model_url = model_url
        self.preprocessor_url = preprocessor_url

    def post_init(self):
        self.to_device()
        import tensorflow_hub as hub
        self.preprocessor = None
        if self.preprocessor_url:
            self.preprocessor = hub.KerasLayer(self.preprocessor_url)
        self.model = hub.KerasLayer(self.model_url)

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """

        :param data: a 1d array of string type in size `B`
        :param args:
        :param kwargs:
        :return: an ndarray in size `B x D`
        """
        if self.preprocessor:
            data = self.preprocessor(data)
        return self.model(data)
