__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseTFEncoder


UNIVERSAL_SENTENCE_ENCODER_CMLM = "https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1"
UNIVERSAL_SENTENCE_ENCODER = 'https://tfhub.dev/google/universal-sentence-encoder/4'


class UniversalSentenceEncoder(BaseTFEncoder):
    """
    :class:`UniversalSentenceEncoder` is a encoder based on the Universal Sentence
    Encoder family (https://tfhub.dev/google/collections/universal-sentence-encoder/1).
    It encodes data from an 1d array of string in size `B` into an ndarray in size `B x D`.
    """

    def __init__(
            self,
            model_url: str = UNIVERSAL_SENTENCE_ENCODER,
            * args,
            **kwargs):
        """
        :param model_url: the url of the model (TensorFlow Hub). For supported models see
                          family overview: https://tfhub.dev/google/collections/universal-sentence-encoder/1)
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.model_url = model_url

    def post_init(self):
        self.to_device()
        import tensorflow_hub as hub
        if UNIVERSAL_SENTENCE_ENCODER_CMLM == self.model_url:
            @as_ndarray
            def __customized_model(data: 'np.ndarray') -> 'np.ndarray':
                preprocessor = hub.KerasLayerc(
                    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2")
                encoder = hub.KerasLayer(UNIVERSAL_SENTENCE_ENCODER_CMLM)
                return encoder(preprocessor(data))["default"]
            self.model = __customized_model

        else:
            self.model = hub.load(self.model_url)

    @ batching
    @ as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """

        :param data: a 1d array of string type in size `B`
        :param args:
        :param kwargs:
        :return: an ndarray in size `B x D`
        """
        return self.model(data)
