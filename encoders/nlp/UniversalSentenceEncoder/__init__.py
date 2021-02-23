__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseTFEncoder

UNIVERSAL_SENTENCE_ENCODER = 'https://tfhub.dev/google/universal-sentence-encoder/4'
MODEL_ENCODER_CMLM = "https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1"
PREPROCESOR_CMLM = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2"


class UniversalSentenceEncoder(BaseTFEncoder):
    """
    Encode an 1d array of string in size `B` into an ndarray in size `B x D`

    The ndarray potentially is BatchSize x (Channel x Height x Width)

    :class:`UniversalSentenceEncoder` is a encoder based on the Universal Sentence
    Encoder family (https://tfhub.dev/google/collections/universal-sentence-encoder/1).

    :param model_url: the url of the model (TensorFlow Hub).
        For supported models see family overview:
        https://tfhub.dev/google/collections/universal-sentence-encoder/1)
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    class GeneralEncoder:
        """
        General universal sentence encoder.

        Loads a model and it encodes from an 1d array of string
        in size `B` into an ndarray in size `B x D`.

        :param model_url: the url of the model (TensorFlow Hub).
            For supported models see family overview:
            https://tfhub.dev/google/collections/universal-sentence-encoder/1)
        """

        def __init__(self, model_url: str):
            """Set GeneralEncoder Constructor."""
            import tensorflow_hub as hub
            self.model_url = model_url
            self.model = hub.load(self.model_url)

        def encode(self, data: 'np.ndarray') -> 'np.ndarray':
            """
            Encode data into an ndarray

            :param data: a 1d array of string type in size `B`
            :param args:  Additional positional arguments
            :param kwargs: Additional keyword arguments
            :return: an ndarray in size `B x D`
            """
            return self.model(data)

    class CMLMEncoder:
        """
        Private class encoder to represent a CMLM universal Sentence Encoder family.

        It encodes data from an 1d array of string in size `B`
        into an ndarray in size `B x D`.
        (https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1).
        """

        def __init__(self):
            """Set CMLMEncoder Constructor."""
            import tensorflow_text as text
            import tensorflow_hub as hub
            self.bert_preprocessor = hub.KerasLayer(PREPROCESOR_CMLM)
            self.encoder = hub.KerasLayer(MODEL_ENCODER_CMLM)

        def encode(self, data: 'np.ndarray') -> 'np.ndarray':
            """
            Encode data into an ndarray

            :param data: a 1d array of string type in size `B`
            :param args:  Additional positional arguments
            :param kwargs: Additional keyword arguments
            :return: an ndarray in size `B x D`
            """
            return self.encoder(self.bert_preprocessor(data))['default']

    def __init__(
            self,
            model_url: str = UNIVERSAL_SENTENCE_ENCODER,
            *args,
            **kwargs):
        """Set UniversalSentenceEncoder constructor."""
        super().__init__(*args, **kwargs)
        self.model_url = model_url

    def post_init(self):
        """Load Sentence encoder model"""
        self.to_device()

        if self.model_url == MODEL_ENCODER_CMLM:
            self.sentence_encoder = UniversalSentenceEncoder.CMLMEncoder()
        else:
            self.sentence_encoder = UniversalSentenceEncoder.GeneralEncoder(self.model_url)

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        Encode an array of string in size `B` into an ndarray in size `B x D`

        The ndarray potentially is BatchSize x (Channel x Height x Width)

        :param data: a 1d array of string type in size `B`
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: an ndarray in size `B x D`
        """
        return self.sentence_encoder.encode(data)
