__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional

import tensorflow_hub as hub
import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseTFEncoder


UNIVERSAL_SENTENCE_ENCODER = 'https://tfhub.dev/google/universal-sentence-encoder/4'
MODEL_ENCODER_CMLM = "https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1"
PREPROCESOR_CMLM = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2"


class CMLMEncoder:
    """
    :class:`CMLMEncoder` is an private class  encoder to represent a CMLM
    Universal Sentence Encoder family
    (https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1).
    It encodes data from an 1d array of string in size `B` into an ndarray in size `B x D`.
    """

    def __init__(self):
        import tensorflow_text as text
        self.bert_preprocessor = hub.KerasLayer(PREPROCESOR_CMLM)
        self.encoder = hub.KerasLayer(MODEL_ENCODER_CMLM)

    def encode(self, data: 'np.ndarray') -> 'np.ndarray':
        """
        :param data: a 1d array of string type in size `B`
        :param args:
        :param kwargs:
        :return: an ndarray in size `B x D`
        """
        return self.encoder(self.bert_preprocessor(data))['default']


class GeneralEncoder:
    """
    :class:`GeneralEncoder` is general universal sentence encoder
    which load a model and it encodes from an 1d array of string
    in size `B` into an ndarray in size `B x D`.
    """

    def __init__(self, model_url: str):
        self.model_url = model_url
        self.model = hub.load(self.model_url)

    def encode(self, data: 'np.ndarray') -> 'np.ndarray':
        """
        :param data: a 1d array of string type in size `B`
        :param args:
        :param kwargs:
        :return: an ndarray in size `B x D`
        """
        return self.model(data)


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

        if self.model_url == MODEL_ENCODER_CMLM:
            self.sentence_encoder = CMLMEncoder()
        else:
            self.sentence_encoder = GeneralEncoder(self.model_url)

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        :param data: a 1d array of string type in size `B`
        :param args:
        :param kwargs:
        :return: an ndarray in size `B x D`
        """
        return self.sentence_encoder.encode(data)
