__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseTorchEncoder


class LaserEncoder(BaseTorchEncoder):
    """
    Encode an array of string in size `B` into an ndarray in size `B x D`

    The ndarray potentially is BatchSize x (Channel x Height x Width)

    :class:`LaserEncoder` is a encoder based on Facebook Research's LASER
    (Language-Agnostic SEntence Representations) to compute multilingual
    sentence embeddings: https://github.com/facebookresearch/LASER
    :param path_to_bpe_codes: path to bpe codes from Laser.
        Defaults to Laser.DEFAULT_BPE_CODES_FILE.
    :param path_to_bpe_vocab: path to bpe vocabs from Laser.
        Defaults to Laser.DEFAULT_BPE_VOCAB_FILE.
    :param path_to_encoder: path to the encoder from Laser.
        Defaults to Laser.DEFAULT_ENCODER_FILE.
    :param language: language of the text. Defaults to english(en).
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(
            self,
            path_to_bpe_codes: str = None,
            path_to_bpe_vocab: str = None,
            path_to_encoder: str = None,
            language: str = 'en',
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        from laserembeddings import Laser
        self._path_to_bpe_codes = path_to_bpe_codes or Laser.DEFAULT_BPE_CODES_FILE
        self._path_to_bpe_vocab = path_to_bpe_vocab or Laser.DEFAULT_BPE_VOCAB_FILE
        self._path_to_encoder = path_to_encoder or Laser.DEFAULT_ENCODER_FILE
        self.language = language.lower()

    def post_init(self):
        """Load LaserEncoder model"""
        from laserembeddings import Laser
        self.model = Laser(
            bpe_codes=self._path_to_bpe_codes,
            bpe_vocab=self._path_to_bpe_vocab,
            encoder=self._path_to_encoder,
        )
        self.to_device(self.model.bpeSentenceEmbedding.encoder.encoder)

    @batching
    @as_ndarray
    def encode(self, content: "np.ndarray", *args, **kwargs) -> "np.ndarray":
        """
         Encode ``Document`` content into an ndarray in size `B x D`.

         B is the `Batch size` and `D` the dimension.

        :param content: a 1d array of string type in size `B`
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: an ndarray in size `B x D`.
        """
        return self.model.embed_sentences(content, lang=self.language)
