__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseTorchEncoder

class SpacyTextEncoder(BaseTorchEncoder):
    """
    :class:`SpacyTextEncoder` encodes data from a `np.ndarray` (of strings) of length `BatchSize` into
    a `np.ndarray` of shape `Batchsize x EmbeddingDimension`.

    :param lang: pre-trained spaCy language pipeline (model name HashEmbedCNN by default for tok2vec), `en_core_web_sm`
        by default. Allows models `en_core_web_md`, `en_core_web_lg`, `en_core_web_trf`. Refer https://spacy.io/models/en.
    :param use_default_encoder: if True will use parser component,
        otherwise tok2vec implementation will be chosen,
        by default False.
    :param args: Additional positional arguments.
    :param kwargs: Additional positional arguments.
    """

    SPACY_COMPONENTS = [
        'tagger',
        'parser',
        'ner',
        'senter',
        'tok2vec',
        'lemmatizer',
        'attribute_ruler',
    ]
    def __init__(self, lang: str = 'en_core_web_sm', use_default_encoder: bool = False, *args, **kwargs):
        """Set constructor."""
        super().__init__(*args, **kwargs)
        self.lang = lang
        self.use_default_encoder = use_default_encoder

    def post_init(self):
        """Load a model from spacy specified in `lang`. """
        import spacy

        try:
            self.spacy_model = spacy.load(self.lang)
            # Disable everything as we only requires certain pipelines to turned on.
            ignored_components = []
            for comp in self.SPACY_COMPONENTS:
                try:
                    self.spacy_model.disable_pipe(comp)
                except Exception:
                    ignored_components.append(comp)
            self.logger.info(f'Ignoring {ignored_components} pipelines as it does not available on the model package.')
        except IOError:
            self.logger.error(
                f'spaCy model for language {self.lang} can not be found. Please install by referring to the '
                'official page https://spacy.io/usage/models.'
            )
            raise

        if self.use_default_encoder:
            try:
                self.spacy_model.enable_pipe('parser')
            except ValueError:
                self.logger.error(
                    f'Parser for language {self.lang} can not be found. The default sentence encoder requires'
                    'DependencyParser to be trained. Please refer to https://spacy.io/api/tok2vec for more clarity.'
                )
                raise
        else:
            try:
                self.spacy_model.enable_pipe('tok2vec')
            except ValueError:
                self.logger.error(
                    f'TokenToVector is not available for language {self.lang}. Please refer to'
                    'https://github.com/explosion/spaCy/issues/6615 for training your own recognizer.'
                )
                raise

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        Transform a `np.ndarray` of strings of length `BatchSize` into
        a `np.ndarray` of shape `Batchsize x EmbeddingDimension`.
        Calculate word embeddings by applying spacy's tok2vec component
        model in a sentence wise manner.

        :param data: A `np.ndarray` of strings. Each string is a sentence.
        :param args: Additional positional arguments.
        :param kwargs: Additional positional arguments.
        :return: A `BachSize x EmbeddingSize` numpy `ndarray`.
        """
        embedding = []
        for sent_data in data:
            processed_data = self.spacy_model(str(sent_data))
            result = [self.tensor2array(token.tensor) for token in processed_data]
            embedding.append(result)
        return np.vstack(embedding)

    def tensor2array(self, tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.cpu().numpy() if self.on_gpu else tensor.numpy()
