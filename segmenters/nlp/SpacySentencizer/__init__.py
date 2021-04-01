from typing import Dict, List

from jina.executors.decorators import single
from jina.executors.segmenters import BaseSegmenter


class SpacySentencizer(BaseSegmenter):
    """
    :class:`SpacySentencizer` split the text on the doc-level
    into sentences on the chunk-level with spaCy as the backend.
    The splitting technique is divided into two different approaches:
    1) Default Segmenter: Utilizes dependency parsing to determine the rule for
        splitting the text. For more info please refer to https://spacy.io/api/sentencizer.
    2) Machine Learning-based Segmenter: Utilizes SentenceRecognizer model trained
        on a dedicated data created for sentence segmentation. For more info please
        refer to https://spacy.io/api/sentencerecognizer.

    :param lang: pre-trained spaCy language pipeline
    :param use_default_segmenter: if True will use dependency based sentence segmentation,
        otherwise ml-based implementation will be chosen,
        by default False.
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    SPACY_COMPONENTS = [
        'tagger',
        'parser',
        'ner',
        'senter',
        'entity_linker',
        'textcat',
        'entity_ruler',
        'merge_noun_chunks',
        'merge_entities',
        'merge_subtokens',
        'lemmatizer',
        'attribute_ruler',
    ]

    def __init__(self, lang: str, use_default_segmenter: bool = False, *args, **kwargs):
        """Set constructor."""
        super().__init__(*args, **kwargs)
        self.lang = lang
        self.use_default_segmenter = use_default_segmenter

    def post_init(self):
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

        if self.use_default_segmenter:
            try:
                self.spacy_model.enable_pipe('parser')
            except ValueError:
                self.logger.error(
                    f'Parser for language {self.lang} can not be found. The default sentence segmenter requires'
                    'DependencyParser to be trained. Please refer to https://spacy.io/api/sentencizer for more clarity.'
                )
                raise
        else:
            try:
                self.spacy_model.enable_pipe('senter')
            except ValueError:
                self.logger.error(
                    f'SentenceRecognizer is not available for language {self.lang}. Please refer to'
                    'https://github.com/explosion/spaCy/issues/6615 for training your own recognizer.'
                )
                raise

    @single
    def segment(self, text: str, *args, **kwargs) -> List[Dict]:
        """
        Split the text into sentences.

        :param text: the raw text
        :return: a list of chunk dicts with the split sentences
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments

        """
        sentences = self.spacy_model(str(text)).sents
        results = [dict(text=sent) for sent in sentences]
        return results
