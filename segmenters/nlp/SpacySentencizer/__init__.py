from typing import Dict, List

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

    :param lang: pre-trained spaCy language pipeline,
        by default "xx_sent_ud_sm".
    :param use_default_segmenter: boolean to determine which segmentation algorithm to use,
        by default False.
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    SPACY_COMPONENTS = [
        "tagger",
        "parser",
        "ner",
        "senter",
        "entity_linker",
        "textcat",
        "entity_ruler",
        "merge_noun_chunks",
        "merge_entities",
        "merge_subtokens",
        "lemmatizer",
        "attribute_ruler",
    ]

    def __init__(self, lang: str = "xx_sent_ud_sm", use_default_segmenter: bool = False, *args, **kwargs):
        """Set constructor."""
        super().__init__(*args, **kwargs)

        import spacy

        try:
            self.spacy_model = spacy.load(lang)
            # Disable everything as we only requires certain pipelines to turned on.
            ignored_components = []
            for comp in self.SPACY_COMPONENTS:
                try:
                    self.spacy_model.disable_pipe(comp)
                except Exception:
                    ignored_components.append(comp)
            self.logger.info(f"Ignoring {ignored_components} pipelines as it does not available on the model package.")
        except IOError:
            self.logger.error(
                f"spaCy model for language {lang} can't be found. Please install by referring to the "
                "official page https://spacy.io/usage/models"
            )
            raise

        if use_default_segmenter:
            try:
                self.spacy_model.enable_pipe("parser")
            except ValueError:
                self.logger.error(
                    f"Parser for language {lang} can't be found. The default sentence segmenter requires "
                    "DependencyParser to be trained. Please refer to https://spacy.io/api/sentencizer for more clarity."
                )
                raise
        else:
            try:
                self.spacy_model.enable_pipe("senter")
            except ValueError:
                self.logger.error(
                    f"SentenceRecognizer is not available for language {lang}. Please refer to "
                    "https://github.com/explosion/spaCy/issues/6615 for training your own recognizer."
                )
                raise

    def segment(self, text: str, *args, **kwargs) -> List[Dict]:
        """
        Split the text into sentences.

        :param text: the raw text
        :return: a list of chunk dicts with the split sentences
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments

        """
        sentences = self.spacy_model(text).sents
        results = [dict(text=sent, offset=idx, weight=1.0) for idx, sent in enumerate(sentences)]
        return results
