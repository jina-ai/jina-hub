__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Optional

from jina.executors.decorators import single
from jina.executors.crafters import BaseCrafter


class PySpellChecker(BaseCrafter):

    """
    :class:`PySpellChecker` wraps pyspellchecker (https://github.com/barrust/pyspellchecker) library
    to provide spelling correction capacity as a crafter in Jina

    :param language: The language of the dictionary to load or None \
            for no dictionary. Supported languages are `en`, `es`, `de`, `fr`, \
            `pt` and `ru`. Defaults to `en`. A list of languages may be \
            provided and all languages will be loaded.
    :param local_dictionary: he path to a locally stored word \
            frequency dictionary; if provided, no language will be loaded
    :param distance: The edit distance to use. Defaults to 2.
    :param case_sensitive: Flag to use a case sensitive dictionary or \
            not, only available when not using a language dictionary.
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments

    """

    def __init__(self,
                 language: str = 'en',
                 local_dictionary: Optional[str] = None,
                 distance: int = 2,
                 case_sensitive: bool = False,
                 *args, **kwargs):
        """Set constructor."""
        super().__init__(*args, **kwargs)
        self.language = language
        self.local_dictionary = local_dictionary
        self.distance = distance
        self.case_sensitive = case_sensitive

    def post_init(self):
        from spellchecker import SpellChecker

        super().post_init()
        self.speller = SpellChecker(language=self.language,
                                    local_dictionary=self.local_dictionary,
                                    distance=self.distance,
                                    case_sensitive=self.case_sensitive)

    @single
    def craft(self, text: str, *args, **kwargs):
        """
        Craft sentences correcting misspelled words

        :param text: The text to be corrected
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: A dictionary with the extracted text
        """
        words = self.speller.split_words(text)
        corrected_text = ' '.join([self.speller.correction(word) for word in words])
        return dict(text=corrected_text)
