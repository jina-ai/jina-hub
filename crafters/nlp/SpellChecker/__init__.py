__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

from jina.executors.decorators import single
from jina.executors.crafters import BaseCrafter


class SpellChecker(BaseCrafter):

    def __init__(self,
                 tika_ocr_strategy: str = 'ocr_only',
                 tika_extract_inline_images: str = 'true',
                 tika_ocr_language: str = 'eng',
                 tika_request_timeout: int = 600,
                 *args, **kwargs):
    super().__init__(*args, **kwargs)

    def post_init(self):
        super().post_init()
        load_spellchecker()

    def load_spellchecker(self):
    	import pickle

    	# load model
    	pass


