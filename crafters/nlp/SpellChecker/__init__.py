__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

from jina.executors.decorators import single
from jina.executors.crafters import BaseCrafter


class SpellChecker(BaseCrafter):

    # TODO: What about the tokenizer function option, should be passed to the transform?

    def __init__(self,
                 model_path: str,
                 *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.model_path = model_path

    def post_init(self):
        super().post_init()
        import pickle

        self.model = None
        if self.model_path:
            with open(self.model_path, 'rb') as model_file:
                self.model = pickle.load(model_file)

    @single
    def craft(self, text: str, *args, **kwargs):
        return dict(text=self.model.transform(text))
