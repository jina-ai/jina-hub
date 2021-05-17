__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

from jina.executors.decorators import single
from jina.executors.crafters import BaseCrafter
from jina.excepts import PretrainedModelFileDoesNotExist

cur_dir = os.path.dirname(os.path.abspath(__file__))


class SpellChecker(BaseCrafter):

    def __init__(self,
                 model_path: str = os.path.join(cur_dir, 'model/model.pickle'),
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path

    def post_init(self):
        super().post_init()
        import pickle
        from .pyngramspell import PyNgramSpell, BKTree

        self.model = None
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as model_file:
                self.model = pickle.load(model_file)
        else:
            raise PretrainedModelFileDoesNotExist(
                f'{self.model_path} not found, cannot find a fitted spell checker'
            )

    @single
    def craft(self, text: str, *args, **kwargs):
        return dict(text=' '.join(self.model.transform(text)))
