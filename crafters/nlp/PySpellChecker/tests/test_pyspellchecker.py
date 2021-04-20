import numpy as np

import pytest

from .. import PySpellChecker


@pytest.mark.parametrize('language, input, correction',
                         [
                             ('en', 'wrng sentenca', 'wrong sentence'),
                             ('en', 'wrng,sentenca', 'wrong sentence'),
                             ('es', 'frasi increcta', 'frase incorrecta'),
                             ('es', 'frasi,increcta', 'frase incorrecta'),
                         ]
                         )
def test_pyspellchecker(language, input, correction):
    crafter = PySpellChecker(language=language, distance=3)
    crafted_docs = crafter.craft(np.stack([input, input]))
    assert len(crafted_docs) == 2
    for doc in crafted_docs:
        assert doc['text'] == correction
