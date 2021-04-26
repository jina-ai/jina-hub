import os
import pytest

from jina.excepts import PretrainedModelFileDoesNotExist
from .. import SpellChecker
from ..pyngramspell import PyNgramSpell

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_missing_model_path():
    wrong_path_vectorizer = os.path.join(
        cur_dir, '/model/non_existant.pickle'
    )
    with pytest.raises(PretrainedModelFileDoesNotExist):
        _ = SpellChecker(wrong_path_vectorizer)


@pytest.fixture
def input_training_data():
    correct_sentences = [
        ['they', 'can', 'go', 'quite', 'fast'],
        ['there', 'were', 'the', 'new', 'Japanese', 'Honda']
    ]
    return [' '.join(x) for x in correct_sentences]


@pytest.fixture
def model_path(tmpdir, input_training_data):
    model_path = os.path.join(str(tmpdir), 'tmp_model.pickle')
    speller = PyNgramSpell()
    speller.fit(input_training_data)
    speller.save(model_path)
    yield model_path


def test_spell_checker_correct(model_path):
    input_text = ['they can go quite fast', 'they can go', 'there japanese honda', 'new fast honda',
                  'the new fast, japanese']

    spell_checker = SpellChecker(model_path=model_path)
    crafted_docs = spell_checker.craft(input_text)

    assert len(crafted_docs) == len(input_text)
    for crafted_doc, expected in zip(crafted_docs, input_text):
        assert crafted_doc['text'] == expected
