import pytest

from jina.excepts import PretrainedModelFileDoesNotExist
from .. import SpellChecker
from ..pyngramspell import PyNgramSpell


def test_missing_model_path():
    wrong_path_vectorizer = os.path.join(
        cur_dir, '/model/non_existant.pickle'
    )
    with pytest.raises(PretrainedModelFileDoesNotExist):
        _ = SpellChecker(wrong_path_vectorizer)


@pytest.fixture
def input_training_data():
    return ['a a']


@pytest.fixture
def model_path(tmpdir, input_training_data):
    model_path = os.path.join(str(tmpdir), 'tmp_model.pickle')
    speller = PyNgramSpell()
    speller.fit(input_training_data)
    speller.save(model_path)
    yield model_path


def test_spell_checker_correct(model_path):
    input_text = ['correct sentence', 'this sentence is correct and found in the model']

    spell_checker = SpellChecker(model_path=model_path)
    crafted_docs = spell_checker.craft(input_text)

    assert len(crafted_docs) == 2
    for crafted_doc, expected in zip(crafted_docs, input_text):
        assert crafted_doc['text'] == expected
