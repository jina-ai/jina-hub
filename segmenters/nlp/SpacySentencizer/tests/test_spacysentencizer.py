import pytest
import spacy
import numpy as np

from .. import SpacySentencizer


@pytest.fixture
def multilingual_model_name():
    return 'xx_sent_ud_sm'


@pytest.mark.parametrize(
    'multilingual_test_inputs,multilingual_test_expected',
    [
        ('It is a sunny day!!!! When Andy comes back, we are going to the zoo.', 2),
        ('ini adalah sebuah kalimat. ini adalah sebuah kalimat lain.', 2),
        ('今天是个大晴天！安迪回来以后，我们准备去动物园。', 1),
    ],
)
def test_multilingual_sentencizer(multilingual_model_name, multilingual_test_inputs, multilingual_test_expected):
    # xx_sent_ud_sm does not have DependencyParser model, ignoring use_default_segmenter=True
    segmenter = SpacySentencizer(multilingual_model_name, use_default_segmenter=False)
    text = multilingual_test_inputs
    expected_num_of_splits = multilingual_test_expected
    docs_chunks = segmenter.segment(np.stack([text, text]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == expected_num_of_splits


def test_unsupported_lang(tmp_path):
    dummy1 = spacy.blank('xx')
    dummy1_dir_path = tmp_path / 'xx1'
    dummy1.to_disk(dummy1_dir_path)
    dummy2 = spacy.blank('xx')
    dummy2_dir_path = tmp_path / 'xx2'
    dummy2.to_disk(dummy2_dir_path)
    # No available language
    with pytest.raises(IOError):
        SpacySentencizer('abcd')

    # Language does not have DependencyParser should thrown an error
    # when try to use default segmenter
    with pytest.raises(ValueError):
        SpacySentencizer(dummy1_dir_path, use_default_segmenter=True)

    # And should be fine when 'parser' pipeline is added
    dummy1.add_pipe('parser')
    dummy1.to_disk(dummy1_dir_path)
    SpacySentencizer(dummy1_dir_path, use_default_segmenter=True)

    # Language does not have SentenceRecognizer should thrown an error
    # when try to use non default segmenter
    with pytest.raises(ValueError):
        SpacySentencizer(dummy2_dir_path, use_default_segmenter=False)

    # And should be fine when 'senter' pipeline is added
    dummy2.add_pipe('senter')
    dummy2.to_disk(dummy2_dir_path)
    SpacySentencizer(dummy2_dir_path, use_default_segmenter=False)
