import pytest
import spacy

from .. import SpacySentencizer


def test_multilingual_sentencizer_with_en_lang():
    sentencizer = SpacySentencizer()
    text = "It is a sunny day!!!! When Andy comes back, we are going to the zoo."
    crafted_chunk_list = sentencizer.segment(text, 0)
    assert len(crafted_chunk_list) == 2


def test_multilingual_sentencizer_with_id_lang():
    sentencizer = SpacySentencizer()
    text = "ini adalah sebuah kalimat. ini adalah sebuah kalimat lain."
    crafted_chunk_list = sentencizer.segment(text, 0)
    assert len(crafted_chunk_list) == 2


def test_sentencier_cn():
    sentencizer = SpacySentencizer()
    text = "今天是个大晴天！安迪回来以后，我们准备去动物园。"
    crafted_chunk_list = sentencizer.segment(text, 0)
    assert len(crafted_chunk_list) == 1


def test_unsupported_lang():
    dummy1 = spacy.blank("xx")
    dummy1.to_disk("/tmp/xx")
    dummy2 = spacy.blank("ja")
    dummy2.to_disk("/tmp/ja")
    # No available language
    with pytest.raises(IOError):
        SpacySentencizer("abcd")

    # Language does not have DependencyParser should thrown an error
    # when try to use default segmenter
    with pytest.raises(ValueError):
        SpacySentencizer("/tmp/xx", use_default_segmenter=True)

    # And should be fine when "parser" pipeline is added
    dummy1.add_pipe("parser")
    dummy1.to_disk("/tmp/xx")
    SpacySentencizer("/tmp/xx", use_default_segmenter=True)

    # Language does not have SentenceRecognizer should thrown an error
    # when try to use non default segmenter
    with pytest.raises(ValueError):
        SpacySentencizer("/tmp/ja", use_default_segmenter=False)

    # And should be fine when "senter" pipeline is added
    dummy2.add_pipe("senter")
    dummy2.to_disk("/tmp/ja")
    SpacySentencizer("/tmp/ja", use_default_segmenter=False)
