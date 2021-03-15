import numpy as np
import pytest

from .. import NLTKSentencizer


@pytest.mark.parametrize(
    'language, expected_len, expected_first_chunk, expected_second_chunk, text',
    [
        (
            None,
            2,
            'Today is a good day.',
            "Can't wait for tomorrow!",
            "Today is a good day. Can't wait for tomorrow!",
        ),
        (
            'french',
            2,
            "Aujourd'hui est un bon jour.",
            'Je ne peux pas attendre demain!',
            "Aujourd'hui est un bon jour. Je ne peux pas attendre demain!",
        ),
        (
            'german',
            2,
            'Heute ist ein guter Tag.',
            'Ich kann nicht auf morgen warten!',
            'Heute ist ein guter Tag. Ich kann nicht auf morgen warten!',
        ),
        (
            'italian',
            2,
            'Oggi è una buona giornata.',
            "Non vedo l'ora che arrivi domani!",
            "Oggi è una buona giornata. Non vedo l'ora che arrivi domani!",
        ),
        (
            'russian',
            2,
            'Сегодня хороший день.',
            'Не могу дождаться завтра!',
            'Сегодня хороший день. Не могу дождаться завтра!',
        ),
        (
            'greek',
            2,
            'Σήμερα είναι μια καλή μέρα.',
            'Δεν μπορώ να περιμένω αύριο!',
            'Σήμερα είναι μια καλή μέρα. Δεν μπορώ να περιμένω αύριο!',
        ),
        (
            'norwegian',
            2,
            'I dag er en god dag.',
            'Gleder meg ikke til i morgen!',
            'I dag er en god dag. Gleder meg ikke til i morgen!',
        ),
    ],
)
def test_nltksentencizer(
    language, expected_len, expected_first_chunk, expected_second_chunk, text
):
    """
    Test multiple scenarios with various languages
    """
    if language:
        segmenter = NLTKSentencizer(language)
    else:
        # default language is English
        segmenter = NLTKSentencizer()
    docs_chunks = segmenter.segment(np.stack([text, text]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == expected_len
        assert chunks[0]['text'] == expected_first_chunk
        assert chunks[1]['text'] == expected_second_chunk


def test_locations():
    """Test simple logics regarding the ``location`` key of sentences returned by the sentencizer"""
    segmenter = NLTKSentencizer()
    text = (
        "This is a sentence. Here's another sentence. One more sentence?    Aaand, yes, one more! \n"
        "Lastly, this one is the last sentence."
    )
    docs_chunks = segmenter.segment(np.stack([text, text]))

    for chunks in docs_chunks:
        # first sentence should start at the first index or later
        assert chunks[0]['location'][0] >= 0
        # last sentence can not end at an index greater than the length of text
        assert chunks[-1]['location'][-1] <= len(text)
        # sentences beginning and ending indeces cannot overlap
        for i in range(1, len(chunks)):
            assert chunks[i]['location'][0] > chunks[i - 1]['location'][-1]


def test_nltk_sentencizer_unsupported_language():
    """Unsupported and/or mis-spelt languages must raise a LookUp error"""
    with pytest.raises(LookupError):
        NLTKSentencizer('eng')
    with pytest.raises(LookupError):
        NLTKSentencizer('abcd')


def test_offset():
    """Test that last offset is the same as the length of sentences minus one"""
    segmenter = NLTKSentencizer()
    text = '  This ,  text is...  . Amazing !!'
    docs_chunks = segmenter.segment(np.stack([text, text]))
    for chunks in docs_chunks:
        assert len(chunks) - 1 == chunks[-1]['offset']


@pytest.mark.parametrize(
    'text',
    [
        ("T.C. Sağlık Bakalığı\'nın son açıklamasına göre, koronavirüs daha bitmedi."),
        ('Doç. Dr. Özkonuk, açıklama yaptı.'),
        ('3 No.lu madde onaylandı.'),
    ],
)
def test_turkish_abbreviations(text):
    """Check that Turkish sentences that include dots in abbreviations do not separate on those"""
    segmenter = NLTKSentencizer(language='turkish')
    # turkish abbreviations include dot, and they should not be segmented
    docs_chunks = segmenter.segment(np.stack([text, text]))
    for chunks in docs_chunks:
        assert len(chunks) == 1
