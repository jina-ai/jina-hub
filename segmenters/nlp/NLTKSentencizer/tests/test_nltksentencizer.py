import pytest
from jina import Flow

from .. import NLTKSentencizer


@pytest.mark.parametrize(
    'language, expected_len, expected_first_sentence, expected_second_sentence, text',
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
    language, expected_len, expected_first_sentence, expected_second_sentence, text
):
    """
    Test multiple scenarios with various languages
    """
    if language:
        sentencizer = NLTKSentencizer(language)
    else:
        # default language is English
        sentencizer = NLTKSentencizer()
    segmented = sentencizer.segment(text)
    sentences = [i['text'] for i in segmented]
    assert len(segmented) == expected_len
    assert sentences[0] == expected_first_sentence
    assert sentences[1] == expected_second_sentence


def test_locations():
    """Test simple logics regarding the ``location`` key of sentences returned by the sentencizer"""
    sentencizer = NLTKSentencizer()
    text = (
        "This is a sentence. Here's another sentence. One more sentence?    Aaand, yes, one more! \n"
        "Lastly, this one is the last sentence."
    )
    sentences = sentencizer.segment(text)

    # first sentence should start at the first index or later
    assert sentences[0]['location'][0] >= 0
    # last sentence can not end at an index greater than the length of text
    assert sentences[-1]['location'][-1] <= len(text)
    # sentences beginning and ending indeces cannot overlap
    for i in range(1, len(sentences)):
        assert sentences[i]['location'][0] > sentences[i - 1]['location'][-1]


def test_nltk_sentencizer_unsupported_language():
    """Unsupported and/or mis-spelt languages must raise a LookUp error"""
    with pytest.raises(LookupError):
        NLTKSentencizer('eng')
    with pytest.raises(LookupError):
        NLTKSentencizer('abcd')


def test_offset():
    """Test that last offset is the same as the length of sentences minus one"""
    sentencizer = NLTKSentencizer()
    text = '  This ,  text is...  . Amazing !!'
    sentences = sentencizer.segment(text)
    assert len(sentences) - 1 == sentences[-1]['offset']


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
    sentencizer = NLTKSentencizer(language='turkish')
    # turkish abbreviations include dot, and they should not be segmented
    assert len(sentencizer.segment(text)) == 1


def test_nltk_sentencizer_in_flow():
    def validate(req):
        assert req.docs[0].chunks[0].text == 'Today is a good day.'
        assert req.docs[0].chunks[1].location[0] == 21

    f = Flow().add(uses='!NLTKSentencizer')
    with f:
        f.index_lines(
            ["Today is a good day. Can't wait for tomorrow!"],
            on_done=validate,
            callback_on_body=True,
            line_format='csv',
        )
