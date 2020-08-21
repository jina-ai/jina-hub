from jina.flow import Flow

from .. import Sentencizer


def test_sentencier_en():
    sentencizer = Sentencizer()
    text = 'It is a sunny day!!!! When Andy comes back, we are going to the zoo.'
    crafted_chunk_list = sentencizer.craft(text, 0)
    assert len(crafted_chunk_list) == 2


def test_sentencier_en_new_lines():
    """
    New lines are also considered as a separator.
    """
    sentencizer = Sentencizer()
    text = 'It is a sunny day!!!! When Andy comes back,\n' \
           'we are going to the zoo.'
    crafted_chunk_list = sentencizer.craft(text, 0)
    assert len(crafted_chunk_list) == 3


def test_sentencier_en_float_numbers():
    """
    Separators in float numbers, URLs, emails, abbreviations (like 'Mr.')
    are not taking into account.
    """
    sentencizer = Sentencizer()
    text = 'With a 0.99 probability this sentence will be ' \
           'tokenized in 2 sentences.'
    crafted_chunk_list = sentencizer.craft(text, 0)
    assert len(crafted_chunk_list) == 2


def test_sentencier_en_trim_spaces():
    """
    Trimming all spaces at the beginning an end of the chunks.
    Keeping extra spaces inside chunks.
    Ignoring chunks with only spaces.
    """
    sentencizer = Sentencizer()
    text = '  This ,  text is...  . Amazing !!'
    chunks = [i['text'] for i in sentencizer.craft(text, 0)]
    locs = [i['location'] for i in sentencizer.craft(text, 0)]
    assert chunks, ["This ,  text is..." == "Amazing"]
    assert text[locs[0][0]:locs[0][1]], '  This  ==   text is...'
    assert text[locs[1][0]:locs[1][1]] == ' Amazing'

    def validate(req):
        assert req.docs[0].chunks[0].text, 'This  ==   text is...'
        assert req.docs[0].chunks[1].text == 'Amazing'

    f = Flow().add(uses='!Sentencizer')
    with f:
        f.index_lines(['  This ,  text is...  . Amazing !!'], output_fn=validate, callback_on_body=True)


def test_sentencier_cn():
    sentencizer = Sentencizer()
    text = '今天是个大晴天！安迪回来以后，我们准备去动物园。'
    crafted_chunk_list = sentencizer.craft(text, 0)
    # Sentencizer does not work for chinese because string.printable does not contain Chinese characters
    assert len(crafted_chunk_list) == 0
