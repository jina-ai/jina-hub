import numpy as np

import pytest

from .. import Sentencizer


def test_sentencizer_en():
    segmenter = Sentencizer()
    text = 'It is a sunny day!!!! When Andy comes back, we are going to the zoo.'
    docs_chunks = segmenter.segment(np.stack([text, text]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == 2


def test_sentencizer_en_new_lines():
    """
    New lines are also considered as a separator.
    """
    segmenter = Sentencizer()
    text = 'It is a sunny day!!!! When Andy comes back,\n' \
           'we are going to the zoo.'
    docs_chunks = segmenter.segment(np.stack([text, text]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == 3


def test_sentencizer_en_float_numbers():
    """
    Separators in float numbers, URLs, emails, abbreviations (like 'Mr.')
    are not taking into account.
    """
    segmenter = Sentencizer()
    text = 'With a 0.99 probability this sentence will be ' \
           'tokenized in 2 sentences.'
    docs_chunks = segmenter.segment(np.stack([text, text]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == 2


@pytest.mark.parametrize(
    'expected_len, expected_first_chunk, expected_second_chunk, sentence',
    [(2, 'إنه يوم مشمس!!!!', 'عندما يعود آندي ، سنذهب إلى حديقة الحيوانات.',
      'إنه يوم مشمس!!!! عندما يعود آندي ، سنذهب إلى حديقة الحيوانات.'),
     (2, '今天是个大晴天！！！！', '安迪回来以后，我们准备去动物园。',
      '今天是个大晴天！！！！安迪回来以后，我们准备去动物园。'),
     (2, 'Het is een zonnige dag!!!!', 'Als Andy terugkomt, gaan we naar de dierentuin.',
      'Het is een zonnige dag!!!! Als Andy terugkomt, gaan we naar de dierentuin.'),
     (2, "C'est une journée ensoleillée!!!!", "Quand Andy revient, nous allons au zoo.",
      "C'est une journée ensoleillée!!!! Quand Andy revient, nous allons au zoo."),
     (2, 'Es ist ein sonniger Tag!!!!', 'Wenn Andy zurückkommt, gehen wir in den Zoo.',
      'Es ist ein sonniger Tag!!!! Wenn Andy zurückkommt, gehen wir in den Zoo.'),
     (2, 'यह एक धूपवाला दिन है!!!!', 'जब एंडी वापस आता है, हम चिड़ियाघर जा रहे हैं।',
      'यह एक धूपवाला दिन है!!!! जब एंडी वापस आता है, हम चिड़ियाघर जा रहे हैं।'),
     (2, '晴れた日です!!!!', 'アンディが戻ってきたら、動物園に行きます。',
      '晴れた日です!!!!アンディが戻ってきたら、動物園に行きます。'),
     (2, '화창한 날입니다!!!!', 'Andy가 돌아 오면 우리는 동물원에갑니다.',
      '화창한 날입니다!!!! Andy가 돌아 오면 우리는 동물원에갑니다.'),
     (2, 'É um dia ensolarado!!!!', 'Quando Andy voltar, vamos ao zoológico.',
      'É um dia ensolarado!!!! Quando Andy voltar, vamos ao zoológico.'),
     (2, 'Это солнечный день!!!!', 'Когда Энди вернется, мы идем в зоопарк.',
      'Это солнечный день!!!! Когда Энди вернется, мы идем в зоопарк.'),
     (2, '¡¡¡¡Es un día soleado!!!!', 'Cuando Andy regrese, iremos al zoológico.',
      '¡¡¡¡Es un día soleado!!!! Cuando Andy regrese, iremos al zoológico.'),
     (2, 'It is ä suñny dáy!!!!', 'When Andy comes back, we are going to the 动物园!',
      'It is ä suñny dáy!!!! When Andy comes back, we are going to the 动物园!')],
)
def test_sentencizer_multi_lang(expected_len, expected_first_chunk, expected_second_chunk, sentence):
    """
    Test multiple scenarios with various languages
    """
    segmenter = Sentencizer()
    docs_chunks = segmenter.segment(np.stack([sentence, sentence]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == expected_len
        assert chunks[0]['text'] == expected_first_chunk
        assert chunks[1]['text'] == expected_second_chunk
