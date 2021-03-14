# NLTKSentencizer

Segments text into sentences using NLTK.


if __name__ == '__main__':
    tokenizer = NLTKSentencizer('turkish')
    s = 'Benim adım Gizem. Kanalıma hoşgeldiniz arkadaşlaarr!!! T.C. Sağlık Bakamlığı\'nın son açıklamasına göre, koronavirüs daha bitmedi. ' \
        '"SON DAKİKA" diye çıkan haberleri duymaktan da bıktık...' \
        '\n T.C. Sağlık Bakamlığı\'nın son açıklamasına göre, koronavirüs daha bitmedi. Gerçekten bitmedi!!!'

    sents = tokenizer.segment(s)
    for s in sents:
        print(s)
    s = "Today is a good day. Can't wait for tomorrow!"
    tok = NLTKSentencizer()
    a = tok.segment(s)
    print(a)

"""
Sentencizer output:
from segmenters.nlp.Sentencizer import Sentencizer
sentencizer = Sentencizer()
sentencizer.segment(s)
[{'text': 'Benim adım Gizem.', 'offset': 0, 'weight': 1.0, 'location': [0, 17]}, {'text': 'Kanalıma hoşgeldiniz arkadaşlaarr!!!', 'offset': 1, 'weight': 1.0, 'location': [17, 54]}, {'text': 'T.', 'offset': 2, 'weight': 1.0, 'location': [54, 57]}, {'text': 'C.', 'offset': 3, 'weight': 1.0, 'location': [57, 59]}, {'text': "Sağlık Bakamlığı'nın son açıklamasına göre, koronavirüs daha bitmedi.", 'offset': 4, 'weight': 1.0, 'location': [59, 129]}, {'text': '"SON DAKİKA" diye çıkan haberleri duymaktan da bıktık...', 'offset': 5, 'weight': 1.0, 'location': [129, 186]}]

"""
 