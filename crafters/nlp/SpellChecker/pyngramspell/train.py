import os

from pathlib import Path
from pyngramspell import PyNgramSpell

cur_dir = os.path.dirname(os.path.abspath(__file__))


def get_evaluation(n_test=100):
    lines = open(os.path.join(cur_dir, 'evaluation.txt')).readlines()
    data = []

    import nltk

    for line in lines:
        indexes = []
        tokenize_line = nltk.word_tokenize(line)
        misspelled_line = tokenize_line.copy()
        corrected_line = tokenize_line.copy()
        for index, word in enumerate(tokenize_line):
            if '|' in word:
                misspelled_line[index] = word.split('|')[0]
                corrected_line[index] = word.split('|')[1]
                indexes.append(index)
        data.append({'original': misspelled_line, 'corrected': corrected_line, 'indexes': indexes})

    test = data[:n_test]
    train = data[n_test:]
    return train, test


def train_model(model_path: str = os.path.join(cur_dir, '../model/model.pickle')):
    import nltk
    nltk.download('punkt')
    train, test = get_evaluation()
    y_tr = [x['corrected'] for x in train]
    train_data = [' '.join(x) for x in y_tr]
    speller = PyNgramSpell()
    speller.fit(train_data)
    assert ' '.join(speller.transform('the maan went to the store')) == 'the man went to the store'
    os.makedirs(Path(model_path).parent, exist_ok=True)
    speller.save(model_path)


if __name__ == '__main__':
    train_model()
