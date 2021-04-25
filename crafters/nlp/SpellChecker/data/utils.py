import nltk
import os
import inspect


def get_evaluation(n_test=100):
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    lines = open(os.path.join(current_dir, 'evaluation.txt')).readlines()
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
